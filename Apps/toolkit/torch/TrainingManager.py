# My code starts from here
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict, Any, Callable
from torch.distributed.algorithms.join import Join
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import enable_wrap, wrap, size_based_auto_wrap_policy
import time
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from packaging import version as LooseVersion

class ParallelStrategy:
    DATA_PARALLEL: str = "data_parallel"
    MODEL_PARALLEL: str = "model_parallel"
    PIPELINE_PARALLEL: str = "pipeline_parallel"
    DDP: str = "ddp"
    TORCH_DATA_PARALLEL: str = "torch_data_parallel" 
    FSDP: str = "fsdp"

class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        rank: int = 0,
        backend: str = 'nccl',
        parallel_strategy: str = ParallelStrategy.DATA_PARALLEL,
        split_size: int = 20,
        enable_checkpointing: bool = True,
        enable_ddp: bool = False,
        enable_join: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        delay_all_reduce_named_params: Optional[List[Tuple[str, nn.Parameter]]] = None,
        param_to_hook_all_reduce: Optional[nn.Parameter] = None,
        enable_ddp_optimizer: bool = False,
        enable_fsdp: bool = False,
        fsdp_cpu_offload: bool = False,
        fsdp_auto_wrap_policy: Optional[Callable] = None,
        fsdp_backward_prefetch: Optional[BackwardPrefetch] = None,
        fsdp_sharding_strategy: str = "FULL_SHARD",
        use_enhanced_fsdp: bool = False,
    ) -> None:
        self.model: nn.Module = model
        self.world_size: int = world_size
        self.rank: int = rank
        self.backend: str = backend
        self.parallel_strategy: str = parallel_strategy
        self.split_size: int = split_size
        self.device: torch.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.enable_checkpointing: bool = enable_checkpointing
        
        self.enable_ddp: bool = enable_ddp
        self.enable_join: bool = enable_join
        self.enable_ddp_optimizer: bool = enable_ddp_optimizer
        
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        self.delay_all_reduce_named_params = delay_all_reduce_named_params
        self.param_to_hook_all_reduce = param_to_hook_all_reduce

        self.enable_fsdp = enable_fsdp
        self.fsdp_cpu_offload = fsdp_cpu_offload
        self.fsdp_auto_wrap_policy = fsdp_auto_wrap_policy
        self.fsdp_backward_prefetch = fsdp_backward_prefetch
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.use_enhanced_fsdp = use_enhanced_fsdp
        
        if self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                gradient_as_bucket_view=self.gradient_as_bucket_view,
                static_graph=self.static_graph,
                delay_all_reduce_named_params=self.delay_all_reduce_named_params,
                param_to_hook_all_reduce=self.param_to_hook_all_reduce,
            )
            self.register_ddp_comm_hook()
            
            if self.enable_ddp_optimizer:
                self.setup_ddp_optimizer()
        
        if self.parallel_strategy == ParallelStrategy.FSDP and self.enable_fsdp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = self.wrap_fsdp_model()
        
        if self.parallel_strategy == ParallelStrategy.MODEL_PARALLEL:
            self.model = ModelParallelModule(self.model, self.world_size)
        elif self.parallel_strategy == ParallelStrategy.PIPELINE_PARALLEL:
            self.model = PipelineParallelModule(self.model, self.world_size, self.split_size)
        elif self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        elif self.parallel_strategy == ParallelStrategy.TORCH_DATA_PARALLEL:
            self.model = nn.DataParallel(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        
        if self.world_size > 1 and not (self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp):
            self.setup_distributed()
    
    def setup_ddp_optimizer(self):
        # Initialize TorchDynamo DDPOptimizer
        torch._dynamo.config.optimize_ddp = True
        torch._dynamo.config.log_level = "INFO"  # Set to "DEBUG" for more detailed logs
        self.model = torch.compile(self.model)

    def register_ddp_comm_hook(self):
        def ddp_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            fut = torch.distributed.all_reduce(bucket.buffer()).get_future()
            return fut

        self.model.register_comm_hook(state=None, hook=ddp_comm_hook)

    def wrap_fsdp_model(self) -> nn.Module:
        if self.use_enhanced_fsdp:
            fsdp_config = {
                "cpu_offload": CPUOffload(offload_params=self.fsdp_cpu_offload),
                "backward_prefetch": self.fsdp_backward_prefetch,
                "sharding_strategy": getattr(ShardingStrategy, self.fsdp_sharding_strategy),
            }
            
            if self.fsdp_auto_wrap_policy:
                auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=100000)
                with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
                    wrapped_model = wrap(self.model, auto_wrap_policy=auto_wrap_policy)
            else:
                wrapped_model = FSDP(self.model, **fsdp_config)
        else:
            # Original FSDP implementation
            fsdp_config = {
                "cpu_offload": CPUOffload(offload_params=self.fsdp_cpu_offload),
            }
            
            if self.fsdp_auto_wrap_policy:
                with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
                    wrapped_model = wrap(self.model)
            else:
                wrapped_model = FSDP(self.model, **fsdp_config)
        
        return wrapped_model

    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, path: str) -> None:
        if not self.enable_checkpointing:
            return
        
        if self.rank == 0:  # Only save checkpoint on the main process
            if self.parallel_strategy == ParallelStrategy.FSDP:
                if self.use_enhanced_fsdp:
                    # Enhanced FSDP checkpointing
                    full_state_dict = FSDP.full_state_dict(self.model)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': full_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, path)
                else:
                    # Original FSDP checkpointing
                    FSDP.save_model_checkpoint(self.model, path)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'parallel_strategy': self.parallel_strategy,
                    'world_size': self.world_size,
                    'split_size': self.split_size,
                    'enable_ddp_optimizer': self.enable_ddp_optimizer,
                    'enable_fsdp': self.enable_fsdp,
                    'fsdp_cpu_offload': self.fsdp_cpu_offload,
                }
                torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
            
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
        if not self.enable_checkpointing:
            return 0, 0.0
        
        if self.parallel_strategy == ParallelStrategy.FSDP:
            if self.use_enhanced_fsdp:
                # Enhanced FSDP checkpoint loading
                checkpoint = torch.load(path, map_location=self.device)
                FSDP.load_state_dict(self.model, checkpoint['model_state_dict'])
                if optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return checkpoint['epoch'], checkpoint['loss']
            else:
                # Original FSDP checkpoint loading
                FSDP.load_model_checkpoint(self.model, path)
                return 0, 0.0  # Return default values for now
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)
        
        if checkpoint['parallel_strategy'] != self.parallel_strategy:
            raise ValueError("Checkpoint parallel strategy does not match current strategy")
        
        if checkpoint['world_size'] != self.world_size:
            raise ValueError("Checkpoint world size does not match current world size")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore DDPOptimizer state
        self.enable_ddp_optimizer = checkpoint.get('enable_ddp_optimizer', False)
        if self.enable_ddp_optimizer:
            self.setup_ddp_optimizer()
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']

    def export_model(self, path: str) -> None:
        """Export the trained model in TorchScript format."""
        if self.rank == 0:  # Only export from the main process
            # If using DDPOptimizer, we need to handle the compiled model differently
            if self.enable_ddp_optimizer:
                # Export the original model, not the compiled one
                original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                model_scripted = torch.jit.script(original_model)
            else:
                model_scripted = torch.jit.script(self.model)
            model_scripted.save(path)
            print(f"Model exported to {path}")

    def setup_distributed(self) -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)

    def partition_dataset(self, dataset: Dataset, uneven: bool = False) -> DataLoader:
        if uneven:
            # Create uneven partitions for testing Join context manager
            partition_sizes: List[float] = [1.0 / self.world_size * (1 + 0.1 * i) for i in range(self.world_size)]
            partition_sizes = [size / sum(partition_sizes) for size in partition_sizes]  # Normalize
        else:
            partition_sizes: List[float] = [1.0 / self.world_size] * self.world_size
        
        partition: DataPartitioner = DataPartitioner(dataset, partition_sizes)
        partition = partition.use(self.rank)
        batch_size: int = 128 // self.world_size  # Adjust batch size based on number of processes
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    def average_gradients(self) -> None:
        size: float = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    def train(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        if self.parallel_strategy == ParallelStrategy.FSDP:
            self.train_fsdp(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.enable_join:
            self.train_with_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        else:
            self.train_without_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)

    def _train_epoch(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for i, (data, target) in enumerate(train_set):
            if i % 100 == 0 and isinstance(self.model, DDP):
                # Use no_sync every 100 iterations to accumulate gradients locally
                with self.model.no_sync():
                    loss = self.train_step(data, target, optimizer, criterion)
            else:
                loss = self.train_step(data, target, optimizer, criterion)
            
            epoch_loss += loss
            num_samples += len(data)

        return epoch_loss, num_samples

    def train_without_join(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def train_with_join(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        if self.parallel_strategy not in [ParallelStrategy.DATA_PARALLEL, ParallelStrategy.DDP]:
            raise ValueError("Join context manager is only supported for data parallel and DDP strategies.")

        train_set: DataLoader = self.partition_dataset(dataset, uneven=True)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            with Join([self.model]):
                epoch_loss, num_samples = self._train_epoch(train_set, optimizer, criterion)
            
            # Synchronize loss and num_samples across all ranks
            epoch_loss = torch.tensor([epoch_loss], device=self.device)
            num_samples = torch.tensor([num_samples], device=self.device)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
            
            avg_loss = epoch_loss.item() / num_samples.item()
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def train_step(self, data: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        data, target = data.to(self.device), target.to(self.device)
        optimizer.zero_grad()
        output: torch.Tensor = self.model(data)
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        
        if self.world_size > 1 and self.parallel_strategy == ParallelStrategy.DATA_PARALLEL and not self.enable_ddp:
            self.average_gradients()
        
        optimizer.step()
        return loss.item()

    def train_data_parallel(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss: float = 0.0
            start_time = time.time()
            for data, target in train_loader:
                loss: float = self.train_step(data, target, optimizer, criterion)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_loader)
            end_time = time.time()
            print(f'Epoch {epoch}: loss={avg_loss:.4f}, time={end_time - start_time:.2f}s')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")
    
    def train_pipeline_parallel(self, data: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        splits: iter = iter(data.split(self.split_size, dim=0))
        split_targets: iter = iter(target.split(self.split_size, dim=0))
        
        total_loss: float = 0
        optimizer.zero_grad()
        
        for split, split_target in zip(splits, split_targets):
            split, split_target = split.to(self.device), split_target.to(self.device)
            output: torch.Tensor = self.model(split)
            loss: torch.Tensor = criterion(output, split_target)
            total_loss += loss.item()
            loss.backward()
        
        if self.world_size > 1:
            self.average_gradients()
        
        optimizer.step()
        return total_loss
    
    def check_bf16_support(self) -> bool:
        bf16_ready = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and dist.is_nccl_available()
            and LooseVersion(torch.distributed.nccl.version()) >= (2, 10)
        )
        return bf16_ready
    
    def train_fsdp(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_fsdp(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_fsdp(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for data, target in train_set:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples

    @staticmethod
    def init_process(rank: int, size: int, fn: callable, backend: str = 'gloo') -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)

    @classmethod
    def run_distributed(cls, world_size: int, fn: callable) -> None:
        mp.spawn(cls.init_process, args=(world_size, fn), nprocs=world_size, join=True)

    def benchmark(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        start_time = time.time()
        self.train(dataset, num_epochs, optimizer, criterion)
        end_time = time.time()
        
        total_samples = len(dataset) * num_epochs
        throughput = total_samples / (end_time - start_time)
        
        fsdp_logging_data = {}
        if isinstance(self.model, FSDP):
            fsdp_logging_data = {
                "full_params_size": self.model.module.numel() * 4 / 1e9,  # in GB
                "sharded_params_size": sum(p.numel() for p in self.model.parameters()) * 4 / 1e9,  # in GB
                "cpu_offload": self.fsdp_cpu_offload,
                "sharding_strategy": self.fsdp_sharding_strategy,
                "backward_prefetch": self.fsdp_backward_prefetch is not None,
            }
        
        return {
            "strategy": self.parallel_strategy,
            "throughput": throughput,
            "total_time": end_time - start_time,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(self.device) / 1e9,  # in GB
            "fsdp_logging_data": fsdp_logging_data,
            "fsdp_enabled": self.enable_fsdp,
            "enhanced_fsdp": self.use_enhanced_fsdp,
        }

class DataPartitioner:
    def __init__(self, data: Dataset, sizes: List[float]) -> None:
        self.data: Dataset = data
        self.partitions: List[List[int]] = self.partition_dataset(sizes)

    def partition_dataset(self, sizes: List[float]) -> List[List[int]]:
        data_len: int = len(self.data)
        indexes: List[int] = list(range(data_len))
        return [indexes[int(sum(sizes[:i])*data_len):int(sum(sizes[:i+1])*data_len)] for i in range(len(sizes))]

    def use(self, partition: int) -> Dataset:
        return torch.utils.data.Subset(self.data, self.partitions[partition])

class ModelParallelModule(nn.Module):
    def __init__(self, module: nn.Module, num_gpus: int):
        super().__init__()
        self.num_gpus: int = num_gpus
        self.layers: List[nn.Sequential] = self._split_model(module)

    def _split_model(self, module: nn.Module) -> List[nn.Sequential]:
        layers: List[nn.Module] = list(module.children())
        split_size: int = len(layers) // self.num_gpus
        return [nn.Sequential(*layers[i * split_size:(i + 1) * split_size]).to(f'cuda:{i}')
                for i in range(self.num_gpus)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x.to(f'cuda:{i}'))
        return x

class PipelineParallelModule(nn.Module):
    def __init__(self, module: nn.Module, num_gpus: int, split_size: int):
        super().__init__()
        self.num_gpus: int = num_gpus
        self.split_size: int = split_size
        self.layers: List[nn.Sequential] = self._split_model(module)

    def _split_model(self, module: nn.Module) -> List[nn.Sequential]:
        layers: List[nn.Module] = list(module.children())
        split_size: int = len(layers) // self.num_gpus
        return [nn.Sequential(*layers[i * split_size:(i + 1) * split_size]).to(f'cuda:{i}')
                for i in range(self.num_gpus)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits: iter = iter(x.split(self.split_size, dim=0))
        s_next: torch.Tensor = next(splits)
        s_prev: torch.Tensor = self.layers[0](s_next).to('cuda:1')
        ret: List[torch.Tensor] = []

        for s_next in splits:
            s_prev = self.layers[1](s_prev)
            ret.append(s_prev)
            s_prev = self.layers[0](s_next).to('cuda:1')

        s_prev = self.layers[1](s_prev)
        ret.append(s_prev)

        return torch.cat(ret)