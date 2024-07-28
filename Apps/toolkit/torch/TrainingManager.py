# My code starts from here
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict, Any
from torch.distributed.algorithms.join import Join
import time

class ParallelStrategy:
    DATA_PARALLEL: str = "data_parallel"
    MODEL_PARALLEL: str = "model_parallel"
    PIPELINE_PARALLEL: str = "pipeline_parallel"
    DDP: str = "ddp"
    TORCH_DATA_PARALLEL: str = "torch_data_parallel" 

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
        enable_join: bool = False
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

    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, path: str) -> None:
        if not self.enable_checkpointing:
            return
        
        if self.rank == 0:  # Only save checkpoint on the main process
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'parallel_strategy': self.parallel_strategy,
                'world_size': self.world_size,
                'split_size': self.split_size
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
        if not self.enable_checkpointing:
            return 0, 0.0
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)
        
        if checkpoint['parallel_strategy'] != self.parallel_strategy:
            raise ValueError("Checkpoint parallel strategy does not match current strategy")
        
        if checkpoint['world_size'] != self.world_size:
            raise ValueError("Checkpoint world size does not match current world size")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']

    def export_model(self, path: str) -> None:
        """Export the trained model in TorchScript format."""
        if self.rank == 0:  # Only export from the main process
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
        if self.enable_join:
            self.train_with_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        else:
            self.train_without_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)

    def train_without_join(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss: float = 0.0
            for data, target in train_set:
                if self.parallel_strategy == ParallelStrategy.PIPELINE_PARALLEL:
                    loss: float = self.train_pipeline_parallel(data, target, optimizer, criterion)
                else:
                    loss: float = self.train_step(data, target, optimizer, criterion)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_set)
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
            epoch_loss: float = 0.0
            num_samples: int = 0
            
            with Join([self.model]):
                for data, target in train_set:
                    loss: float = self.train_step(data, target, optimizer, criterion)
                    epoch_loss += loss
                    num_samples += len(data)
            
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
        
        return {
            "strategy": self.parallel_strategy,
            "throughput": throughput,
            "total_time": end_time - start_time,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(self.device) / 1e9  # in GB
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