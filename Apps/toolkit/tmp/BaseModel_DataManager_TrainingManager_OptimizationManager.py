from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Literal, Optional, Union, List, Tuple
from loguru import logger
from pydantic import BaseModel, Field
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchinfo import summary as torch_summary
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

class BaseModel(nn.Module, ABC):
    """
    Enhanced abstract base class for all models in the framework.
    """

    def __init__(self):
        super().__init__()
        self.model_type: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.task_type: Optional[str] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        pass

    @abstractmethod
    def inference(self, x: torch.Tensor, **kwargs) -> Any:
        """
        Perform inference on the input.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the loss for the model.
        """
        pass

    @abstractmethod
    def compute_prediction(self, outputs: torch.Tensor, **kwargs) -> Any:
        """
        Compute predictions from the model outputs.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        """
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024),
            "layers": [f"{name}: {module.__class__.__name__}" for name, module in self.named_children()],
        }

    def set_model_attributes(self, model_type: str, task_type: str, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """
        Set multiple model attributes at once.
        """
        self.model_type = model_type
        self.task_type = task_type
        self.input_shape = input_shape
        self.output_shape = output_shape

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze specified layers of the model.
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """
        Unfreeze specified layers of the model.
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        """
        Get all trainable parameters of the model.
        """
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def load_pretrained_weights(self, weights_path: str, strict: bool = True) -> None:
        """
        Load pretrained weights into the model with option for non-strict loading.
        """
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict, strict=strict)

    def get_layer_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Get the output of a specific layer given an input tensor.
        """
        for name, module in self.named_modules():
            x = module(x)
            if name == layer_name:
                return x
        raise ValueError(f"Layer {layer_name} not found in the model.")

    def summary(self, input_size: Optional[Tuple[int, ...]] = None, **kwargs) -> None:
        """
        Print a summary of the model architecture with additional options.
        """
        if input_size is None and self.input_shape is None:
            raise ValueError("Please provide input_size or set input_shape for the model.")
        
        input_size = input_size or self.input_shape
        torch_summary(self, input_size=input_size, **kwargs)

    def to_onnx(self, file_path: str, input_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> None:
        """
        Export the model to ONNX format with additional options.
        """
        if input_shape is None and self.input_shape is None:
            raise ValueError("Please provide input_shape or set input_shape for the model.")
        
        input_shape = input_shape or self.input_shape
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(self, dummy_input, file_path, verbose=True, **kwargs)

    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Get a specific layer of the model by name.
        """
        for name, module in self.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in the model.")

    def apply_weight_initialization(self, init_func: callable) -> None:
        """
        Apply a weight initialization function to all the model's parameters.
        """
        self.apply(init_func)

    def get_activation_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Get activation maps for a specific layer.
        """
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.get_layer(layer_name).register_forward_hook(get_activation(layer_name))
        self.forward(x)
        return activation[layer_name]

class CustomDataset(Dataset, ABC):
    """
    Abstract base class for custom datasets.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample from the dataset.
        """
        pass

class DataParams(BaseModel):
    """
    Pydantic model for all data-related parameters.
    """
    data_path: Union[str, List[str]] = Field(..., description="Path(s) to the dataset")
    task_type: str = Field(..., description="Type of task (e.g., 'vision', 'nlp', 'tabular')")
    batch_size: int = Field(32, description="Batch size for data loading")
    num_workers: int = Field(4, description="Number of workers for data loading")
    shuffle: bool = Field(True, description="Whether to shuffle the dataset")
    validation_split: float = Field(0.2, description="Fraction of data to use for validation")
    test_split: float = Field(0.1, description="Fraction of data to use for testing")
    transforms: Optional[Dict[str, Any]] = Field(None, description="Transform configurations")

    class Config:
        arbitrary_types_allowed = True

class DataManager(ABC):
    """
    Abstract base class for data management.
    """
    def __init__(self, params: DataParams) -> None:
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.setup_transforms()

    @abstractmethod
    def load_data(self) -> Any:
        """
        Load the data from the specified path(s).
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the loaded data.
        """
        pass

    @abstractmethod
    def create_dataset(self, data: Any, is_train: bool = True) -> Dataset:
        """
        Create a dataset from the preprocessed data.
        """
        pass

    @abstractmethod
    def setup_transforms(self) -> None:
        """
        Set up transforms based on the task type and specified transforms.
        """
        pass

    @abstractmethod
    def apply_transforms(self, data: Any, is_train: bool = True) -> Any:
        """
        Apply the specified transforms to the data.
        """
        pass

    def setup(self) -> None:
        """
        Set up the datasets for training, validation, and testing.
        """
        data = self.load_data()
        preprocessed_data = self.preprocess_data(data)

        # Split data into train+val and test
        if self.params.test_split > 0:
            train_val_data, test_data = train_test_split(
                preprocessed_data, 
                test_size=self.params.test_split, 
                random_state=42
            )
        else:
            train_val_data, test_data = preprocessed_data, None

        # Split train+val into train and val
        if self.params.validation_split > 0:
            train_size = int((1 - self.params.validation_split) * len(train_val_data))
            val_size = len(train_val_data) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.create_dataset(train_val_data, is_train=True),
                [train_size, val_size]
            )
        else:
            self.train_dataset = self.create_dataset(train_val_data, is_train=True)
            self.val_dataset = None

        if test_data is not None:
            self.test_dataset = self.create_dataset(test_data, is_train=False)

    def get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create and return data loaders for train, validation, and test datasets.
        """
        if not self.train_dataset:
            raise ValueError("Datasets are not set up. Call setup() first.")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle,
            num_workers=self.params.num_workers
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_workers
            )
        
        test_loader = None
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_workers
            )

        return train_loader, val_loader, test_loader

class MetricsManager(ABC):
    def __init__(self, metrics_config: List[Dict[str, Any]], device: str = 'cpu') -> None:
        super().__init__()
        self.metrics: Dict[str, torchmetrics.Metric] = {}
        self.device = device
        self._initialize_metrics(metrics_config)

    def _initialize_metrics(self, metrics_config: List[Dict[str, Any]]) -> None:
        for metric_info in metrics_config:
            metric_name = metric_info['name']
            metric_class = getattr(torchmetrics, metric_info['class'])
            metric_params = metric_info.get('params', {})
            self.metrics[metric_name] = metric_class(**metric_params).to(self.device)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        for metric in self.metrics.values():
            metric.update(outputs, targets)

    def compute(self) -> Dict[str, torch.Tensor]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def get_metric(self, name: str) -> torchmetrics.Metric:
        return self.metrics[name]

    def add_metric(self, name: str, metric: torchmetrics.Metric) -> None:
        self.metrics[name] = metric.to(self.device)

    def remove_metric(self, name: str) -> None:
        if name in self.metrics:
            del self.metrics[name]

    def to(self, device: str) -> 'MetricsManager':
        self.device = device
        for metric in self.metrics.values():
            metric.to(device)
        return self

class TrainingParams(BaseModel):
    """
    Pydantic model for all training parameters.
    """
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", description="Device to use for training")
    learning_rate: float = Field(1e-3, description="Learning rate for optimization")
    batch_size: int = Field(32, description="Batch size for training")
    epochs: int = Field(10, description="Number of training epochs")
    use_tensorboard: bool = Field(False, description="Whether to use TensorBoard for logging")
    optimizer: str = Field("adam", description="Optimizer to use (e.g., 'adam', 'sgd')")
    use_scheduler: bool = Field(False, description="Whether to use a learning rate scheduler")
    scheduler_type: Optional[str] = Field(None, description="Type of learning rate scheduler to use")
    early_stopping: bool = Field(False, description="Whether to use early stopping")
    patience: int = Field(5, description="Patience for early stopping")
    clip_grad_norm: Optional[float] = Field(None, description="Clip gradient norm if specified")
    use_mixed_precision: bool = Field(False, description="Whether to use mixed precision training")
    checkpoint_dir: str = Field("checkpoints", description="Directory to save model checkpoints")
    log_interval: int = Field(100, description="Interval for logging training progress")
    val_interval: int = Field(1, description="Interval for validation (in epochs)")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for specific tasks")

    class Config:
        arbitrary_types_allowed = True

class TrainingManager(ABC):
    """
    Abstract base class for managing the training process.
    """

    def __init__(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader],
        test_data_loader: Optional[DataLoader],
        model: nn.Module,
        loss_fn: nn.Module,
        train_params: TrainingParams,
        metrics_config: List[Dict[str, Any]]
    ) -> None:
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.model = model
        self.loss_fn = loss_fn
        self.train_params = train_params
        
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler() if self.train_params.use_scheduler else None
        self.scaler = GradScaler() if self.train_params.use_mixed_precision else None
        
        if self.train_params.use_tensorboard:
            self.writer = SummaryWriter()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.metrics_manager = MetricsManager(metrics_config, device=self.train_params.device)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self.train_params.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.train_params.learning_rate)
        elif self.train_params.optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.train_params.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.train_params.optimizer}")

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.train_params.scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        elif self.train_params.scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.train_params.scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.train_params.epochs)
        return None

    def to_device(self) -> None:
        self.model.to(self.train_params.device)
        self.loss_fn.to(self.train_params.device)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def train_step(self, batch: Iterable[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.train_params.device), targets.to(self.train_params.device)

        self.optimizer.zero_grad()

        if self.train_params.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss: torch.Tensor = self.loss_fn(outputs, targets)
            
            self.scaler.scale(loss).backward()
            if self.train_params.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            if self.train_params.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
            self.optimizer.step()

        self.metrics_manager.update(outputs, targets)
        return {'loss': loss.item()}


    @abstractmethod
    def val_step(self, batch: Iterable[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        inputs, targets = batch
        inputs, targets = inputs.to(self.train_params.device), targets.to(self.train_params.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        self.metrics_manager.update(outputs, targets)
        return {'loss': loss.item()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(self.train_data_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{self.train_params.epochs}") as pbar:
            for i, batch in enumerate(self.train_data_loader):
                step_results = self.train_step(batch)
                total_loss += step_results['loss']
                
                if i % self.train_params.log_interval == 0:
                    metrics = self.metrics_manager.compute()
                    metrics['loss'] = step_results['loss']
                    self._log_progress('train', epoch, i, metrics)
                
                pbar.update(1)
                pbar.set_postfix(loss=f"{step_results['loss']:.4f}")
        
        metrics = self.metrics_manager.compute()
        metrics['loss'] = total_loss / num_batches
        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(self.val_data_loader)
        
        with torch.no_grad():
            for batch in self.val_data_loader:
                step_results = self.val_step(batch)
                total_loss += step_results['loss']
        
        metrics = self.metrics_manager.compute()
        metrics['loss'] = total_loss / num_batches
        self._log_progress('val', epoch, 0, metrics)
        return metrics

    def train_loop(self) -> None:
        for epoch in range(self.train_params.epochs):
            train_results = self.train_epoch(epoch)
            
            if self.val_data_loader and epoch % self.train_params.val_interval == 0:
                val_results = self.validate(epoch)
                
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_results['loss'])
                    else:
                        self.scheduler.step()
                
                if val_results['loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['loss']
                    self.save_model(os.path.join(self.train_params.checkpoint_dir, 'best_model.pth'))
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.train_params.early_stopping and self.patience_counter >= self.train_params.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            self.save_model(os.path.join(self.train_params.checkpoint_dir, f'model_epoch_{epoch+1}.pth'))

    def test_loop(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_data_loader)
        
        with torch.no_grad():
            for batch in self.test_data_loader:
                step_results = self.val_step(batch)
                total_loss += step_results['loss']
        
        avg_loss = total_loss / num_batches
        logger.info(f"Test Loss: {avg_loss:.4f}")
        return {'loss': avg_loss}

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_params': self.train_params.dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_params = TrainingParams(**checkpoint['train_params'])
        logger.info(f"Model loaded from {path}")

    def _log_progress(self, phase: str, epoch: int, step: int, metrics: Dict[str, Union[float, torch.Tensor]]) -> None:
        if self.train_params.use_tensorboard:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"{phase}/{metric_name}", metric_value, epoch * len(self.train_data_loader) + step)
        
        log_str = f"{phase.capitalize()} Epoch {epoch+1}, Step {step}: "
        log_str += ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        logger.info(log_str)

class OptimizationManager:
    """
    Class for managing model optimization and profiling.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def profile(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Profile the model's performance and memory usage.
        """
        logger.info("Profiling model performance and memory usage...")
        self.model.eval()
        with torch.no_grad():
            summary_data = torch_summary(self.model, input_data=input_tensor, verbose=0)
        return summary_data

    def benchmark(self, input_tensor: torch.Tensor, num_runs: int = 100) -> float:
        """
        Benchmark the model's forward pass.
        """
        logger.info(f"Benchmarking model with {num_runs} runs...")
        self.model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(num_runs):
                _ = self.model(input_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            avg_time = elapsed_time / num_runs
            
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        return avg_time