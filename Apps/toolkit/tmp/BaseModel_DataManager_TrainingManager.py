from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Literal, Optional, Union, List, Tuple
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field, field_validator
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.hooks import RemovableHandle
from sklearn.model_selection import train_test_split
from torchinfo import summary as torch_summary
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

class BaseModel(nn.Module, ABC):
    """
    Enhanced abstract base class for all models in the framework.
    Provides a common interface and utility methods for various model types.
    """

    def __init__(self):
        super().__init__()
        self.model_type: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.task_type: Optional[str] = None
        self._layer_shapes: Dict[Union[int, str], Tuple[int, ...]] = {}
        self._hooks: List[RemovableHandle] = []
        self._device: Optional[torch.device] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def inference(self, x: torch.Tensor, **kwargs) -> Any:
        """Perform inference on the input."""
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the loss for the model."""
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def compute_prediction(self, outputs: torch.Tensor, **kwargs) -> Any:
        """Compute predictions from the model outputs."""
        raise NotImplementedError("Subclass must implement abstract method")

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
        
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "layers": [f"{name}: {module.__class__.__name__}" for name, module in self.named_children()],
        }

    def set_model_attributes(
        self,
        model_type: str,
        task_type: str,
        input_shape: Optional[Tuple[int, ...]],
        output_shape: Optional[Tuple[int, ...]]
    ) -> None:
        """Set multiple model attributes at once."""
        self.model_type = model_type
        self.task_type = task_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        if input_shape:
            self._layer_shapes[0] = input_shape
        if output_shape:
            self._layer_shapes[-1] = output_shape

    @property
    def device(self) -> torch.device:
        """Get the device on which the model is currently loaded."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specified layers of the model."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        logger.info(f"Frozen layers: {layer_names}")

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """Unfreeze specified layers of the model."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
        logger.info(f"Unfrozen layers: {layer_names}")

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        """Get all trainable parameters of the model."""
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def load_pretrained_weights(self, weights_path: str, strict: bool = True) -> None:
        """Load pretrained weights into the model."""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=strict)
            logger.info(f"Pretrained weights loaded from {weights_path}")
        except FileNotFoundError:
            logger.error(f"Weights file not found: {weights_path}")
            raise
        except RuntimeError as e:
            logger.error(f"Error loading pretrained weights: {str(e)}")
            raise

    def get_layer_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Get the output of a specific layer given an input tensor."""
        output = {}

        def hook(module, input, out):
            output['value'] = out

        layer = self.get_layer(layer_name)
        handle = layer.register_forward_hook(hook)
        self.forward(x)
        handle.remove()

        if 'value' not in output:
            raise ValueError(f"Layer {layer_name} did not produce any output.")
        return output['value']

    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a specific layer of the model by name."""
        for name, module in self.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in the model")

    def get_shape(self, layer: Union[int, str], dummy_input: Optional[torch.Tensor] = None) -> Tuple[int, ...]:
        """Get the shape of a specific layer."""
        if dummy_input is not None:
            self.compute_shapes(dummy_input.shape)

        if isinstance(layer, int):
            if layer == 0:
                return self.input_shape if self.input_shape is not None else tuple()
            elif layer == -1:
                return self.output_shape if self.output_shape is not None else tuple()
            elif layer in self._layer_shapes:
                return self._layer_shapes[layer]
        elif isinstance(layer, str):
            for name, module in self.named_modules():
                if name == layer:
                    return tuple(module.weight.shape) if hasattr(module, 'weight') else tuple()

        raise ValueError(f"Shape for layer {layer} not found or not computed yet.")

    def compute_shapes(self, input_shape: Tuple[int, ...]) -> None:
        """Compute and store the shapes of all layers in the model."""
        def hook(module, input, output):
            self._layer_shapes[len(self._layer_shapes)] = tuple(output.shape[1:])

        self._layer_shapes.clear()
        self._layer_shapes[0] = input_shape

        for name, module in self.named_modules():
            if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential):
                self._hooks.append(module.register_forward_hook(hook))

        dummy_input = torch.randn(input_shape)
        self(dummy_input)

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        self.input_shape = input_shape
        self.output_shape = self._layer_shapes[max(self._layer_shapes.keys())]

    def summary(self, input_size: Optional[Tuple[int, ...]] = None, **kwargs) -> None:
        """Print a summary of the model architecture with additional options."""
        from torchinfo import summary as torch_summary

        if input_size is None and self.input_shape is None:
            raise ValueError("Please provide input_size or set input_shape for the model.")
        
        input_size = input_size or self.input_shape
        torch_summary(self, input_size=input_size, **kwargs)

    def apply_weight_initialization(self, init_func: Callable[[nn.Module], None]) -> None:
        """Apply a weight initialization function to all the model's parameters."""
        self.apply(init_func)

    def get_activation_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Get activation maps for a specific layer."""
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = self.get_layer(layer_name)
        handle = layer.register_forward_hook(get_activation(layer_name))
        self(x)
        handle.remove()

        if layer_name not in activation:
            raise ValueError(f"Layer {layer_name} did not produce any output.")
        return activation[layer_name]

class DataParams(BaseModel):
    """
    Pydantic model for all data-related parameters.
    This class defines the structure and validation rules for dataset configuration.
    """

    data_path: Union[str, List[str]] = Field(..., description="Path(s) to the dataset")
    task_type: Optional[str] = Field(..., description="Type of task (e.g., 'vision', 'nlp', 'tabular')")
    batch_size: int = Field(32, ge=1, description="Batch size for data loading")
    num_workers: int = Field(4, ge=0, description="Number of workers for data loading")
    shuffle: bool = Field(True, description="Whether to shuffle the dataset")
    validation_split: float = Field(0.2, ge=0.0, le=1.0, description="Fraction of data to use for validation")
    test_split: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of data to use for testing")
    transforms: Optional[Dict[str, Any]] = Field(None, description="Transform configurations")
    input_size: Optional[Tuple[int, ...]] = Field(None, description="Input size for the model (e.g., image dimensions)")
    num_classes: Optional[int] = Field(None, ge=1, description="Number of classes for classification tasks")
    class_names: Optional[List[str]] = Field(None, description="List of class names")
    augmentations: Optional[Dict[str, Any]] = Field(None, description="Data augmentation configurations")

    class Config:
        arbitrary_types_allowed = True

    @field_validator('task_type')
    def validate_task_type(cls, v):
        """Validate that the task type is one of the accepted values."""
        accepted_tasks = {'vision', 'nlp', 'tabular'}
        if v not in accepted_tasks:
            raise ValueError(f"task_type must be one of {accepted_tasks}")
        return v

    @field_validator('validation_split', 'test_split')
    def validate_splits(cls, v):
        """Validate that the split values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Split values must be between 0 and 1")
        return v

    @field_validator('class_names')
    def validate_class_names(cls, v, values: Dict):
        """Validate that the number of class names matches num_classes if both are provided."""
        num_classes = values.get('num_classes')
        if num_classes is not None and v is not None:
            if len(v) != num_classes:
                raise ValueError(f"Number of class names ({len(v)}) does not match num_classes ({num_classes})")
        return v

class DataManager(ABC):
    """
    Abstract base class for data management.
    This class provides a framework for loading, preprocessing, and managing datasets.
    """

    def __init__(self, params: DataParams) -> None:
        """
        Initialize the DataManager.

        Args:
            params (DataParams): Configuration parameters for data management.
        """
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.setup_transforms()

    @abstractmethod
    def load_data(self) -> Any:
        """
        Load the data from the specified path(s).

        Returns:
            Any: The loaded raw data.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the loaded data.

        Args:
            data (Any): The raw data to preprocess.

        Returns:
            Any: The preprocessed data.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def create_dataset(self, data: Any, is_train: bool = True) -> Dataset:
        """
        Create a dataset from the preprocessed data.

        Args:
            data (Any): The preprocessed data.
            is_train (bool): Whether this is a training dataset.

        Returns:
            Dataset: The created dataset.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def setup_transforms(self) -> None:
        """Set up transforms based on the task type and specified transforms."""
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def apply_transforms(self, data: Any, is_train: bool = True) -> Any:
        """
        Apply the specified transforms to the data.

        Args:
            data (Any): The data to transform.
            is_train (bool): Whether this is for training data.

        Returns:
            Any: The transformed data.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def setup(self) -> None:
        """Set up the datasets for training, validation, and testing."""
        logger.info("Setting up datasets...")
        data = self.load_data()
        preprocessed_data = self.preprocess_data(data)

        if self.params.test_split > 0:
            train_val_data, test_data = train_test_split(
                preprocessed_data,
                test_size=self.params.test_split,
                random_state=42
            )
        else:
            train_val_data, test_data = preprocessed_data, None

        if self.params.validation_split > 0:
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=self.params.validation_split / (1 - self.params.test_split),
                random_state=42
            )
            self.train_dataset = self.create_dataset(train_data, is_train=True)
            self.val_dataset = self.create_dataset(val_data, is_train=False)
        else:
            self.train_dataset = self.create_dataset(train_val_data, is_train=True)
            self.val_dataset = None

        if test_data is not None:
            self.test_dataset = self.create_dataset(test_data, is_train=False)

        logger.info("Datasets setup completed.")

    def get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create and return data loaders for train, validation, and test datasets.

        Returns:
            Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: Train, validation, and test data loaders.
        """
        if not self.train_dataset:
            raise ValueError("Datasets are not set up. Call setup() first.")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle,
            num_workers=self.params.num_workers,
            collate_fn=self.collate_fn
        )

        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                collate_fn=self.collate_fn
            )

        test_loader = None
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                collate_fn=self.collate_fn
            )

        return train_loader, val_loader, test_loader

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Calculate class weights for imbalanced datasets.
        Useful for adjusting loss functions to handle class imbalance.

        Returns:
            Optional[torch.Tensor]: Tensor of class weights, or None if not applicable.
        """
        if self.params.task_type != 'classification' or not self.train_dataset:
            return None

        labels = [sample[1] for sample in self.train_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the datasets.

        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics.
        """
        stats = {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "num_classes": self.params.num_classes,
            "class_names": self.params.class_names,
            "input_size": self.params.input_size,
        }

        if self.params.task_type == 'classification' and self.train_dataset:
            class_distribution = self.get_class_distribution(self.train_dataset)
            stats["class_distribution"] = class_distribution

        return stats

    def get_class_distribution(self, dataset: Dataset) -> Dict[int, int]:
        """
        Get the distribution of classes in a dataset.

        Args:
            dataset (Dataset): The dataset to analyze.

        Returns:
            Dict[int, int]: Dictionary mapping class labels to their counts.
        """
        class_counts = {}
        for _, label in dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def get_sample(self, index: int, dataset: str = 'train') -> Tuple[Any, Any]:
        """
        Get a specific sample from the specified dataset.

        Args:
            index (int): Index of the sample to retrieve.
            dataset (str): Which dataset to retrieve from ('train', 'val', or 'test').

        Returns:
            Tuple[Any, Any]: A tuple containing the sample data and label.

        Raises:
            ValueError: If an invalid dataset is specified or the dataset is not available.
        """
        if dataset == 'train' and self.train_dataset:
            return self.train_dataset[index]
        elif dataset == 'val' and self.val_dataset:
            return self.val_dataset[index]
        elif dataset == 'test' and self.test_dataset:
            return self.test_dataset[index]
        else:
            raise ValueError(f"Invalid dataset specified or dataset not available: {dataset}")

    @abstractmethod
    def collate_fn(self, batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        """
        Custom collate function for DataLoader.
        This method should be implemented in subclasses based on the specific data type.

        Args:
            batch (List[Tuple[Any, Any]]): A list of samples from the dataset.

        Returns:
            Tuple[Any, Any]: A tuple containing batched data and labels.
        """
        raise NotImplementedError("Subclass must implement this method")

class MetricsManager:
    """
    Manages metrics computation and tracking for model evaluation.
    """

    def __init__(self, metrics_config: List[Dict[str, Any]], device: Union[str, torch.device] = 'cpu') -> None:
        """
        Initialize the MetricsManager.

        Args:
            metrics_config (List[Dict[str, Any]]): Configuration for metrics to be tracked.
            device (Union[str, torch.device]): Device to compute metrics on.
        """
        self.metrics: Dict[str, Metric] = {}
        self.device = torch.device(device)
        self._initialize_metrics(metrics_config)

    def _initialize_metrics(self, metrics_config: List[Dict[str, Any]]) -> None:
        """
        Initialize metrics based on the provided configuration.

        Args:
            metrics_config (List[Dict[str, Any]]): Configuration for metrics to be initialized.
        """
        for metric_info in metrics_config:
            metric_name = metric_info['name']
            metric_class = getattr(torchmetrics, metric_info['class'])
            metric_params = metric_info.get('params', {})
            try:
                self.metrics[metric_name] = metric_class(**metric_params).to(self.device)
            except Exception as e:
                logger.error(f"Failed to initialize metric {metric_name}: {str(e)}")
                raise

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update all metrics with new predictions and targets.

        Args:
            outputs (torch.Tensor): Model outputs/predictions.
            targets (torch.Tensor): Ground truth targets.
        """
        for metric in self.metrics.values():
            try:
                metric.update(outputs, targets)
            except Exception as e:
                logger.error(f"Error updating metric {metric.__class__.__name__}: {str(e)}")
                raise

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute and return all metrics.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of computed metrics.
        """
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def get_metric(self, name: str) -> Metric:
        """
        Get a specific metric by name.

        Args:
            name (str): Name of the metric to retrieve.

        Returns:
            Metric: The requested metric object.

        Raises:
            KeyError: If the metric is not found.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found.")
        return self.metrics[name]

    def add_metric(self, name: str, metric: Metric) -> None:
        """
        Add a new metric to the manager.

        Args:
            name (str): Name of the metric.
            metric (Metric): The metric to add.

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already exists.")
        self.metrics[name] = metric.to(self.device)

    def remove_metric(self, name: str) -> None:
        """
        Remove a metric from the manager.

        Args:
            name (str): Name of the metric to remove.

        Raises:
            KeyError: If the metric is not found.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found.")
        del self.metrics[name]

    def to(self, device: Union[str, torch.device]) -> 'MetricsManager':
        """
        Move all metrics to the specified device.

        Args:
            device (Union[str, torch.device]): The device to move metrics to.

        Returns:
            MetricsManager: Self reference for method chaining.
        """
        self.device = torch.device(device)
        for metric in self.metrics.values():
            metric.to(self.device)
        return self

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics, including their current values and configurations.

        Returns:
            Dict[str, Any]: A dictionary containing metric summaries.
        """
        summary = {}
        for name, metric in self.metrics.items():
            summary[name] = {
                'value': metric.compute().item(),
                'type': metric.__class__.__name__,
                'config': metric._defaults  # Access to protected member, might change in future torchmetrics versions
            }
        return summary

    def log_metrics(self, step: int) -> None:
        """
        Log current metric values.

        Args:
            step (int): Current step or epoch number.
        """
        metric_values = self.compute()
        for name, value in metric_values.items():
            logger.info(f"Step {step}, {name}: {value.item():.4f}")

class ModelStorageManager:
    """Manages model storage, including saving, loading, and versioning models."""

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.version_file = os.path.join(self.base_dir, "version_info.txt")
        self._load_version_info()

    def _load_version_info(self) -> None:
        """Load version information from the version file."""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                self.current_version = int(f.read().strip())
        else:
            self.current_version = 0

    def _update_version_info(self) -> None:
        """Update the version information in the version file."""
        with open(self.version_file, 'w') as f:
            f.write(str(self.current_version))

    def save_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_params: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save the model, optimizer state, training parameters, and metrics.

        Args:
            model (nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            train_params (Dict[str, Any]): Training parameters.
            epoch (int): Current epoch number.
            metrics (Dict[str, float]): Current metric values.
            filename (Optional[str]): Custom filename for the saved model.
            tags (Optional[List[str]]): Tags to associate with the saved model.

        Returns:
            str: Path to the saved model file.
        """
        self.current_version += 1
        if filename is None:
            filename = f'model_v{self.current_version}_epoch_{epoch}.pth'

        path = os.path.join(self.base_dir, filename)

        torch.save({
            'version': self.current_version,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_params': train_params,
            'metrics': metrics,
            'tags': tags or []
        }, path)

        self._update_version_info()
        logger.info(f"Model saved to {path}")
        return path

    def load_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a saved model and return related information.

        Args:
            model (nn.Module): The model to load weights into.
            optimizer (torch.optim.Optimizer): The optimizer to load state into.
            path (str): Path to the saved model file.
            device (str): Device to load the model onto.

        Returns:
            Dict[str, Any]: Dictionary containing loaded model information.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Model loaded from {path}")

        return {
            'version': checkpoint.get('version', 0),
            'epoch': checkpoint['epoch'],
            'train_params': checkpoint['train_params'],
            'metrics': checkpoint['metrics'],
            'tags': checkpoint.get('tags', [])
        }

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved model files in the base directory with their information.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing information about each saved model.
        """
        models = []
        for f in os.listdir(self.base_dir):
            if f.endswith('.pth'):
                path = os.path.join(self.base_dir, f)
                info: Dict = torch.load(path, map_location='cpu')
                models.append({
                    'filename': f,
                    'version': info.get('version', 0),
                    'epoch': info['epoch'],
                    'metrics': info['metrics'],
                    'tags': info.get('tags', [])
                })
        return models

    def delete_model(self, filename: str) -> None:
        """
        Delete a saved model file.

        Args:
            filename (str): Name of the file to delete.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        path = os.path.join(self.base_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted model: {path}")
        else:
            raise FileNotFoundError(f"No model found at {path}")

    def get_best_model(self, metric: str = 'val_loss', mode: str = 'min') -> Optional[str]:
        """
        Get the filename of the best model based on a specific metric.

        Args:
            metric (str): The metric to use for comparison.
            mode (str): 'min' if lower is better, 'max' if higher is better.

        Returns:
            Optional[str]: Filename of the best model, or None if no models found.
        """
        models = self.list_saved_models()
        if not models:
            return None

        best_model = min(models, key=lambda x: x['metrics'].get(metric, float('inf'))) if mode == 'min' else \
                     max(models, key=lambda x: x['metrics'].get(metric, float('-inf')))
        
        return best_model['filename']

    def get_latest_model(self) -> Optional[str]:
        """
        Get the filename of the latest saved model based on version number.

        Returns:
            Optional[str]: Filename of the latest model, or None if no models found.
        """
        models = self.list_saved_models()
        if not models:
            return None
        return max(models, key=lambda x: x['version'])['filename']

    def to_torchscript(self, model: nn.Module, input_shape: Optional[Tuple[int, ...]] = None, filename: Optional[str] = None) -> str:
        """Convert the model to TorchScript and save it."""
        if input_shape is None:
            if hasattr(model, 'input_shape'):
                input_shape = model.input_shape
            else:
                raise ValueError("Please provide input_shape or ensure the model has an input_shape attribute.")

        example_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(model, example_input)

        if filename is None:
            filename = f"{model.__class__.__name__}_torchscript.pt"

        path = os.path.join(self.base_dir, filename)
        torch.jit.save(traced_model, path)

        logger.info(f"TorchScript model saved to {path}")
        return path

    def load_torchscript(self, path: str) -> torch.jit.ScriptModule:
        """Load a TorchScript model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No TorchScript model found at {path}")

        model = torch.jit.load(path)
        logger.info(f"TorchScript model loaded from {path}")
        return model
  
class TrainingParams(BaseModel):
    """
    Pydantic model for all training parameters.
    This class defines the structure and validation rules for training configuration.
    """

    # Device and hardware settings
    device: str = Field(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="Device to use for training"
    )
    use_mixed_precision: bool = Field(False, description="Whether to use mixed precision training")

    # Optimization settings
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate for optimization")
    optimizer: Literal["adam", "sgd"] = Field("adam", description="Optimizer to use")
    clip_grad_norm: Optional[float] = Field(None, gt=0, description="Clip gradient norm if specified")

    # Training loop settings
    batch_size: int = Field(32, gt=0, description="Batch size for training")
    epochs: int = Field(10, gt=0, description="Number of training epochs")
    early_stopping: bool = Field(False, description="Whether to use early stopping")
    patience: int = Field(5, ge=0, description="Patience for early stopping")

    # Learning rate scheduler settings
    use_scheduler: bool = Field(False, description="Whether to use a learning rate scheduler")
    scheduler_type: Optional[Literal["reduce_on_plateau", "step", "cosine"]] = Field(
        None, description="Type of learning rate scheduler to use"
    )

    # Logging and checkpoint settings
    use_tensorboard: bool = Field(False, description="Whether to use TensorBoard for logging")
    checkpoint_dir: str = Field("checkpoints", description="Directory to save model checkpoints")
    log_interval: int = Field(100, gt=0, description="Interval for logging training progress")
    val_interval: int = Field(1, gt=0, description="Interval for validation (in epochs)")

    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for specific tasks")

    class Config:
        arbitrary_types_allowed = True

    @field_validator('scheduler_type')
    def validate_scheduler_type(cls, v, values: Dict):
        """Validate that scheduler_type is set only when use_scheduler is True."""
        if values.get('use_scheduler') and v is None:
            raise ValueError("scheduler_type must be set when use_scheduler is True")
        if not values.get('use_scheduler') and v is not None:
            raise ValueError("scheduler_type should be None when use_scheduler is False")
        return v

    def get_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        """
        Get the optimizer based on the specified parameters.

        Args:
            model_parameters: The parameters of the model to optimize.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        if self.optimizer == "adam":
            return torch.optim.Adam(model_parameters, lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(model_parameters, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Get the learning rate scheduler based on the specified parameters.

        Args:
            optimizer: The optimizer to schedule.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: The initialized scheduler, if use_scheduler is True.
        """
        if not self.use_scheduler:
            return None
        
        if self.scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

class TrainingManager:
    """Manages the training process for deep learning models."""

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
        self.model_storage = ModelStorageManager(self.train_params.checkpoint_dir)
        
        self.optimizer = self.train_params.get_optimizer(self.model.parameters())
        self.scheduler = self.train_params.get_scheduler(self.optimizer)
        self.scaler = GradScaler() if self.train_params.use_mixed_precision else None
        
        self.writer = SummaryWriter() if self.train_params.use_tensorboard else None
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.metrics_manager = MetricsManager(metrics_config, device=self.train_params.device)

    def to_device(self) -> None:
        """Move the model and loss function to the specified device."""
        self.model.to(self.train_params.device)
        self.loss_fn.to(self.train_params.device)
    
    def load_model(self, path: str) -> None:
        """Load a saved model and update training parameters."""
        try:
            loaded_info = self.model_storage.load_model(self.model, self.optimizer, path, self.train_params.device)
            self.train_params = TrainingParams(**loaded_info['train_params'])
            logger.info(f"Model loaded from {path}")
            logger.info(f"Loaded model info: Epoch {loaded_info['epoch']}, Metrics: {loaded_info['metrics']}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""
        if self.train_params.use_mixed_precision:
            with autocast():
                return self.model(inputs)
        else:
            return self.model(inputs)

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        if self.train_params.use_mixed_precision:
            with autocast():
                return self.loss_fn(outputs, targets)
        else:
            return self.loss_fn(outputs, targets)

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Perform a backward pass."""
        if self.train_params.use_mixed_precision:
            self.scaler.scale(loss).backward()
            if self.train_params.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.train_params.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
            self.optimizer.step()

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Perform a single training step."""
        inputs, targets = self._prepare_batch(batch)
        self.optimizer.zero_grad()
        
        outputs = self._forward_pass(inputs)
        loss = self._compute_loss(outputs, targets)
        self._backward_pass(loss)

        self.metrics_manager.update(outputs, targets)
        return {'loss': loss.item()}

    def val_step(self, batch: Any) -> Dict[str, float]:
        """Perform a single validation step."""
        inputs, targets = self._prepare_batch(batch)

        with torch.no_grad():
            outputs = self._forward_pass(inputs)
            loss = self._compute_loss(outputs, targets)

        self.metrics_manager.update(outputs, targets)
        return {'loss': loss.item()}

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of data for training or validation."""
        try:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, dict):
                inputs = batch['input']
                targets = batch['target']
            else:
                raise ValueError("Unsupported batch format")
            
            return inputs.to(self.train_params.device), targets.to(self.train_params.device)
        except Exception as e:
            logger.error(f"Error preparing batch: {str(e)}")
            raise

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(self.train_data_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{self.train_params.epochs}", dynamic_ncols=True) as pbar:
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
        """Validate the model on the validation set."""
        self.model.eval()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(self.val_data_loader)
        
        with torch.no_grad(), tqdm(total=num_batches, desc=f"Validation Epoch {epoch+1}", dynamic_ncols=True) as pbar:
            for batch in self.val_data_loader:
                step_results = self.val_step(batch)
                total_loss += step_results['loss']
                pbar.update(1)
                pbar.set_postfix(loss=f"{step_results['loss']:.4f}")
        
        metrics = self.metrics_manager.compute()
        metrics['loss'] = total_loss / num_batches
        self._log_progress('val', epoch, 0, metrics)
        return metrics

    def train_loop(self) -> None:
        """Main training loop."""
        for epoch in range(self.train_params.epochs):
            train_results = self.train_epoch(epoch)
            
            if self.val_data_loader and epoch % self.train_params.val_interval == 0:
                val_results = self.validate(epoch)
                self._update_scheduler(val_results['loss'])
                if self._check_early_stopping(val_results['loss'], epoch):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            self._save_checkpoint(epoch, train_results)

    def _update_scheduler(self, val_loss: float) -> None:
        """Update the learning rate scheduler."""
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Check if early stopping criteria are met."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_best_model(epoch, val_loss)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.train_params.early_stopping and self.patience_counter >= self.train_params.patience

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a checkpoint of the model."""
        self.model_storage.save_model(
            self.model, 
            self.optimizer, 
            self.train_params.model_dump(), 
            epoch, 
            metrics
        )

    def _save_best_model(self, epoch: int, val_loss: float) -> None:
        """Save the best model based on validation loss."""
        self.model_storage.save_model(
            self.model, 
            self.optimizer, 
            self.train_params.model_dump(), 
            epoch, 
            {'val_loss': val_loss}, 
            'best_model.pth'
        )

    def test_loop(self) -> Dict[str, float]:
        """Evaluate the model on the test set."""
        self.model.eval()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(self.test_data_loader)
        
        with torch.no_grad():
            for batch in self.test_data_loader:
                step_results = self.val_step(batch)
                total_loss += step_results['loss']
        
        metrics = self.metrics_manager.compute()
        metrics['loss'] = total_loss / num_batches
        logger.info(f"Test Results: {metrics}")
        return metrics

    def _log_progress(self, phase: str, epoch: int, step: int, metrics: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Log training progress to console and TensorBoard."""
        if self.writer:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"{phase}/{metric_name}", metric_value, epoch * len(self.train_data_loader) + step)
        
        log_str = f"{phase.capitalize()} Epoch {epoch+1}, Step {step}: "
        log_str += ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        logger.info(log_str)

...