from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Any, Dict, Literal, Optional, Union, List, Tuple, Type
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.hooks import RemovableHandle
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from loguru import logger
import json

class BaseModelError(Exception):
    """Base exception class for BaseModel errors."""
    pass

class LayerNotFoundError(BaseModelError):
    """Exception raised when a specified layer is not found in the model."""
    pass

class ShapeComputationError(BaseModelError):
    """Exception raised when there's an error computing layer shapes."""
    pass

class BaseModel(nn.Module):
    """
    Enhanced abstract base class for all models in the framework.
    Provides a common interface and utility methods for various model types.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BaseModel.
        """
        super().__init__()
        self.model_type: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.task_type: Optional[str] = None
        self._layer_shapes: Dict[str, Tuple[int, ...]] = {}
        self._hooks: List[RemovableHandle] = []
        self._device: Optional[torch.device] = None
        self.config = config or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def inference(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """
        Perform inference on the input.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Compute the loss for the model.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def compute_prediction(self, outputs: torch.Tensor, **kwargs: Any) -> Any:
        """
        Compute predictions from the model outputs.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        """
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
            "config": self.config
        }

    def set_model_attributes(
        self,
        model_type: str,
        task_type: str,
        input_shape: Optional[Tuple[int, ...]],
        output_shape: Optional[Tuple[int, ...]]
    ) -> None:
        """
        Set multiple model attributes at once.\
        """
        self.model_type = model_type
        self.task_type = task_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        if input_shape:
            self._layer_shapes["input"] = input_shape
        if output_shape:
            self._layer_shapes["output"] = output_shape
        logger.info(f"Model attributes set: type={model_type}, task={task_type}")

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model is currently loaded.
        """
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def _set_layer_trainable(self, layer_names: List[str], trainable: bool) -> None:
        """
        Set the trainable status of specified layers.
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = trainable
        logger.info(f"{'Unfrozen' if trainable else 'Frozen'} layers: {layer_names}")

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze specified layers of the model.
        """
        self._set_layer_trainable(layer_names, False)

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """
        Unfreeze specified layers of the model.

        """
        self._set_layer_trainable(layer_names, True)

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        """
        Get all trainable parameters of the model.
        """
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def load_pretrained_weights(self, weights_path: str, strict: bool = True) -> None:
        """
        Load pretrained weights into the model.
        """
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
        """
        Get the output of a specific layer given an input tensor.
        """
        output = {}

        def hook(module: nn.Module, input: Any, out: torch.Tensor) -> None:
            output['value'] = out

        try:
            layer = self.get_layer(layer_name)
            handle = layer.register_forward_hook(hook)
            self.forward(x)
            handle.remove()
        except LayerNotFoundError:
            logger.error(f"Layer {layer_name} not found in the model")
            raise

        if 'value' not in output:
            raise ValueError(f"Layer {layer_name} did not produce any output.")
        return output['value']

    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Get a specific layer of the model by name.
        """
        for name, module in self.named_modules():
            if name == layer_name:
                return module
        raise LayerNotFoundError(f"Layer {layer_name} not found in the model")

    def get_shape(self, layer: Union[str, int], dummy_input: Optional[torch.Tensor] = None) -> Tuple[int, ...]:
        """
        Get the shape of a specific layer.
        """
        try:
            if dummy_input is not None:
                self.compute_shapes(dummy_input.shape)

            if isinstance(layer, str):
                return self._get_shape_by_name(layer)
            elif isinstance(layer, int):
                return self._get_shape_by_index(layer)
        except Exception as e:
            raise ShapeComputationError(f"Error computing shape for layer {layer}: {str(e)}")

    def _get_shape_by_name(self, layer_name: str) -> Tuple[int, ...]:
        """Helper method to get shape by layer name."""
        if layer_name in self._layer_shapes:
            return self._layer_shapes[layer_name]
        for name, module in self.named_modules():
            if name == layer_name and hasattr(module, 'weight'):
                return tuple(module.weight.shape)
        raise ShapeComputationError(f"Shape for layer {layer_name} not found or not computed.")

    def _get_shape_by_index(self, layer_index: int) -> Tuple[int, ...]:
        """Helper method to get shape by layer index."""
        if layer_index == 0:
            return self.input_shape or tuple()
        elif layer_index == -1:
            return self.output_shape or tuple()
        elif str(layer_index) in self._layer_shapes:
            return self._layer_shapes[str(layer_index)]
        raise ShapeComputationError(f"Shape for layer index {layer_index} not found or not computed.")

    def compute_shapes(self, input_shape: Tuple[int, ...]) -> None:
        """
        Compute and store the shapes of all layers in the model.
        """
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._layer_shapes[str(len(self._layer_shapes))] = tuple(output.shape[1:])

        self._layer_shapes.clear()
        self._layer_shapes["input"] = input_shape

        try:
            for module in self.modules():
                if not isinstance(module, nn.Sequential):
                    self._hooks.append(module.register_forward_hook(hook))

            dummy_input = torch.randn(input_shape).to(self.device)
            self(dummy_input)

            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

            self.input_shape = input_shape
            self.output_shape = self._layer_shapes[str(max(int(k) for k in self._layer_shapes if k.isdigit()))]
        except Exception as e:
            raise ShapeComputationError(f"Error computing shapes: {str(e)}")

    def summary(self, input_size: Optional[Tuple[int, ...]] = None, **kwargs: Any) -> None:
        """
        Print a summary of the model architecture with additional options.
        """
        from torchinfo import summary as torch_summary

        if input_size is None and self.input_shape is None:
            raise ValueError("Please provide input_size or set input_shape for the model.")

        input_size = input_size or self.input_shape
        torch_summary(self, input_size=input_size, **kwargs)

    def apply_weight_initialization(self, init_func: callable) -> None:
        """
        Apply a weight initialization function to all the model's parameters.
        """
        self.apply(init_func)
        logger.info(f"Applied weight initialization: {init_func.__name__}")

    def get_activation_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Get activation maps for a specific layer.
        """
        activation = {}

        def get_activation(name: str) -> callable:
            def hook(model: nn.Module, input:Any, output: torch.Tensor) -> None:
                activation[name] = output.detach()
            return hook

        try:
            layer = self.get_layer(layer_name)
            handle = layer.register_forward_hook(get_activation(layer_name))
            self(x)
            handle.remove()
        except LayerNotFoundError:
            logger.error(f"Layer {layer_name} not found in the model")
            raise

        if layer_name not in activation:
            raise ValueError(f"Layer {layer_name} did not produce any output.")
        return activation[layer_name]

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        """
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_config': self.config,
                'model_info': self.get_info()
            }, path)
            logger.info(f"Model saved to {path}")
        except IOError as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> 'BaseModel':
        """
        Load a model from a file.
        """
        try:
            checkpoint = torch.load(path, map_location=device)
            model = cls(config=checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.set_model_attributes(**checkpoint['model_info'])
            logger.info(f"Model loaded from {path}")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {path}")
            raise
        except RuntimeError as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise

    def to_torchscript(self, example_inputs: torch.Tensor, path: Optional[str] = None) -> torch.jit.ScriptModule:
        """
        Convert the model to TorchScript format.
        """
        try:
            scripted_model = torch.jit.trace(self, example_inputs)
            if path:
                torch.jit.save(scripted_model, path)
                logger.info(f"TorchScript model saved to {path}")
            return scripted_model
        except Exception as e:
            logger.error(f"Error converting model to TorchScript: {str(e)}")
            raise RuntimeError(f"TorchScript conversion failed: {str(e)}")

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only (bool): If True, count only trainable parameters.

        Returns:
            int: Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimization.
        """
        return [{'params': self.parameters()}]

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set or update the model configuration.
        """
        self.config.update(config)
        logger.info("Model configuration updated")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current model configuration.
        """
        return self.config

    def export_onnx(self, path: str, example_inputs: torch.Tensor, input_names: List[str], output_names: List[str]) -> None:
        """
        Export the model to ONNX format.
        """
        try:
            torch.onnx.export(self, example_inputs, path, input_names=input_names, output_names=output_names)
            logger.info(f"Model exported to ONNX format at {path}")
        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {str(e)}")
            raise RuntimeError(f"ONNX export failed: {str(e)}")

    def get_memory_usage(self) -> float:
        """
        Get the estimated memory usage of the model.
        """
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        mem_total = (mem_params + mem_bufs) / 1024 / 1024  # Convert to MB
        return mem_total

    def log_gradients(self) -> None:
        """
        Log the gradients of all parameters.
        """
        for name, param in self.named_parameters():
            if param.grad is not None:
                logger.debug(f"Gradient for {name}: mean={param.grad.mean():.4f}, std={param.grad.std():.4f}")

    def get_layer_weights(self, layer_name: str) -> torch.Tensor:
        """
        Get the weights of a specific layer.
        """
        try:
            layer = self.get_layer(layer_name)
            if hasattr(layer, 'weight'):
                return layer.weight.data
            else:
                raise AttributeError(f"Layer {layer_name} does not have weights.")
        except LayerNotFoundError:
            logger.error(f"Layer {layer_name} not found in the model")
            raise

    def visualize_model(self, input_size: Tuple[int, ...], output_file: str) -> None:
        """
        Visualize the model architecture and save it to a file.
        """
        try:
            from torchviz import make_dot
            x = torch.randn(input_size).to(self.device)
            y = self(x)
            dot = make_dot(y, params=dict(self.named_parameters()))
            dot.render(output_file, format='png')
            logger.info(f"Model visualization saved to {output_file}.png")
        except ImportError:
            logger.error("graphviz is not installed. Please install it to use this feature.")
            raise
        except Exception as e:
            logger.error(f"Error visualizing model: {str(e)}")
            raise RuntimeError(f"Model visualization failed: {str(e)}")

    def freeze(self) -> None:
        """
        Freeze all parameters of the model.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("All model parameters frozen")

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters of the model.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen")

    def get_intermediate_activations(self, x: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get intermediate activations for specified layers.
        """
        activations = {}
        handles = []

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        try:
            for name in layer_names:
                layer = self.get_layer(name)
                handles.append(layer.register_forward_hook(get_activation(name)))

            self(x)

            for handle in handles:
                handle.remove()

            return activations
        except LayerNotFoundError as e:
            logger.error(f"Error getting intermediate activations: {str(e)}")
            raise

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: String representation of the model.
        """
        model_info = self.get_info()
        return (f"{self.__class__.__name__}(\n"
                f"  Model Type: {model_info['model_type']}\n"
                f"  Task Type: {model_info['task_type']}\n"
                f"  Input Shape: {model_info['input_shape']}\n"
                f"  Output Shape: {model_info['output_shape']}\n"
                f"  Total Parameters: {model_info['num_parameters']}\n"
                f"  Trainable Parameters: {model_info['trainable_parameters']}\n"
                f")")

    def to_json(self) -> str:
        """
        Convert model information to JSON format.
        """
        return json.dumps(self.get_info(), indent=2)

class DataParams(BaseModel):
    """
    Configuration parameters for dataset management.

    This class encapsulates all necessary parameters for managing datasets
    in machine learning tasks, including data paths, task types, loading
    configurations, and data processing options.
    """

    class TaskType(str, Enum):
        VISION = "vision"
        NLP = "nlp"
        TABULAR = "tabular"
        AUDIO = "audio"
        TIME_SERIES = "time_series"

    class SampleStrategy(str, Enum):
        RANDOM = "random"
        STRATIFIED = "stratified"

    class DataFormat(str, Enum):
        CSV = "csv"
        JSON = "json"
        PARQUET = "parquet"
        IMAGE = "image"
        AUDIO = "audio"
        TEXT = "text"

    # Data source and type
    data_path: Union[str, List[str]] = Field(..., description="Path(s) to the dataset(s)")
    task_type: TaskType = Field(..., description="Type of machine learning task")
    data_format: DataFormat = Field(..., description="Format of the dataset")

    # Data loading parameters
    batch_size: int = Field(32, ge=1, description="Batch size for data loading")
    num_workers: int = Field(4, ge=0, description="Number of workers for data loading")
    shuffle: bool = Field(True, description="Whether to shuffle the dataset")

    # Data splitting parameters
    validation_split: float = Field(0.2, ge=0.0, le=1.0, description="Fraction of data to use for validation")
    test_split: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of data to use for testing")

    # Data processing options
    transforms: Optional[Dict[str, Any]] = Field(None, description="Transform configurations")
    augmentations: Optional[Dict[str, Any]] = Field(None, description="Data augmentation configurations")
    preprocessing_steps: Optional[List[str]] = Field(None, description="List of preprocessing steps to apply")
    preprocessing_params: Optional[Dict[str, Any]] = Field(None, description="Parameters for preprocessing steps")

    # Model input parameters
    input_size: Optional[Tuple[int, ...]] = Field(None, description="Input size for the model")
    num_classes: Optional[int] = Field(None, ge=1, description="Number of classes for classification tasks")
    class_names: Optional[List[str]] = Field(None, description="List of class names")

    # Advanced options
    sample_strategy: SampleStrategy = Field(SampleStrategy.RANDOM, description="Strategy for sampling data")
    sample_size: Optional[int] = Field(None, ge=1, description="Number of samples to use (for subsampling)")
    balance_method: Optional[str] = Field(None, description="Method for balancing classes (e.g., 'oversample', 'undersample')")
    cache_data: bool = Field(False, description="Whether to cache data in memory")
    distributed: bool = Field(False, description="Whether to use distributed data loading")

    # Custom parameters for flexibility
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for specific tasks")

    class Config:
        use_enum_values = True

    @field_validator('validation_split', 'test_split')
    def validate_splits(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Split values must be between 0 and 1")
        return v

    @field_validator('class_names')
    def validate_class_names(cls, v, values):
        num_classes = values.get('num_classes')
        if num_classes is not None and v is not None:
            if len(v) != num_classes:
                raise ValueError(f"Number of class names ({len(v)}) does not match num_classes ({num_classes})")
        return v

    @field_validator('data_path')
    def validate_data_path(cls, v):
        if isinstance(v, str):
            if not os.path.exists(v):
                raise ValueError(f"Data path does not exist: {v}")
        elif isinstance(v, list):
            for path in v:
                if not os.path.exists(path):
                    raise ValueError(f"Data path does not exist: {path}")
        else:
            raise ValueError("data_path must be a string or a list of strings")
        return v

    def get_split_sizes(self) -> Tuple[float, float, float]:
        """
        Calculate the split sizes for train, validation, and test sets.

        Returns:
            Tuple[float, float, float]: Proportions for train, validation, and test sets.
        """
        test_size = self.test_split
        val_size = self.validation_split * (1 - test_size)
        train_size = 1 - test_size - val_size
        return train_size, val_size, test_size

    def get_transform_config(self, phase: str) -> Dict[str, Any]:
        """
        Get the transform configuration for a specific phase.

        Args:
            phase (str): The dataset phase ('train', 'val', or 'test').

        Returns:
            Dict[str, Any]: Transform configuration for the specified phase.
        """
        if self.transforms is None:
            return {}
        return self.transforms.get(phase, {})

    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get the data augmentation configuration.

        Returns:
            Dict[str, Any]: Data augmentation configuration.
        """
        return self.augmentations or {}

class DataManager(ABC):
    """Abstract base class for data management."""

    def __init__(self, params: DataParams):
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.transforms = self.setup_transforms()

    def load_data(self) -> Any:
        """Load the data from the specified path(s)."""
        if isinstance(self.params.data_path, list):
            return [self._load_single_dataset(path) for path in self.params.data_path]
        return self._load_single_dataset(self.params.data_path)

    def _load_single_dataset(self, path: str) -> Any:
        if self.params.data_format == DataParams.DataFormat.CSV:
            return pd.read_csv(path)
        elif self.params.data_format == DataParams.DataFormat.JSON:
            return pd.read_json(path)
        elif self.params.data_format == DataParams.DataFormat.PARQUET:
            return pd.read_parquet(path)
        elif self.params.data_format in [DataParams.DataFormat.IMAGE, DataParams.DataFormat.AUDIO, DataParams.DataFormat.TEXT]:
            return self._load_file_dataset(path)
        else:
            raise ValueError(f"Unsupported data format: {self.params.data_format}")

    @abstractmethod
    def _load_file_dataset(self, path: str) -> Any:
        pass

    def preprocess_data(self, data: Any) -> Any:
        """Preprocess the loaded data."""
        if self.params.preprocessing_steps:
            for step in self.params.preprocessing_steps:
                preprocess_method = getattr(self, f"_preprocess_{step}", None)
                if preprocess_method:
                    data = preprocess_method(data)
                else:
                    logger.warning(f"Preprocessing step '{step}' not implemented. Skipping.")
        return data

    @abstractmethod
    def _preprocess_normalize(self, data: Any) -> Any:
        pass

    @abstractmethod
    def _preprocess_augment(self, data: Any) -> Any:
        pass

    @abstractmethod
    def create_dataset(self, data: Any, is_train: bool = True) -> Dataset:
        """Create a dataset from the preprocessed data."""
        pass

    def setup_transforms(self) -> Dict[str, Any]:
        """Set up transforms based on the task type and specified transforms."""
        transforms = {}
        if self.params.transforms:
            for phase, config in self.params.transforms.items():
                transforms[phase] = self._create_transform_pipeline(config)
        return transforms

    @abstractmethod
    def _create_transform_pipeline(self, config: Dict[str, Any]) -> Any:
        """Create a transform pipeline based on the configuration."""
        pass

    def setup(self) -> None:
        """Set up the datasets for training, validation, and testing."""
        logger.info("Setting up datasets...")
        try:
            data = self.load_data()
            data = self._sample_data(data)
            data = self._balance_data(data)
            preprocessed_data = self.preprocess_data(data)
            self._split_data(preprocessed_data)
        except Exception as e:
            logger.error(f"Error setting up datasets: {str(e)}")
            raise

    def _split_data(self, data: Any) -> None:
        """Split the data into train, validation, and test sets."""
        split_func = self._get_split_function()
        train_data, val_data, test_data = self._perform_splits(data, split_func)

        self.train_dataset = self.create_dataset(train_data, is_train=True)
        self.val_dataset = self.create_dataset(val_data, is_train=False) if val_data is not None else None
        self.test_dataset = self.create_dataset(test_data, is_train=False) if test_data is not None else None

    def _get_split_function(self):
        if self.params.sample_strategy == DataParams.SampleStrategy.STRATIFIED and self.params.task_type == DataParams.TaskType.VISION:
            return self._stratified_split
        return train_test_split

    def _perform_splits(self, data: Any, split_func) -> Tuple[Any, Any, Any]:
        if self.params.test_split > 0:
            train_val_data, test_data = split_func(data, test_size=self.params.test_split, random_state=42)
            if self.params.validation_split > 0:
                train_data, val_data = split_func(
                    train_val_data,
                    test_size=self.params.validation_split / (1 - self.params.test_split),
                    random_state=42
                )
            else:
                train_data, val_data = train_val_data, None
        else:
            train_data, val_data = split_func(data, test_size=self.params.validation_split, random_state=42) if self.params.validation_split > 0 else (data, None)
            test_data = None

        return train_data, val_data, test_data

    @abstractmethod
    def _stratified_split(self, data: Any, test_size: float, random_state: int) -> Tuple[Any, Any]:
        """Perform a stratified split of the data."""
        pass

    def get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Create and return data loaders for train, validation, and test datasets."""
        if not self.train_dataset:
            raise ValueError("Datasets are not set up. Call setup() first.")

        train_loader = self._create_data_loader(self.train_dataset, shuffle=self.params.shuffle)
        val_loader = self._create_data_loader(self.val_dataset) if self.val_dataset else None
        test_loader = self._create_data_loader(self.test_dataset) if self.test_dataset else None

        return train_loader, val_loader, test_loader

    def _create_data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            num_workers=self.params.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.params.num_workers > 0
        )

    @abstractmethod
    def collate_fn(self, batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        """Custom collate function for DataLoader."""
        pass

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Calculate class weights for imbalanced datasets."""
        if self.params.task_type != DataParams.TaskType.VISION or not self.train_dataset:
            return None

        labels = [sample[1] for sample in self.train_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the datasets."""
        stats = {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "num_classes": self.params.num_classes,
            "class_names": self.params.class_names,
            "input_size": self.params.input_size,
        }

        if self.params.task_type == DataParams.TaskType.VISION and self.train_dataset:
            stats["class_distribution"] = self.get_class_distribution(self.train_dataset)

        return stats

    def get_class_distribution(self, dataset: Dataset) -> Dict[int, int]:
        """Get the distribution of classes in a dataset."""
        class_counts = {}
        for _, label in dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def get_sample(self, index: int, dataset: str = 'train') -> Tuple[Any, Any]:
        """Get a specific sample from the specified dataset."""
        dataset_map = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        selected_dataset = dataset_map.get(dataset)
        if not selected_dataset:
            raise ValueError(f"Invalid dataset specified or dataset not available: {dataset}")
        return selected_dataset[index]

    @abstractmethod
    def apply_augmentations(self, data: Any) -> Any:
        """Apply data augmentations to the input data."""
        pass

    @abstractmethod
    def cache_dataset(self, dataset: Dataset) -> Dataset:
        """Cache the entire dataset in memory for faster access."""
        pass

    @abstractmethod
    def setup_distributed(self) -> None:
        """Set up distributed data loading if enabled."""
        pass

    def _sample_data(self, data: Any) -> Any:
        if self.params.sample_size:
            if isinstance(data, pd.DataFrame):
                return data.sample(n=min(self.params.sample_size, len(data)))
            else:
                # Implement sampling for other data types
                logger.warning("Sampling for non-DataFrame data types is not implemented. Returning original data.")
                return data
        return data

    def _balance_data(self, data: Any) -> Any:
        if self.params.balance_method:
            if self.params.balance_method == 'oversample':
                return self._oversample(data)
            elif self.params.balance_method == 'undersample':
                return self._undersample(data)
            else:
                logger.warning(f"Unsupported balance method: {self.params.balance_method}. Skipping balancing.")
        return data

    @abstractmethod
    def _oversample(self, data: Any) -> Any:
        pass

    @abstractmethod
    def _undersample(self, data: Any) -> Any:
        pass

    def get_data_statistics(self) -> Dict[str, Any]:
        stats = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
        }
        
        if isinstance(self.train_dataset, Dataset):
            stats.update(self._get_dataset_statistics(self.train_dataset))
        
        return stats

    @abstractmethod
    def _get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        pass

    @abstractmethod
    def visualize_data_distribution(self) -> None:
        pass

    def get_num_features(self) -> int:
        """Get the number of features in the dataset."""
        if self.train_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        
        sample = self.train_dataset[0][0]  # Get the first sample
        if isinstance(sample, torch.Tensor):
            return sample.shape[-1]  # Assumes the last dimension is the feature dimension
        elif isinstance(sample, np.ndarray):
            return sample.shape[-1]
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")

    def get_feature_names(self) -> List[str]:
        """Get the names of features in the dataset."""
        if self.params.task_type != DataParams.TaskType.TABULAR:
            raise ValueError("Feature names are only available for tabular data.")
        
        if isinstance(self.train_dataset, pd.DataFrame):
            return list(self.train_dataset.columns)
        else:
            raise ValueError("Feature names are not available for the current dataset type.")

    def get_data_shape(self) -> Tuple[int, ...]:
        """Get the shape of the data samples."""
        if self.train_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        
        sample = self.train_dataset[0][0]  # Get the first sample
        if isinstance(sample, (torch.Tensor, np.ndarray)):
            return tuple(sample.shape)
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")

    def get_label_distribution(self) -> Dict[str, float]:
        """Get the distribution of labels in the dataset."""
        if self.train_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        
        if not hasattr(self.train_dataset, 'targets'):
            raise ValueError("Dataset does not have a 'targets' attribute.")
        
        labels = self.train_dataset.targets
        label_counts = pd.Series(labels).value_counts(normalize=True)
        return label_counts.to_dict()

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset properties."""
        return {
            "task_type": self.params.task_type,
            "data_format": self.params.data_format,
            "num_samples": len(self.train_dataset),
            "num_features": self.get_num_features(),
            "data_shape": self.get_data_shape(),
            "num_classes": self.params.num_classes,
            "class_names": self.params.class_names,
            "label_distribution": self.get_label_distribution(),
            "data_statistics": self.get_data_statistics(),
        }

    def validate_data(self) -> List[str]:
        """Perform basic validations on the dataset."""
        warnings = []
        
        if self.train_dataset is None:
            warnings.append("Train dataset is not initialized.")
        
        if len(self.train_dataset) == 0:
            warnings.append("Train dataset is empty.")
        
        if self.val_dataset is not None and len(self.val_dataset) == 0:
            warnings.append("Validation dataset is empty.")
        
        if self.test_dataset is not None and len(self.test_dataset) == 0:
            warnings.append("Test dataset is empty.")
        
        if self.params.num_classes is not None:
            unique_labels = len(set(self.train_dataset.targets))
            if unique_labels != self.params.num_classes:
                warnings.append(f"Number of unique labels ({unique_labels}) does not match num_classes ({self.params.num_classes}).")
        
        return warnings

    def apply_transforms(self, sample: Any, is_train: bool = True) -> Any:
        """Apply the appropriate transforms to a single sample."""
        phase = 'train' if is_train else 'val'
        transform = self.transforms.get(phase)
        if transform:
            return transform(sample)
        return sample

    def get_batch(self, batch_size: int, dataset: str = 'train') -> Tuple[Any, Any]:
        """Get a random batch of samples from the specified dataset."""
        dataset_map = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        selected_dataset = dataset_map.get(dataset)
        if not selected_dataset:
            raise ValueError(f"Invalid dataset specified or dataset not available: {dataset}")
        
        indices = torch.randperm(len(selected_dataset))[:batch_size]
        batch = [selected_dataset[i] for i in indices]
        return self.collate_fn(batch)

    def save_dataset(self, path: str) -> None:
        """Save the processed dataset to a file."""
        data = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset,
            'params': self.params
        }
        torch.save(data, path)
        logger.info(f"Dataset saved to {path}")

    @classmethod
    def load_dataset(cls, path: str) -> 'DataManager':
        """Load a processed dataset from a file."""
        data = torch.load(path)
        manager = cls(data['params'])
        manager.train_dataset = data['train']
        manager.val_dataset = data['val']
        manager.test_dataset = data['test']
        logger.info(f"Dataset loaded from {path}")
        return manager

    def get_data_loader_infinite(self, dataset: str = 'train') -> DataLoader:
        """Get an infinite data loader for the specified dataset."""
        dataset_map = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        selected_dataset = dataset_map.get(dataset)
        if not selected_dataset:
            raise ValueError(f"Invalid dataset specified or dataset not available: {dataset}")

        return DataLoader(
            selected_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.params.num_workers > 0,
            sampler=torch.utils.data.RandomSampler(selected_dataset, replacement=True, num_samples=int(1e10))
        )

class MetricsManager:
    """
    Manages metrics computation and tracking for model evaluation.
    Supports both torchmetrics and custom metric functions.
    """

    def __init__(self, metrics_config: List[Dict[str, Any]], device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the MetricsManager.
        
        Args:
            metrics_config (List[Dict[str, Any]]): Configuration for metrics to be initialized.
            device (Union[str, torch.device]): Device to compute metrics on.
        """
        self.metrics: Dict[str, Union[Metric, Callable]] = {}
        self.device = torch.device(device)
        self.metric_values: Dict[str, List[float]] = {}
        self._initialize_metrics(metrics_config)

    class MetricInitializationError(Exception):
        """Custom exception for metric initialization errors."""
        pass

    def _initialize_metrics(self, metrics_config: List[Dict[str, Any]]) -> None:
        """
        Initialize metrics based on the provided configuration.

        Args:
            metrics_config (List[Dict[str, Any]]): Configuration for metrics to be initialized.

        Raises:
            MetricInitializationError: If there's an error initializing a metric.
        """
        for metric_info in metrics_config:
            self._initialize_single_metric(metric_info)

    def _initialize_single_metric(self, metric_info: Dict[str, Any]) -> None:
        """
        Initialize a single metric based on its configuration.
        """
        metric_name = metric_info['name']
        metric_type = metric_info.get('type', 'torchmetrics')

        try:
            if metric_type == 'torchmetrics':
                self._initialize_torchmetric(metric_name, metric_info)
            elif metric_type == 'custom':
                self._initialize_custom_metric(metric_name, metric_info)
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")
            self.metric_values[metric_name] = []
        except Exception as e:
            raise self.MetricInitializationError(f"Failed to initialize metric {metric_name}: {str(e)}")

    def _initialize_torchmetric(self, metric_name: str, metric_info: Dict[str, Any]) -> None:
        """Initialize a torchmetrics metric."""
        metric_class = getattr(torchmetrics, metric_info['class'])
        metric_params = metric_info.get('params', {})
        self.metrics[metric_name] = metric_class(**metric_params).to(self.device)

    def _initialize_custom_metric(self, metric_name: str, metric_info: Dict[str, Any]) -> None:
        """Initialize a custom metric function."""
        if 'function' not in metric_info:
            raise ValueError(f"Custom metric {metric_name} must provide a 'function' key")
        self.metrics[metric_name] = metric_info['function']

    def update(self, outputs: Tensor, targets: Tensor) -> None:
        """
        Update all metrics with new predictions and targets.

        Args:
            outputs (Tensor): Model outputs/predictions.
            targets (Tensor): Ground truth targets.

        Raises:
            RuntimeError: If there's an error updating a metric.
        """
        for name, metric in self.metrics.items():
            try:
                if isinstance(metric, Metric):
                    metric.update(outputs, targets)
                elif callable(metric):
                    # For custom metrics, we compute them on-the-fly and store the result
                    result = metric(outputs, targets)
                    self.metric_values[name].append(result.item() if isinstance(result, Tensor) else result)
            except Exception as e:
                raise RuntimeError(f"Error updating metric {name}: {str(e)}")

    def compute(self) -> Dict[str, Tensor]:
        """
        Compute and return all metrics.
        
        Returns:
            Dict[str, Tensor]: A dictionary of computed metric values.
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                if isinstance(metric, Metric):
                    results[name] = metric.compute()
                elif callable(metric):
                    # For custom metrics, we compute the mean of stored values
                    results[name] = torch.tensor(np.mean(self.metric_values[name]))
            except Exception as e:
                raise RuntimeError(f"Error computing metric {name}: {str(e)}")
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for name, metric in self.metrics.items():
            if isinstance(metric, Metric):
                metric.reset()
            else:
                self.metric_values[name] = []

    def get_metric(self, name: str) -> Union[Metric, Callable]:
        """
        Get a specific metric by name.
        
        Args:
            name (str): Name of the metric to retrieve.
        
        Returns:
            Union[Metric, Callable]: The requested metric.
        
        Raises:
            KeyError: If the metric is not found.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found.")
        return self.metrics[name]

    def add_metric(self, name: str, metric: Union[Metric, Callable]) -> None:
        """
        Add a new metric to the manager.

        Args:
            name (str): Name of the metric.
            metric (Union[Metric, Callable]): The metric to add.

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already exists.")
        if isinstance(metric, Metric):
            self.metrics[name] = metric.to(self.device)
        else:
            self.metrics[name] = metric
        self.metric_values[name] = []

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
        del self.metric_values[name]

    def to(self, device: Union[str, torch.device]) -> 'MetricsManager':
        """
        Move all metrics to the specified device.
        
        Args:
            device (Union[str, torch.device]): The device to move metrics to.
        
        Returns:
            MetricsManager: Self, for method chaining.
        """
        self.device = torch.device(device)
        for name, metric in self.metrics.items():
            if isinstance(metric, Metric):
                self.metrics[name] = metric.to(self.device)
        return self

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics, including their current values and configurations.

        Returns:
            Dict[str, Any]: A dictionary containing metric summaries.
        """
        return {
            name: self._get_metric_summary(name, metric)
            for name, metric in self.metrics.items()
        }

    def _get_metric_summary(self, name: str, metric: Union[Metric, Callable]) -> Dict[str, Any]:
        """Get summary for a single metric."""
        if isinstance(metric, Metric):
            return {
                'value': metric.compute().item(),
                'type': 'torchmetrics',
                'class': metric.__class__.__name__,
                'config': getattr(metric, '_defaults', {})
            }
        else:
            return {
                'value': np.mean(self.metric_values[name]) if self.metric_values[name] else None,
                'type': 'custom',
                'function': str(metric)
            }

    def log_metrics(self, step: int, logger: Optional[SummaryWriter] = None) -> None:
        """
        Log current metric values.

        Args:
            step (int): Current step or epoch number.
            logger (Optional[SummaryWriter]): TensorBoard SummaryWriter for logging.
        """
        metric_values = self.compute()
        for name, value in metric_values.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"Step {step}, {name}: {value:.4f}")
            if logger:
                logger.add_scalar(f"metrics/{name}", value, step)

    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """
        Get the best value for a specific metric and the step it occurred.

        Args:
            metric_name (str): Name of the metric.
            mode (str): 'min' for metrics where lower is better, 'max' for metrics where higher is better.

        Returns:
            Tuple[float, int]: The best metric value and the step it occurred.

        Raises:
            ValueError: If the metric name is not found or mode is invalid.
        """
        if metric_name not in self.metric_values:
            raise ValueError(f"Metric '{metric_name}' not found.")
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be either 'min' or 'max'.")

        values = self.metric_values[metric_name]
        if not values:
            return None, None

        if mode == 'min':
            best_value = min(values)
        else:
            best_value = max(values)

        best_step = values.index(best_value)
        return best_value, best_step

    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get the history of a specific metric.

        Args:
            metric_name (str): Name of the metric.

        Returns:
            List[float]: List of metric values over time.

        Raises:
            KeyError: If the metric name is not found.
        """
        if metric_name not in self.metric_values:
            raise KeyError(f"Metric '{metric_name}' not found.")
        return self.metric_values[metric_name]

    def plot_metric_history(self, metric_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot the history of a specific metric.

        Args:
            metric_name (str): Name of the metric to plot.
            save_path (Optional[str]): Path to save the plot. If None, the plot will be displayed.

        Raises:
            KeyError: If the metric name is not found.
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Please install it.")

        if metric_name not in self.metric_values:
            raise KeyError(f"Metric '{metric_name}' not found.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.metric_values[metric_name])
        plt.title(f"{metric_name} over time")
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class ModelStorageManager:
    """Manages model storage, including saving, loading, and versioning models."""

    def __init__(self, base_dir: str = "checkpoints", file_extension: str = ".pth"):
        self.base_dir = base_dir
        self.file_extension = file_extension
        self.version_file = os.path.join(self.base_dir, "version_info.json")
        self._version_info = self._load_or_create_version_info()
        os.makedirs(self.base_dir, exist_ok=True)

    class ModelStorageError(Exception):
        """Custom exception for ModelStorageManager errors."""
        pass

    def _load_or_create_version_info(self) -> Dict[str, Any]:
        """Load version information from file or create if not exists."""
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"current_version": 0, "models": {}}
        except json.JSONDecodeError:
            logger.warning("Invalid version info found. Resetting to default.")
            return {"current_version": 0, "models": {}}

    def _update_version_info(self) -> None:
        """Update the version information in the version file."""
        with open(self.version_file, 'w') as f:
            json.dump(self._version_info, f, indent=2)

    def save_model(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_params: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Save the model, optimizer state, training parameters, and metrics."""
        try:
            self._version_info["current_version"] += 1
            version = self._version_info["current_version"]
            filename = filename or f'model_v{version}_epoch_{epoch}{self.file_extension}'
            path = os.path.join(self.base_dir, filename)

            checkpoint = {
                'version': version,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_params': train_params,
                'metrics': metrics,
                'tags': tags or []
            }

            torch.save(checkpoint, path)

            self._version_info["models"][str(version)] = {
                "filename": filename,
                "epoch": epoch,
                "metrics": metrics,
                "tags": tags or []
            }
            self._update_version_info()

            logger.info(f"Model saved to {path}")
            return path
        except Exception as e:
            raise self.ModelStorageError(f"Error saving model to {path}: {str(e)}")

    def load_model(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        path: str,
        device: Union[str, torch.device] = 'cpu'
    ) -> Dict[str, Any]:
        """Load a saved model and return related information."""
        if not os.path.exists(path):
            raise self.ModelStorageError(f"No model found at {path}")

        try:
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {path}")

            return {
                'version': checkpoint['version'],
                'epoch': checkpoint['epoch'],
                'train_params': checkpoint['train_params'],
                'metrics': checkpoint['metrics'],
                'tags': checkpoint['tags']
            }
        except Exception as e:
            raise self.ModelStorageError(f"Error loading model from {path}: {str(e)}")

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """List all saved model files in the base directory with their information."""
        return list(self._version_info["models"].values())

    def delete_model(self, version: Union[int, str]) -> None:
        """Delete a saved model file."""
        version = str(version)
        if version not in self._version_info["models"]:
            raise self.ModelStorageError(f"No model found with version {version}")

        filename = self._version_info["models"][version]["filename"]
        path = os.path.join(self.base_dir, filename)

        try:
            os.remove(path)
            del self._version_info["models"][version]
            self._update_version_info()
            logger.info(f"Deleted model: {path}")
        except Exception as e:
            raise self.ModelStorageError(f"Error deleting model {path}: {str(e)}")

    def get_best_model(self, metric: str = 'val_loss', mode: str = 'min') -> Optional[str]:
        """Get the filename of the best model based on a specific metric."""
        if not self._version_info["models"]:
            return None

        key_func = lambda x: x['metrics'].get(metric, float('inf') if mode == 'min' else float('-inf'))
        best_model = min(self._version_info["models"].values(), key=key_func) if mode == 'min' else max(self._version_info["models"].values(), key=key_func)

        return best_model['filename']

    def get_latest_model(self) -> Optional[str]:
        """Get the filename of the latest saved model based on version number."""
        if not self._version_info["models"]:
            return None
        latest_version = max(map(int, self._version_info["models"].keys()))
        return self._version_info["models"][str(latest_version)]['filename']

    def to_torchscript(self, model: nn.Module, input_shape: tuple, filename: Optional[str] = None) -> str:
        """Convert the model to TorchScript and save it."""
        try:
            example_input = torch.randn(input_shape)
            traced_model = torch.jit.trace(model, example_input)

            filename = filename or f"{model.__class__.__name__}_torchscript.pt"
            path = os.path.join(self.base_dir, filename)
            torch.jit.save(traced_model, path)

            logger.info(f"TorchScript model saved to {path}")
            return path
        except Exception as e:
            raise self.ModelStorageError(f"Error converting model to TorchScript: {str(e)}")

    def load_torchscript(self, path: str) -> torch.jit.ScriptModule:
        """Load a TorchScript model."""
        if not os.path.exists(path):
            raise self.ModelStorageError(f"No TorchScript model found at {path}")

        try:
            model = torch.jit.load(path)
            logger.info(f"TorchScript model loaded from {path}")
            return model
        except Exception as e:
            raise self.ModelStorageError(f"Error loading TorchScript model from {path}: {str(e)}")

    def get_model_info(self, version: Union[int, str]) -> Dict[str, Any]:
        """Get information about a specific model version."""
        version = str(version)
        if version not in self._version_info["models"]:
            raise self.ModelStorageError(f"No model found with version {version}")
        return self._version_info["models"][version]

    def add_tag(self, version: Union[int, str], tag: str) -> None:
        """Add a tag to a specific model version."""
        version = str(version)
        if version not in self._version_info["models"]:
            raise self.ModelStorageError(f"No model found with version {version}")
        if tag not in self._version_info["models"][version]["tags"]:
            self._version_info["models"][version]["tags"].append(tag)
            self._update_version_info()
            logger.info(f"Added tag '{tag}' to model version {version}")

    def remove_tag(self, version: Union[int, str], tag: str) -> None:
        """Remove a tag from a specific model version."""
        version = str(version)
        if version not in self._version_info["models"]:
            raise self.ModelStorageError(f"No model found with version {version}")
        if tag in self._version_info["models"][version]["tags"]:
            self._version_info["models"][version]["tags"].remove(tag)
            self._update_version_info()
            logger.info(f"Removed tag '{tag}' from model version {version}")

    def find_models_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Find all models with a specific tag."""
        return [model for model in self._version_info["models"].values() if tag in model["tags"]]

    def compare_models(self, version1: Union[int, str], version2: Union[int, str]) -> Dict[str, Any]:
        """Compare two model versions."""
        v1, v2 = str(version1), str(version2)
        if v1 not in self._version_info["models"] or v2 not in self._version_info["models"]:
            raise self.ModelStorageError(f"One or both model versions not found")

        model1, model2 = self._version_info["models"][v1], self._version_info["models"][v2]
        comparison = {
            "version1": v1,
            "version2": v2,
            "epoch_diff": model2["epoch"] - model1["epoch"],
            "metric_diffs": {k: model2["metrics"].get(k, 0) - v for k, v in model1["metrics"].items()},
            "tag_diff": {
                "added": list(set(model2["tags"]) - set(model1["tags"])),
                "removed": list(set(model1["tags"]) - set(model2["tags"]))
            }
        }
        return comparison  
    
class TrainingParams(BaseModel):
    # Device and hardware settings
    device: torch.device = Field(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="Device to use for training"
    )
    use_mixed_precision: bool = Field(False, description="Whether to use mixed precision training")
    num_workers: int = Field(4, ge=0, description="Number of workers for data loading")

    # Optimization settings
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate for optimization")
    optimizer: str = Field("adam", description="Optimizer to use")
    optimizer_params: Dict[str, Any] = Field(default_factory=dict, description="Additional optimizer parameters")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay for regularization")
    clip_grad_norm: Optional[float] = Field(None, gt=0, description="Clip gradient norm if specified")

    # Training loop settings
    batch_size: int = Field(32, gt=0, description="Batch size for training")
    epochs: int = Field(10, gt=0, description="Number of training epochs")
    early_stopping: bool = Field(False, description="Whether to use early stopping")
    patience: int = Field(5, ge=0, description="Patience for early stopping")

    # Learning rate scheduler settings
    use_scheduler: bool = Field(False, description="Whether to use a learning rate scheduler")
    scheduler_type: Optional[str] = Field(None, description="Type of learning rate scheduler to use")
    scheduler_params: Dict[str, Any] = Field(default_factory=dict, description="Additional scheduler parameters")

    # Logging and checkpoint settings
    use_tensorboard: bool = Field(False, description="Whether to use TensorBoard for logging")
    checkpoint_dir: str = Field("checkpoints", description="Directory to save model checkpoints")
    log_interval: int = Field(100, gt=0, description="Interval for logging training progress")
    val_interval: int = Field(1, gt=0, description="Interval for validation (in epochs)")

    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for specific tasks")

    class Config:
        arbitrary_types_allowed = True

    OPTIMIZER_MAP: Dict[str, Type[Optimizer]] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad
    }

    SCHEDULER_MAP: Dict[str, Type[LRScheduler]] = {
        'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'step': torch.optim.lr_scheduler.StepLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'one_cycle': torch.optim.lr_scheduler.OneCycleLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'cyclic': torch.optim.lr_scheduler.CyclicLR
    }

    def get_optimizer(self, model_parameters) -> Optimizer:
        """Get the optimizer based on the specified parameters."""
        optimizer_class = self.OPTIMIZER_MAP[self.optimizer]
        optimizer_params = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            **self.optimizer_params
        }
        return optimizer_class(model_parameters, **optimizer_params)

    def get_scheduler(self, optimizer: Optimizer) -> Optional[LRScheduler]:
        """Get the learning rate scheduler based on the specified parameters."""
        if not self.use_scheduler:
            return None

        scheduler_class = self.SCHEDULER_MAP.get(self.scheduler_type)
        if scheduler_class is None:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        return scheduler_class(optimizer, **self.scheduler_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TrainingParams to a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'TrainingParams':
        """Create a TrainingParams instance from a dictionary."""
        return cls(**params_dict)

class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader],
        test_data_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        train_params: TrainingParams,
        metrics_config: List[Dict[str, Any]],
        progress_bar_class: type = tqdm,
        logger_class: type = SummaryWriter
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.loss_fn = loss_fn
        self.train_params = train_params
        self.metrics_config = metrics_config

        self.optimizer = self.train_params.get_optimizer(self.model.parameters())
        self.scheduler = self.train_params.get_scheduler(self.optimizer)
        self.scaler = GradScaler() if self.train_params.use_mixed_precision else None

        self.model_storage = ModelStorageManager(self.train_params.checkpoint_dir)
        self.logger = logger_class() if self.train_params.use_tensorboard else None
        self.metrics_manager = MetricsManager(metrics_config, device=self.train_params.device)
        self.progress_bar_class = progress_bar_class

        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self._move_to_device()

    def _move_to_device(self) -> None:
        self.model.to(self.train_params.device)
        self.loss_fn.to(self.train_params.device)

    def load_model(self, path: str) -> None:
        try:
            loaded_info = self.model_storage.load_model(self.model, self.optimizer, path, self.train_params.device)
            self.train_params = TrainingParams.from_dict(loaded_info['train_params'])
            logger.info(f"Model loaded from {path}")
            logger.info(f"Loaded model info: Epoch {loaded_info['epoch']}, Metrics: {loaded_info['metrics']}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _forward_pass(self, inputs: Tensor) -> Tensor:
        with autocast(enabled=self.train_params.use_mixed_precision):
            return self.model(inputs)

    def _compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        with autocast(enabled=self.train_params.use_mixed_precision):
            return self.loss_fn(outputs, targets)

    def _backward_pass(self, loss: Tensor) -> None:
        if self.train_params.use_mixed_precision:
            self._backward_pass_mixed_precision(loss)
        else:
            self._backward_pass_standard(loss)

    def _backward_pass_mixed_precision(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()
        if self.train_params.clip_grad_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _backward_pass_standard(self, loss: Tensor) -> None:
        loss.backward()
        if self.train_params.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.clip_grad_norm)
        self.optimizer.step()

    def _prepare_batch(self, batch: Any) -> Tuple[Tensor, Tensor]:
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

    def _step(self, batch: Any, training: bool = True) -> Dict[str, float]:
        inputs, targets = self._prepare_batch(batch)

        if training:
            self.optimizer.zero_grad()

        outputs = self._forward_pass(inputs)
        loss = self._compute_loss(outputs, targets)

        if training:
            self._backward_pass(loss)

        self.metrics_manager.update(outputs, targets)
        return {'loss': loss.item()}

    def train_step(self, batch: Any) -> Dict[str, float]:
        return self._step(batch, training=True)

    def val_step(self, batch: Any) -> Dict[str, float]:
        with torch.no_grad():
            return self._step(batch, training=False)

    def _run_epoch(self, data_loader: DataLoader, epoch: int, training: bool) -> Dict[str, float]:
        self.model.train() if training else self.model.eval()
        self.metrics_manager.reset()
        total_loss = 0.0
        num_batches = len(data_loader)

        phase = "train" if training else "val"
        desc = f"{'Training' if training else 'Validation'} Epoch {epoch+1}/{self.train_params.epochs}"
        pbar = self.progress_bar_class(total=num_batches, desc=desc)

        for i, batch in enumerate(data_loader):
            step_results = self.train_step(batch) if training else self.val_step(batch)
            total_loss += step_results['loss']

            if training and i % self.train_params.log_interval == 0:
                metrics = self.metrics_manager.compute()
                metrics['loss'] = step_results['loss']
                self._log_progress(phase, epoch, i, metrics)

            pbar.update(1)
            pbar.set_postfix(loss=f"{step_results['loss']:.4f}")

        pbar.close()

        metrics = self.metrics_manager.compute()
        metrics['loss'] = total_loss / num_batches
        return metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        return self._run_epoch(self.train_data_loader, epoch, training=True)

    def validate(self, epoch: int) -> Dict[str, float]:
        return self._run_epoch(self.val_data_loader, epoch, training=False)

    def train_loop(self) -> None:
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
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_best_model(epoch, val_loss)
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.train_params.early_stopping and self.patience_counter >= self.train_params.patience

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.model_storage.save_model(
            self.model, 
            self.optimizer, 
            self.train_params.to_dict(), 
            epoch, 
            metrics
        )

    def _save_best_model(self, epoch: int, val_loss: float) -> None:
        self.model_storage.save_model(
            self.model, 
            self.optimizer, 
            self.train_params.to_dict(), 
            epoch, 
            {'val_loss': val_loss}, 
            'best_model.pth'
        )

    def test_loop(self) -> Dict[str, float]:
        return self._run_epoch(self.test_data_loader, epoch=0, training=False)

    def _log_progress(self, phase: str, epoch: int, step: int, metrics: Dict[str, Union[float, Tensor]]) -> None:
        if self.logger:
            for metric_name, metric_value in metrics.items():
                self.logger.add_scalar(f"{phase}/{metric_name}", metric_value, epoch * len(self.train_data_loader) + step)

        log_str = f"{phase.capitalize()} Epoch {epoch+1}, Step {step}: "
        log_str += ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        logger.info(log_str)

    def train(self) -> Dict[str, float]:
        logger.info("Starting training process...")

        try:
            self.train_loop()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise

        logger.info("Training completed. Loading best model for final evaluation...")
        best_model_path = self.model_storage.get_best_model()
        if best_model_path:
            self.load_model(best_model_path)

        if self.test_data_loader:
            logger.info("Starting final evaluation on test set...")
            test_results = self.test_loop()
        else:
            logger.warning("No test data loader provided. Skipping final evaluation.")
            test_results = {}

        if self.logger:
            self.logger.close()

        return test_results

    def get_learning_rate(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def log_gradient_flow(self, model: nn.Module) -> None:
        """Log the gradient flow of the model."""
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in model.named_parameters():
            if p.requires_grad and ("bias" not in n) and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        
        if self.logger:
            for i, (layer, avg_grad, max_grad) in enumerate(zip(layers, ave_grads, max_grads)):
                self.logger.add_scalar(f"gradients/average/{layer}", avg_grad, i)
                self.logger.add_scalar(f"gradients/max/{layer}", max_grad, i)

    def log_model_parameters(self) -> None:
        """Log histograms of model parameters."""
        if self.logger:
            for name, param in self.model.named_parameters():
                self.logger.add_histogram(f"parameters/{name}", param.data, self.train_params.epochs)

    def visualize_model_graph(self) -> None:
        """Visualize the model graph using TensorBoard."""
        if self.logger:
            dummy_input = torch.randn(1, *self.train_data_loader.dataset[0][0].shape).to(self.train_params.device)
            self.logger.add_graph(self.model, dummy_input)

    def save_model_summary(self, file_path: str) -> None:
        """Save a summary of the model architecture to a file."""
        from torchinfo import summary
        with open(file_path, 'w') as f:
            dummy_input = torch.randn(1, *self.train_data_loader.dataset[0][0].shape).to(self.train_params.device)
            summary_str = str(summary(self.model, input_data=dummy_input, verbose=0))
            f.write(summary_str)

    def export_onnx(self, file_path: str) -> None:
        """Export the model to ONNX format."""
        dummy_input = torch.randn(1, *self.train_data_loader.dataset[0][0].shape).to(self.train_params.device)
        torch.onnx.export(self.model, dummy_input, file_path, verbose=True)
        logger.info(f"Model exported to ONNX format: {file_path}")

    def profile_model(self, num_steps: int = 100) -> None:
        """Profile the model performance."""
        from torch.profiler import profile, record_function, ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(num_steps):
                batch = next(iter(self.train_data_loader))
                with record_function("model_inference"):
                    _ = self.train_step(batch)

        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        if self.logger:
            self.logger.add_text("model_profile", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    def get_model_size(self) -> int:
        """Get the size of the model in bytes."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        logger.info(f"Model size: {size_all_mb:.2f} MB")
        return param_size + buffer_size

    def get_flops(self) -> int:
        """Estimate the number of FLOPs for the model."""
        from thop import profile as thop_profile
        dummy_input = torch.randn(1, *self.train_data_loader.dataset[0][0].shape).to(self.train_params.device)
        flops, _ = thop_profile(self.model, inputs=(dummy_input,))
        logger.info(f"Estimated FLOPs: {flops/1e9:.2f} G")
        return flops

    def on_train_start(self) -> None:
        """Hook method called before training starts."""
        pass

    def on_train_end(self) -> None:
        """Hook method called after training ends."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Hook method called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Hook method called at the end of each epoch."""
        pass

    def on_batch_start(self, batch: Any) -> None:
        """Hook method called at the start of each batch."""
        pass

    def on_batch_end(self, batch: Any, outputs: Any, loss: float) -> None:
        """Hook method called at the end of each batch."""
        pass
    
...