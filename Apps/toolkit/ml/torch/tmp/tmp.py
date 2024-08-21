import os
import time
from typing import Dict, Any, List, Union, Optional, Tuple, Callable, Type
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.utils.data as td
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization.quantizer import Quantizer as TorchQuantizer
import torch.ao.quantization.quantize_fx as tqfx
import torch.ao.quantization.quantize_pt2e as tqpt2e
import torch.ao.quantization.backend_config as backend_config
from torch.ao.quantization.qconfig import (
    default_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_dynamic_qconfig
)
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from torch._export import capture_pre_autograd_graph, dynamic_dim

from loguru import logger

class Backend(Enum):
    X86 = 'x86'
    FBGEMM = 'fbgemm'
    QNNPACK = 'qnnpack'

class QuantizationMethod(Enum):
    STATIC = 'static'
    DYNAMIC = 'dynamic'
    QAT = 'qat'
    PT2E_STATIC = 'pt2e_static'
    PT2E_QAT = 'pt2e_qat'

@dataclass
class Config:
    backend: Backend
    device: str
    method: QuantizationMethod
    calibration_batches: int
    evaluation_batches: int
    input_shape: List[int]
    batch_size: int
    num_samples: int
    log_level: str
    log_file: str
    learning_rate: float
    num_epochs: int
    save_path: str

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Config':
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            backend=Backend(data['backend']),
            device=data['device'],
            method=QuantizationMethod(data['method']),
            calibration_batches=data['calibration_batches'],
            evaluation_batches=data['evaluation_batches'],
            input_shape=data['input_shape'],
            batch_size=data['batch_size'],
            num_samples=data['num_samples'],
            log_level=data['log_level'],
            log_file=data['log_file'],
            learning_rate=data['learning_rate'],
            num_epochs=data['num_epochs'],
            save_path=data['save_path']
        )

class ConfigValidationError(Exception):
    pass

class ConfigValidator:
    @staticmethod
    def validate(config: Config) -> None:
        if config.calibration_batches <= 0:
            raise ConfigValidationError("calibration_batches must be positive")
        if config.evaluation_batches <= 0:
            raise ConfigValidationError("evaluation_batches must be positive")
        if len(config.input_shape) != 3:
            raise ConfigValidationError("input_shape must have 3 dimensions")
        if config.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        if config.num_samples <= 0:
            raise ConfigValidationError("num_samples must be positive")
        if config.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")
        if config.num_epochs <= 0:
            raise ConfigValidationError("num_epochs must be positive")

class QuantizationError(Exception):
    """Base class for quantization-related errors."""

class CalibrationError(QuantizationError):
    """Raised when there's an error during model calibration."""

class BackendError(QuantizationError):
    """Raised when there's an error related to backend configuration."""

@dataclass
class BackendConfig:
    supported_ops: List[str]
    dtype_configs: Dict[str, Any]

class BackendManager:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.backend_config = self._get_backend_config()

    def _get_backend_config(self) -> Any:
        if self.backend == Backend.X86:
            return backend_config.get_native_backend_config()
        elif self.backend == Backend.FBGEMM:
            return backend_config.get_fbgemm_backend_config()
        elif self.backend == Backend.QNNPACK:
            return backend_config.get_qnnpack_backend_config()
        else:
            raise BackendError(f"Unsupported backend: {self.backend}")

    def get_quantizer(self) -> TorchQuantizer:
        if self.backend == Backend.X86:
            return XNNPACKQuantizer()
        elif self.backend in [Backend.FBGEMM, Backend.QNNPACK]:
            raise NotImplementedError(f"Quantizer for {self.backend} is not implemented yet.")
        else:
            raise BackendError(f"Unsupported backend: {self.backend}")

class NonTraceableModule(nn.Module):
    def __init__(self, forward_func: Callable):
        super().__init__()
        self.forward_func = forward_func

    def forward(self, *args, **kwargs):
        return self.forward_func(*args, **kwargs)

class NumericSuite:
    @staticmethod
    def compare_weights(float_dict: Dict[str, torch.Tensor], quantized_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {key: {'float': float_dict[key], 'quantized': quantized_dict[key]} 
                for key in float_dict.keys() if key in quantized_dict}

    @staticmethod
    def compare_model_outputs(float_model: nn.Module, quantized_model: nn.Module, inputs: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        float_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            float_outputs = float_model(inputs)
            quantized_outputs = quantized_model(inputs)
        
        return {'output': {'float': float_outputs, 'quantized': quantized_outputs}}

    @staticmethod
    def prepare_model_with_stubs(float_model: nn.Module, quantized_model: nn.Module, module_swap_list: List[Union[str, nn.Module]]) -> nn.Module:
        for name, module in float_model.named_modules():
            if any(isinstance(module, swap_type) for swap_type in module_swap_list):
                setattr(float_model, name, quantized_model.get_submodule(name))
        return float_model

    @staticmethod
    def get_matching_activations(float_model: nn.Module, quantized_model: nn.Module, example_inputs: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        activations = {'float': {}, 'quantized': {}}

        def hook_fn(module_type):
            def fn(module, input, output):
                activations[module_type][module.name] = output.detach()
            return fn

        for name, module in float_model.named_modules():
            module.register_forward_hook(hook_fn('float'))

        for name, module in quantized_model.named_modules():
            module.register_forward_hook(hook_fn('quantized'))

        with torch.no_grad():
            float_model(example_inputs)
            quantized_model(example_inputs)

        return activations

    @staticmethod
    def visualize_quantization_effects(float_model: nn.Module, quantized_model: nn.Module):
        import matplotlib.pyplot as plt
        
        # Compare weights
        float_weights = dict(float_model.named_parameters())
        quantized_weights = dict(quantized_model.named_parameters())
        
        for name in float_weights.keys():
            if name in quantized_weights:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(float_weights[name].detach().numpy().flatten(), bins=50)
                plt.title(f"Float Weights: {name}")
                plt.subplot(1, 2, 2)
                plt.hist(quantized_weights[name].detach().numpy().flatten(), bins=50)
                plt.title(f"Quantized Weights: {name}")
                plt.tight_layout()
                plt.show()
        
        # Compare activations
        example_input = torch.randn(1, 3, 224, 224)  # Adjust input shape as needed
        float_activations = NumericSuite.get_matching_activations(float_model, quantized_model, example_input)
        
        for layer_name in float_activations['float'].keys():
            if layer_name in float_activations['quantized']:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(float_activations['float'][layer_name].numpy().flatten(), bins=50)
                plt.title(f"Float Activations: {layer_name}")
                plt.subplot(1, 2, 2)
                plt.hist(float_activations['quantized'][layer_name].numpy().flatten(), bins=50)
                plt.title(f"Quantized Activations: {layer_name}")
                plt.tight_layout()
                plt.show()

class ModelEvaluator:
    @staticmethod
    def print_model_size(model: nn.Module, label: str = "") -> int:
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        logger.info(f"Model: {label}\tSize (MB): {size/1e6:.2f}")
        os.remove('temp.p')
        return size

    @staticmethod
    def compare_model_sizes(fp32_model: nn.Module, int8_model: nn.Module) -> float:
        fp32_size = ModelEvaluator.print_model_size(fp32_model, "fp32")
        int8_size = ModelEvaluator.print_model_size(int8_model, "int8")
        reduction = fp32_size / int8_size
        logger.info(f"{reduction:.2f} times smaller")
        return reduction

    @staticmethod
    def compare_model_latency(
        fp32_model: nn.Module,
        int8_model: nn.Module,
        inputs: Any,
        device: str,
        n_iter: int = 100
    ) -> Dict[str, float]:
        def measure_latency(model: nn.Module, inputs: Any, n_iter: int) -> float:
            model = model.to(device)
            inputs = inputs[0].to(device) if isinstance(inputs, tuple) else inputs.to(device)
            
            if device.startswith('cuda'):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            else:
                start = time.time()
            
            with torch.no_grad():
                for _ in range(n_iter):
                    _ = model(inputs)
            
            if device.startswith('cuda'):
                ender.record()
                torch.cuda.synchronize()
                elapsed_time = starter.elapsed_time(ender) / 1000  # convert to seconds
            else:
                elapsed_time = time.time() - start
            
            return elapsed_time / n_iter

        fp32_latency = measure_latency(fp32_model, inputs, n_iter)
        int8_latency = measure_latency(int8_model, inputs, n_iter)
        
        logger.info(f"FP32 Latency: {fp32_latency:.6f} seconds")
        logger.info(f"INT8 Latency: {int8_latency:.6f} seconds")
        logger.info(f"Speedup: {fp32_latency / int8_latency:.2f}x")
        
        return {"fp32_latency": fp32_latency, "int8_latency": int8_latency}

    @staticmethod
    def compare_model_accuracy(
        fp32_model: nn.Module,
        int8_model: nn.Module,
        inputs: Any,
        device: str
    ) -> Dict[str, float]:
        fp32_model = fp32_model.to(device)
        int8_model = int8_model.to(device)
        inputs = inputs[0].to(device) if isinstance(inputs, tuple) else inputs.to(device)

        with torch.no_grad():
            fp32_output = fp32_model(inputs)
            int8_output = int8_model(inputs)

        if isinstance(fp32_output, tuple):
            fp32_output = fp32_output[0]
        if isinstance(int8_output, tuple):
            int8_output = int8_output[0]

        fp32_mag = torch.mean(torch.abs(fp32_output)).item()
        int8_mag = torch.mean(torch.abs(int8_output)).item()
        diff_mag = torch.mean(torch.abs(fp32_output - int8_output)).item()

        logger.info(f"Mean absolute value (FP32): {fp32_mag:.5f}")
        logger.info(f"Mean absolute value (INT8): {int8_mag:.5f}")
        logger.info(f"Mean absolute difference: {diff_mag:.5f} ({diff_mag/fp32_mag*100:.2f}%)")

        return {
            "fp32_magnitude": fp32_mag,
            "int8_magnitude": int8_mag,
            "difference_magnitude": diff_mag,
            "relative_difference": diff_mag / fp32_mag
        }

    @staticmethod
    def compare_accuracy_on_dataset(
        fp32_model: nn.Module,
        int8_model: nn.Module,
        dataloader: DataLoader,
        device: str
    ) -> Dict[str, float]:
        fp32_model.eval()
        int8_model.eval()
        fp32_correct = 0
        int8_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                fp32_outputs = fp32_model(inputs)
                int8_outputs = int8_model(inputs)
                
                _, fp32_predicted = torch.max(fp32_outputs, 1)
                _, int8_predicted = torch.max(int8_outputs, 1)
                
                fp32_correct += (fp32_predicted == labels).sum().item()
                int8_correct += (int8_predicted == labels).sum().item()
                total += labels.size(0)

        fp32_accuracy = fp32_correct / total
        int8_accuracy = int8_correct / total

        return {"fp32_accuracy": fp32_accuracy, "int8_accuracy": int8_accuracy}

    @staticmethod
    def evaluate_binary_classification(model: nn.Module, test_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    @staticmethod
    def evaluate_model(
        fp32_model: nn.Module, 
        int8_model: nn.Module, 
        example_inputs: Any, 
        device: str,
        eval_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        metrics = {}
        logger.info("Comparing model sizes")
        metrics['size_reduction'] = ModelEvaluator.compare_model_sizes(fp32_model, int8_model)
        logger.info("Comparing model latencies")
        metrics['latency'] = ModelEvaluator.compare_model_latency(fp32_model, int8_model, example_inputs, device)
        logger.info("Comparing model accuracies")
        metrics['accuracy'] = ModelEvaluator.compare_model_accuracy(fp32_model, int8_model, example_inputs, device)

        if eval_function:
            logger.info("Evaluating models with provided evaluation function")
            fp32_accuracy = eval_function(fp32_model)
            int8_accuracy = eval_function(int8_model)
            metrics['eval_accuracy'] = {
                'fp32': fp32_accuracy,
                'int8': int8_accuracy,
                'difference': fp32_accuracy - int8_accuracy
            }

        return metrics

    @staticmethod
    def profile_quantization(model: nn.Module, example_inputs: Any, quantization_method: 'QuantizationMethod') -> Dict[str, Any]:
        start_time = time.time()
        quantized_model = quantization_method.quantize(model, example_inputs)
        end_time = time.time()
        
        profiling_results = {
            "quantization_time": end_time - start_time,
            "original_model_size": ModelEvaluator.print_model_size(model, "Original"),
            "quantized_model_size": ModelEvaluator.print_model_size(quantized_model, "Quantized"),
        }
        
        return profiling_results

class QuantizationMethod(ABC):
    def __init__(self, quantizer: 'Quantizer'):
        self.quantizer = quantizer

    @abstractmethod
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        pass

class StaticQuantization(QuantizationMethod):
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.eval()
        logger.info("Preparing model for static quantization")
        prepared_model = tqfx.prepare_fx(model, self.quantizer.qconfig_mapping, example_inputs)
        logger.info("Calibrating model")
        self.quantizer.calibrate(prepared_model, example_inputs)
        logger.info("Converting model to static quantized version")
        return tqfx.convert_fx(prepared_model)

class DynamicQuantization(QuantizationMethod):
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        logger.info("Starting dynamic quantization")
        qconfig_spec = {
            nn.Linear: per_channel_dynamic_qconfig,
            nn.LSTM: per_channel_dynamic_qconfig,
            nn.Embedding: float_qparams_weight_only_qconfig
        }
        qconfig_mapping = tq.QConfigMapping().set_global(None)
        for module_type, qconfig in qconfig_spec.items():
            qconfig_mapping.set_object_type(module_type, qconfig)
        
        logger.info("Preparing model for dynamic quantization using FX")
        prepared_model = tqfx.prepare_fx(model, qconfig_mapping, example_inputs)
        logger.info("Converting model to dynamic quantized version using FX")
        return tqfx.convert_fx(prepared_model)

class QuantizationAwareTraining(QuantizationMethod):
    def __init__(self, quantizer: 'Quantizer', train_function: Callable):
        super().__init__(quantizer)
        self.train_function = train_function

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.train()
        logger.info("Preparing model for quantization-aware training")
        prepared_model = tqfx.prepare_qat_fx(model, self.quantizer.qconfig_mapping, example_inputs)
        logger.info("Starting quantization-aware training")
        self.train_function(prepared_model)  # User-provided training function
        prepared_model.eval()
        logger.info("Converting model to quantized version after QAT")
        return tqfx.convert_fx(prepared_model)

class PT2EStaticQuantization(QuantizationMethod):
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        logger.info("Starting PT2E static quantization")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_pt2e(exported_model, self.quantizer.quantizer)
        self.quantizer.calibrate(prepared_model, example_inputs)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

class PT2EQuantizationAwareTraining(QuantizationMethod):
    def __init__(self, quantizer: 'Quantizer', train_function: Callable):
        super().__init__(quantizer)
        self.train_function = train_function

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        logger.info("Starting PT2E quantization-aware training")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_qat_pt2e(exported_model, self.quantizer.quantizer)
        self.train_function(prepared_model)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

class Quantizer:
    def __init__(self, backend_manager: BackendManager, backend: Backend):
        self.backend_manager = backend_manager
        self.backend = backend
        self.qconfig_mapping = self._get_default_qconfig_mapping()
        self.quantizer = self.backend_manager.get_quantizer()
        self.numeric_suite = NumericSuite()

    def _get_default_qconfig_mapping(self) -> tq.QConfigMapping:
        return get_default_qconfig_mapping(self.backend.value)

    def set_qconfig_mapping(self, qconfig_mapping: tq.QConfigMapping) -> None:
        self.qconfig_mapping = qconfig_mapping

    def set_global_qconfig(self, is_qat: bool = False) -> None:
        try:
            self.qconfig_mapping.set_global(get_symmetric_quantization_config(is_qat=is_qat))
        except Exception as e:
            logger.error(f"Failed to set global qconfig: {str(e)}")
            raise QuantizationError("Failed to set global qconfig") from e
    
    def set_module_qconfig(self, module_type: Union[str, nn.Module], qconfig: Any) -> None:
        self.qconfig_mapping.set_object_type(module_type, qconfig)

    def fuse_modules(self, model: nn.Module, modules_to_fuse: List[List[str]]) -> nn.Module:
        logger.info(f"Fusing modules: {modules_to_fuse}")
        return tq.fuse_modules(model, modules_to_fuse)

    def calibrate(self, prepared_model: nn.Module, calibration_data: Any) -> None:
        logger.info("Calibrating model")
        prepared_model.eval()
        try:
            with torch.no_grad():
                for data in calibration_data:
                    prepared_model(*data)
        except Exception as e:
            logger.error(f"Failed to calibrate model: {str(e)}")
            raise CalibrationError("Failed to calibrate model") from e

    def handle_non_traceable(self, model: nn.Module) -> nn.Module:
        logger.info("Handling non-traceable parts of the model")
        for name, module in model.named_children():
            if isinstance(module, NonTraceableModule):
                logger.warning(f"Non-traceable module found: {name}")
                continue
            if not torch.jit.is_traceable(module):
                logger.warning(f"Module {name} is not traceable. Wrapping with NonTraceableModule.")
                setattr(model, name, NonTraceableModule(module.forward))
            else:
                setattr(model, name, self.handle_non_traceable(module))
        return model

    def determine_best_quantization_method(self, model: nn.Module, example_inputs: Any, comparison_results: Dict[str, Dict[str, Any]]) -> QuantizationMethod:
        logger.info("Determining the best quantization method")
        
        if not comparison_results:
            logger.warning("No comparison results available. Falling back to heuristic-based selection.")
            return self._heuristic_based_selection(model)
        
        # Define weights for each metric (can be adjusted based on priorities)
        weights = {
            'size_reduction': 0.3,
            'latency_speedup': 0.3,
            'accuracy_preservation': 0.4
        }
        
        scores = {}
        for method, results in comparison_results.items():
            size_reduction = results['size_reduction']
            latency_speedup = results['latency']['fp32_latency'] / results['latency']['int8_latency']
            accuracy_preservation = 1 - results['accuracy']['relative_difference']
            
            score = (
                weights['size_reduction'] * size_reduction +
                weights['latency_speedup'] * latency_speedup +
                weights['accuracy_preservation'] * accuracy_preservation
            )
            scores[method] = score
        
        best_method = max(scores, key=scores.get)
        logger.info(f"Best quantization method determined: {best_method}")
        return QuantizationMethod(best_method)

    def _heuristic_based_selection(self, model: nn.Module) -> QuantizationMethod:
        if any(isinstance(m, (nn.LSTM, nn.GRU)) for m in model.modules()):
            return QuantizationMethod.DYNAMIC
        elif any(isinstance(m, nn.Conv2d) for m in model.modules()):
            return QuantizationMethod.STATIC
        else:
            return QuantizationMethod.DYNAMIC

    def update_qconfig_mapping(self, model: nn.Module) -> None:
        logger.info("Updating qconfig mapping based on model architecture")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.set_module_qconfig(nn.Conv2d, per_channel_dynamic_qconfig)
            elif isinstance(module, nn.Linear):
                self.set_module_qconfig(nn.Linear, default_dynamic_qconfig)
        logger.info("QConfig mapping updated")

    def auto_fuse_modules(self, model: nn.Module) -> nn.Module:
        logger.info("Automatically fusing modules")
        fusion_patterns = [
            ['conv', 'bn'],
            ['conv', 'bn', 'relu'],
            ['conv', 'relu'],
            ['linear', 'relu'],
        ]
        for pattern in fusion_patterns:
            model = self.fuse_modules(model, pattern)
        return model

    def export_model(self, model: nn.Module, example_inputs: Any, dynamic: bool = False) -> nn.Module:
        logger.info("Exporting model")
        try:
            if dynamic:
                constraints = [dynamic_dim(example_inputs[0], 0)]
                return capture_pre_autograd_graph(model, example_inputs, constraints=constraints)
            else:
                return capture_pre_autograd_graph(model, example_inputs)
        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            raise QuantizationError("Failed to export model") from e
    
    def prepare_for_quantization(self, exported_model: nn.Module, is_qat: bool = False) -> nn.Module:
        logger.info("Preparing model for quantization")
        try:
            if is_qat:
                return tqpt2e.prepare_qat_pt2e(exported_model, self.quantizer)
            else:
                return tqpt2e.prepare_pt2e(exported_model, self.quantizer)
        except Exception as e:
            logger.error(f"Failed to prepare model for quantization: {str(e)}")
            raise QuantizationError("Failed to prepare model for quantization") from e

class QuantizationMethodFactory:
    @staticmethod
    def create(method: QuantizationMethod, quantizer: Quantizer, train_function: Optional[Callable] = None) -> QuantizationMethod:
        if method == QuantizationMethod.STATIC:
            return StaticQuantization(quantizer)
        elif method == QuantizationMethod.DYNAMIC:
            return DynamicQuantization(quantizer)
        elif method == QuantizationMethod.QAT:
            if train_function is None:
                raise ValueError("train_function must be provided for QAT")
            return QuantizationAwareTraining(quantizer, train_function)
        elif method == QuantizationMethod.PT2E_STATIC:
            return PT2EStaticQuantization(quantizer)
        elif method == QuantizationMethod.PT2E_QAT:
            if train_function is None:
                raise ValueError("train_function must be provided for PT2E QAT")
            return PT2EQuantizationAwareTraining(quantizer, train_function)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

class QuantizationWrapper:
    def __init__(self, config: Config):
        self.config = config
        logger.add(config.log_file, level=config.log_level)
        self.backend_manager = BackendManager(config.backend)
        self.quantizer = Quantizer(self.backend_manager, config.backend)
        self.model_evaluator = ModelEvaluator()

    def quantize_and_evaluate(
        self,
        model: nn.Module,
        example_inputs: Any,
        quantization_type: QuantizationMethod = QuantizationMethod.STATIC,
        calibration_data: Optional[Any] = None,
        eval_function: Optional[Callable] = None,
        train_function: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        logger.info(f"Starting {quantization_type.value} quantization")
        
        try:
            model = self._prepare_model(model)
            quantized_model = self._quantize_model(model, example_inputs, quantization_type, train_function)
            
            if calibration_data:
                self._calibrate_model(quantized_model, calibration_data)

            metrics = self._evaluate_model(model, quantized_model, example_inputs, eval_function)

            return quantized_model, metrics
        except Exception as e:
            logger.error(f"Error during quantization and evaluation: {str(e)}")
            raise

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = self.quantizer.handle_non_traceable(model)
        self.quantizer.update_qconfig_mapping(model)
        return self.quantizer.auto_fuse_modules(model)

    def _quantize_model(self, model: nn.Module, example_inputs: Any, quantization_type: QuantizationMethod, train_function: Optional[Callable]) -> nn.Module:
        quantization_method = QuantizationMethodFactory.create(
            quantization_type, 
            self.quantizer, 
            train_function
        )
        return quantization_method.quantize(model, example_inputs)

    def _calibrate_model(self, model: nn.Module, calibration_data: Any) -> None:
        self.quantizer.calibrate(model, calibration_data)

    def _evaluate_model(self, original_model: nn.Module, quantized_model: nn.Module, example_inputs: Any, eval_function: Optional[Callable]) -> Dict[str, Any]:
        return self.model_evaluator.evaluate_model(
            original_model, 
            quantized_model, 
            example_inputs, 
            self.config.device, 
            eval_function
        )

    def compare_quantization_methods(
        self, 
        model: nn.Module, 
        example_inputs: Any, 
        methods: List[QuantizationMethod],
        train_function: Optional[Callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for method in methods:
            quantization_method = QuantizationMethodFactory.create(method, self.quantizer, train_function)
            quantized_model = quantization_method.quantize(model, example_inputs)
            metrics = self.model_evaluator.evaluate_model(model, quantized_model, example_inputs, self.config.device)
            results[method.value] = metrics
        return results

    def visualize_comparison_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        methods = list(results.keys())
        size_reductions = [results[method]['size_reduction'] for method in methods]
        latency_speedups = [results[method]['latency']['fp32_latency'] / results[method]['latency']['int8_latency'] for method in methods]
        accuracy_diffs = [results[method]['accuracy']['relative_difference'] for method in methods]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.bar(methods, size_reductions)
        ax1.set_ylabel('Size Reduction Factor')
        ax1.set_title('Model Size Reduction')

        ax2.bar(methods, latency_speedups)
        ax2.set_ylabel('Latency Speedup Factor')
        ax2.set_title('Model Latency Speedup')

        ax3.bar(methods, accuracy_diffs)
        ax3.set_ylabel('Relative Accuracy Difference')
        ax3.set_title('Model Accuracy Difference')

        plt.tight_layout()
        plt.show()

    def auto_select_best_method(self, results: Dict[str, Dict[str, Any]]) -> QuantizationMethod:
        best_method = max(results, key=lambda x: results[x]['size_reduction'] * results[x]['latency']['fp32_latency'] / results[x]['latency']['int8_latency'] / (1 + results[x]['accuracy']['relative_difference']))
        logger.info(f"Automatically selected best method: {best_method}")
        return QuantizationMethod(best_method)

class SimpleCNN(nn.Module):
    def __init__(self, input_shape: List[int]):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (input_shape[1] // 4) * (input_shape[2] // 4), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)

def create_binary_dataset(config: Config) -> Tuple[TensorDataset, TensorDataset]:
    total_samples = config.num_samples
    train_samples = int(0.8 * total_samples)  # 80% for training
    test_samples = total_samples - train_samples

    def create_data(num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.randint(0, 2, (num_samples, *config.input_shape), dtype=torch.float32)
        labels = (images.view(num_samples, -1).sum(dim=1) > (images[0].numel() // 2)).float()
        return images, labels

    train_images, train_labels = create_data(train_samples)
    test_images, test_labels = create_data(test_samples)

    return TensorDataset(train_images, train_labels), TensorDataset(test_images, test_labels)

class ModelSaver:
    @staticmethod
    def save_model(model: nn.Module, path: str, use_jit: bool = True) -> None:
        if use_jit:
            torch.jit.save(torch.jit.script(model), path)
        else:
            torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(path: str, model_class: Type[nn.Module], use_jit: bool = True) -> nn.Module:
        if use_jit:
            return torch.jit.load(path)
        else:
            model = model_class()
            model.load_state_dict(torch.load(path))
            return model

def main():
    # Load configuration
    config = Config.from_yaml('config.yaml')
    ConfigValidator.validate(config)

    # Initialize QuantizationWrapper
    qw = QuantizationWrapper(config)

    # Create model
    model = SimpleCNN(config.input_shape)

    # Create datasets
    train_dataset, test_dataset = create_binary_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    # Get example inputs
    example_inputs, _ = next(iter(train_loader))

    # Define a simple training function for QAT
    def train_function(model: nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.BCELoss()
        model.train()
        for epoch in range(config.num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss: torch.Tensor = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

    # Compare different quantization methods
    methods_to_compare = [QuantizationMethod.STATIC, QuantizationMethod.DYNAMIC, QuantizationMethod.QAT, QuantizationMethod.PT2E_STATIC, QuantizationMethod.PT2E_QAT]
    comparison_results = qw.compare_quantization_methods(model, example_inputs, methods_to_compare, train_function)
    logger.info("Quantization method comparison results:")
    for method, metrics in comparison_results.items():
        logger.info(f"{method}: {metrics}")

    # Visualize comparison results
    qw.visualize_comparison_results(comparison_results)

    # Automatically select the best method
    best_method = qw.auto_select_best_method(comparison_results)

    # Quantize and evaluate the model using the best method
    quantized_model, metrics = qw.quantize_and_evaluate(
        model,
        example_inputs,
        quantization_type=best_method,
        calibration_data=train_loader,
        eval_function=lambda m: ModelEvaluator.evaluate_binary_classification(m, test_loader),
        train_function=train_function
    )

    logger.info("Quantization complete. Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    # Save the quantized model
    ModelSaver.save_model(quantized_model, config.save_path)
    logger.info(f"Quantized model saved to {config.save_path}")

    # Load the quantized model
    loaded_model = ModelSaver.load_model(config.save_path, SimpleCNN)
    logger.info(f"Loaded quantized model from {config.save_path}")

    # Compare the loaded model with the original quantized model
    loaded_accuracy = ModelEvaluator.evaluate_binary_classification(loaded_model, test_loader)
    original_accuracy = ModelEvaluator.evaluate_binary_classification(quantized_model, test_loader)
    logger.info(f"Original quantized model accuracy: {original_accuracy:.4f}")
    logger.info(f"Loaded quantized model accuracy: {loaded_accuracy:.4f}")

    # Visualize quantization effects
    NumericSuite.visualize_quantization_effects(model, quantized_model)

    # Profile quantization
    profiling_results = ModelEvaluator.profile_quantization(
        model, 
        example_inputs, 
        QuantizationMethodFactory.create(best_method, qw.quantizer, train_function)
    )
    logger.info("Quantization profiling results:")
    for key, value in profiling_results.items():
        logger.info(f"{key}: {value}")

    logger.info("Quantization process completed successfully.")

if __name__ == "__main__":
    main()