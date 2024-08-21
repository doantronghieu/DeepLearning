import os
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
import logging
import torch
import torch.nn as nn
import torch.ao.quantization as tq
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
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class QuantizationError(Exception):
    """Base class for quantization-related errors."""
    pass

class CalibrationError(QuantizationError):
    """Raised when there's an error during model calibration."""
    pass

class BackendError(QuantizationError):
    """Raised when there's an error related to backend configuration."""
    pass

class Logger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

@dataclass
class BackendConfig:
    supported_ops: List[str]
    dtype_configs: Dict[str, Any]

class BackendManager:
    def __init__(self, backend: str):
        self.backend = backend
        self.backend_config = self._get_backend_config()

    def _get_backend_config(self) -> Any:
        if self.backend == 'x86':
            return backend_config.get_native_backend_config()
        elif self.backend == 'fbgemm':
            return backend_config.get_fbgemm_backend_config()
        elif self.backend == 'qnnpack':
            return backend_config.get_qnnpack_backend_config()
        else:
            raise BackendError(f"Unsupported backend: {self.backend}")

    def get_quantizer(self) -> TorchQuantizer:
        if self.backend == 'x86':
            return XNNPACKQuantizer()
        elif self.backend in ['fbgemm', 'qnnpack']:
            raise NotImplementedError(f"Quantizer for {self.backend} is not implemented yet.")
        else:
            raise BackendError(f"Unsupported backend: {self.backend}")

    def create_custom_backend_config(self, config: BackendConfig) -> BackendConfig:
        return BackendConfig(**config.__dict__)

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

class ModelEvaluator:
    @staticmethod
    def print_model_size(model: nn.Module, label: str = "") -> int:
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        print(f"Model: {label}\tSize (MB): {size/1e6:.2f}")
        os.remove('temp.p')
        return size

    @staticmethod
    def compare_model_sizes(fp32_model: nn.Module, int8_model: nn.Module) -> float:
        fp32_size = ModelEvaluator.print_model_size(fp32_model, "fp32")
        int8_size = ModelEvaluator.print_model_size(int8_model, "int8")
        reduction = fp32_size / int8_size
        print(f"{reduction:.2f} times smaller")
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
                start = torch.backends.mkldnn.time.time()
            
            with torch.no_grad():
                for _ in range(n_iter):
                    _ = model(inputs)
            
            if device.startswith('cuda'):
                ender.record()
                torch.cuda.synchronize()
                elapsed_time = starter.elapsed_time(ender) / 1000  # convert to seconds
            else:
                elapsed_time = torch.backends.mkldnn.time.time() - start
            
            return elapsed_time / n_iter

        fp32_latency = measure_latency(fp32_model, inputs, n_iter)
        int8_latency = measure_latency(int8_model, inputs, n_iter)
        
        print(f"FP32 Latency: {fp32_latency:.6f} seconds")
        print(f"INT8 Latency: {int8_latency:.6f} seconds")
        print(f"Speedup: {fp32_latency / int8_latency:.2f}x")
        
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

        print(f"Mean absolute value (FP32): {fp32_mag:.5f}")
        print(f"Mean absolute value (INT8): {int8_mag:.5f}")
        print(f"Mean absolute difference: {diff_mag:.5f} ({diff_mag/fp32_mag*100:.2f}%)")

        return {
            "fp32_magnitude": fp32_mag,
            "int8_magnitude": int8_mag,
            "difference_magnitude": diff_mag,
            "relative_difference": diff_mag / fp32_mag
        }

    @staticmethod
    def visualize_model(model: nn.Module) -> None:
        print("Model structure visualization:")
        print(model)

    @staticmethod
    def compare_accuracy_on_dataset(
        fp32_model: nn.Module,
        int8_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
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

class QuantizationConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

class QuantizationMethodRegistry:
    def __init__(self):
        self.methods = {}

    def register(self, name: str, method: 'QuantizationMethod'):
        self.methods[name] = method

    def get(self, name: str) -> 'QuantizationMethod':
        if name not in self.methods:
            raise ValueError(f"Unknown quantization method: {name}")
        return self.methods[name]

class Quantizer:
    def __init__(self, backend_manager: BackendManager, logger: Logger):
        self.backend_manager = backend_manager
        self.logger = logger
        self.qconfig_mapping = self._get_default_qconfig_mapping()
        self.quantizer = self.backend_manager.get_quantizer()
        self.numeric_suite = NumericSuite()

    def _get_default_qconfig_mapping(self) -> tq.QConfigMapping:
        return get_default_qconfig_mapping(self.backend_manager.backend)

    def set_qconfig_mapping(self, qconfig_mapping: tq.QConfigMapping) -> None:
        self.qconfig_mapping = qconfig_mapping

    def set_global_qconfig(self, is_qat: bool = False) -> None:
        try:
            self.qconfig_mapping.set_global(get_symmetric_quantization_config(is_qat=is_qat))
        except Exception as e:
            self.logger.error(f"Failed to set global qconfig: {str(e)}")
    
    def set_module_qconfig(self, module_type: Union[str, nn.Module], qconfig: Any) -> None:
        self.qconfig_mapping.set_object_type(module_type, qconfig)

    def fuse_modules(self, model: nn.Module, modules_to_fuse: List[List[str]]) -> nn.Module:
        self.logger.info(f"Fusing modules: {modules_to_fuse}")
        return tq.fuse_modules(model, modules_to_fuse)

    def export_model(self, model: nn.Module, example_inputs: Any, dynamic: bool = False) -> nn.Module:
        self.logger.info("Exporting model")
        try:
            if dynamic:
                constraints = [dynamic_dim(example_inputs[0], 0)]
                return capture_pre_autograd_graph(model, example_inputs, constraints=constraints)
            else:
                return capture_pre_autograd_graph(model, example_inputs)
        except Exception as e:
            self.logger.error(f"Failed to export model: {str(e)}")
            raise QuantizationError("Failed to export model") from e

    def prepare_for_quantization(self, exported_model: nn.Module, is_qat: bool = False) -> nn.Module:
        self.logger.info("Preparing model for quantization")
        try:
            if is_qat:
                return tqpt2e.prepare_qat_pt2e(exported_model, self.quantizer)
            else:
                return tqpt2e.prepare_pt2e(exported_model, self.quantizer)
        except Exception as e:
            self.logger.error(f"Failed to prepare model for quantization: {str(e)}")
            raise QuantizationError("Failed to prepare model for quantization") from e

    def dynamic_quantize(
        self,
        model: nn.Module,
        example_inputs: Any,
        qconfig_spec: Optional[Dict[Any, Any]] = None,
        use_fx: bool = True,
        use_per_channel: bool = False
    ) -> nn.Module:
        self.logger.info("Starting dynamic quantization")
        
        if use_fx:
            if qconfig_spec is None:
                qconfig_spec = {
                    nn.Linear: per_channel_dynamic_qconfig if use_per_channel else default_dynamic_qconfig,
                    nn.LSTM: per_channel_dynamic_qconfig if use_per_channel else default_dynamic_qconfig,
                    nn.Embedding: float_qparams_weight_only_qconfig
                }
            qconfig_mapping = tq.QConfigMapping().set_global(None)
            for module_type, qconfig in qconfig_spec.items():
                qconfig_mapping.set_object_type(module_type, qconfig)
            
            self.logger.info("Preparing model for dynamic quantization using FX")
            prepared_model = tqfx.prepare_fx(model, qconfig_mapping, example_inputs)
            self.logger.info("Converting model to dynamic quantized version using FX")
            return tqfx.convert_fx(prepared_model)
        else:
            self.logger.info("Applying dynamic quantization using torch.quantization.quantize_dynamic")
            dtype = torch.qint8
            qconfig_spec = qconfig_spec or {'': default_dynamic_qconfig}
            return tq.quantize_dynamic(model, qconfig_spec=qconfig_spec, dtype=dtype)

    def static_quantize(self, model: nn.Module, example_inputs: Any, inplace: bool = False) -> nn.Module:
        model.eval()
        self.logger.info("Preparing model for static quantization")
        prepared_model = tqfx.prepare_fx(model, self.qconfig_mapping, example_inputs)
        self.logger.info("Calibrating model")
        self._calibrate(prepared_model, example_inputs)
        self.logger.info("Converting model to static quantized version")
        return tqfx.convert_fx(prepared_model, inplace=inplace)

    def _calibrate(self, prepared_model: nn.Module, example_inputs: Any) -> None:
        prepared_model.eval()
        with torch.no_grad():
            prepared_model(*example_inputs)

    def weight_only_quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        weight_only_qconfig_mapping = tq.QConfigMapping().set_global(float_qparams_weight_only_qconfig)
        self.logger.info("Preparing model for weight-only quantization")
        prepared_model = tqfx.prepare_fx(model, weight_only_qconfig_mapping, example_inputs)
        self.logger.info("Converting model to weight-only quantized version")
        return tqfx.convert_fx(prepared_model)

    def quantize_model(self, prepared_model: nn.Module) -> nn.Module:
        self.logger.info("Converting model to quantized version")
        try:
            quantized_model = tqpt2e.convert_pt2e(prepared_model)
            tq.move_exported_model_to_eval(quantized_model)
            return quantized_model
        except Exception as e:
            self.logger.error(f"Failed to quantize model: {str(e)}")
            raise QuantizationError("Failed to quantize model") from e

    def calibrate(self, prepared_model: nn.Module, calibration_data: Any) -> None:
        self.logger.info("Calibrating model")
        prepared_model.eval()
        try:
            with torch.no_grad():
                for data in calibration_data:
                    prepared_model(*data)
        except Exception as e:
            self.logger.error(f"Failed to calibrate model: {str(e)}")
            raise CalibrationError("Failed to calibrate model") from e

    def quantization_aware_training(
        self,
        model: nn.Module,
        example_inputs: Any,
        train_function: Callable
    ) -> nn.Module:
        model.train()
        self.logger.info("Preparing model for quantization-aware training")
        prepared_model = tqfx.prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
        self.logger.info("Starting quantization-aware training")
        train_function(prepared_model)  # User-provided training function
        prepared_model.eval()
        self.logger.info("Converting model to quantized version after QAT")
        return tqfx.convert_fx(prepared_model)

    def pt2e_static_quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        self.logger.info("Starting PT2E static quantization")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_pt2e(exported_model, self.quantizer)
        self._calibrate(prepared_model, example_inputs)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

    def pt2e_quantization_aware_training(
        self,
        model: nn.Module,
        example_inputs: Any,
        train_function: Callable
    ) -> nn.Module:
        self.logger.info("Starting PT2E quantization-aware training")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_qat_pt2e(exported_model, self.quantizer)
        train_function(prepared_model)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

    def add_custom_quantization_pattern(self, pattern: Union[nn.Module, List[nn.Module]], qconfig: Any) -> None:
        try:
            self.quantizer.set_module_type(pattern, qconfig)
        except Exception as e:
            self.logger.error(f"Failed to add custom quantization pattern: {str(e)}")
            raise QuantizationError("Failed to add custom quantization pattern") from e

    def handle_non_traceable(self, model: nn.Module) -> nn.Module:
        self.logger.info("Handling non-traceable parts of the model")
        for name, module in model.named_children():
            if isinstance(module, NonTraceableModule):
                self.logger.warning(f"Non-traceable module found: {name}")
                continue
            if not torch.jit.is_traceable(module):
                self.logger.warning(f"Module {name} is not traceable. Wrapping with NonTraceableModule.")
                setattr(model, name, NonTraceableModule(module.forward))
            else:
                setattr(model, name, self.handle_non_traceable(module))
        return model

    def validate_calibration_data(self, calibration_data: Any) -> List[Tuple[torch.Tensor, ...]]:
        if not isinstance(calibration_data, (list, tuple, DataLoader)):
            raise ValueError("Calibration data must be a list, tuple, or DataLoader")
        
        validated_data = []
        for batch in calibration_data:
            if not isinstance(batch, (tuple, list)):
                batch = (batch,)
            validated_data.append(tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch))
        
        return validated_data

    def calibrate_with_validated_data(self, prepared_model: nn.Module, calibration_data: List[Tuple[torch.Tensor, ...]]) -> None:
        self.logger.info("Calibrating model with validated data")
        prepared_model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                prepared_model(*batch)

    def determine_best_quantization_method(self, model: nn.Module, example_inputs: Any) -> str:
        self.logger.info("Determining the best quantization method")
        # This is a simple heuristic and can be expanded based on more complex analysis
        if any(isinstance(m, (nn.LSTM, nn.GRU)) for m in model.modules()):
            return 'dynamic'
        elif any(isinstance(m, nn.Conv2d) for m in model.modules()):
            return 'static'
        else:
            return 'weight_only'

    def visualize_quantization_effects(self, fp32_model: nn.Module, int8_model: nn.Module, example_inputs: Any) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib is required for visualization. Please install it.")
            return

        fp32_output = fp32_model(example_inputs).detach().flatten()
        int8_output = int8_model(example_inputs).detach().flatten()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(fp32_output.numpy(), bins=50, alpha=0.5, label='FP32')
        plt.hist(int8_output.numpy(), bins=50, alpha=0.5, label='INT8')
        plt.legend()
        plt.title('Output Distribution')

        plt.subplot(1, 2, 2)
        plt.scatter(fp32_output.numpy(), int8_output.numpy(), alpha=0.5)
        plt.xlabel('FP32 Output')
        plt.ylabel('INT8 Output')
        plt.title('FP32 vs INT8 Output')

        plt.tight_layout()
        plt.show()

    def perform_sensitivity_analysis(self, model: nn.Module, example_inputs: Any) -> Dict[str, float]:
        self.logger.info("Performing sensitivity analysis")
        sensitivities = {}
        original_output = model(example_inputs)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Temporarily quantize the module
                qconfig = torch.quantization.get_default_qconfig('fbgemm')
                qmodule = torch.quantization.quantize_dynamic(module, qconfig)
                
                # Replace the original module with the quantized one
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, qmodule)
                else:
                    setattr(model, child_name, qmodule)
                
                # Compute the output and sensitivity
                quantized_output = model(example_inputs)
                sensitivity = torch.mean(torch.abs(original_output - quantized_output)).item()
                sensitivities[name] = sensitivity
                
                # Restore the original module
                if parent_name:
                    setattr(parent, child_name, module)
                else:
                    setattr(model, child_name, module)

        return sensitivities

class QuantizationMethod(ABC):
    @abstractmethod
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        pass

class StaticQuantization(QuantizationMethod):
    def __init__(self, quantizer: Quantizer):
        self.quantizer = quantizer

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        return self.quantizer.static_quantize(model, example_inputs)

class DynamicQuantization(QuantizationMethod):
    def __init__(self, quantizer: Quantizer):
        self.quantizer = quantizer

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        return self.quantizer.dynamic_quantize(model, example_inputs)

class QuantizationAwareTraining(QuantizationMethod):
    def __init__(self, quantizer: Quantizer, train_function: Callable):
        self.quantizer = quantizer
        self.train_function = train_function

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        return self.quantizer.quantization_aware_training(model, example_inputs, self.train_function)

class PT2EStaticQuantization(QuantizationMethod):
    def __init__(self, quantizer: Quantizer):
        self.quantizer = quantizer

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        return self.quantizer.pt2e_static_quantize(model, example_inputs)

class PT2EQuantizationAwareTraining(QuantizationMethod):
    def __init__(self, quantizer: Quantizer, train_function: Callable):
        self.quantizer = quantizer
        self.train_function = train_function

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        return self.quantizer.pt2e_quantization_aware_training(model, example_inputs, self.train_function)

class QuantizationWrapper:
    def __init__(self, config: QuantizationConfig, backend: str = 'x86', device: str = 'cpu'):
        self.config = config
        self.device = device
        self.logger = Logger('QuantizationWrapper')
        self.backend_manager = BackendManager(backend)
        self.quantizer = Quantizer(self.backend_manager, self.logger)
        self.method_registry = QuantizationMethodRegistry()
        self._register_quantization_methods()

    def _register_quantization_methods(self):
        self.method_registry.register('static', StaticQuantization(self.quantizer))
        self.method_registry.register('dynamic', DynamicQuantization(self.quantizer))
        self.method_registry.register('pt2e_static', PT2EStaticQuantization(self.quantizer))

    def register_qat_method(self, train_function: Callable):
        self.method_registry.register('qat', QuantizationAwareTraining(self.quantizer, train_function))
        self.method_registry.register('pt2e_qat', PT2EQuantizationAwareTraining(self.quantizer, train_function))

    def quantize_and_evaluate(
        self,
        model: nn.Module,
        example_inputs: Any,
        quantization_type: str = 'static',
        calibration_data: Optional[Any] = None,
        eval_function: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        self.logger.info(f"Starting {quantization_type} quantization")
        
        try:
            quantization_method = self.method_registry.get(quantization_type)
            model = self.quantizer.handle_non_traceable(model)
            quantized_model = quantization_method.quantize(model, example_inputs)

            if calibration_data:
                validated_calibration_data = self.quantizer.validate_calibration_data(calibration_data)
                self.quantizer.calibrate_with_validated_data(quantized_model, validated_calibration_data)

            metrics = self._evaluate_model(model, quantized_model, example_inputs, eval_function)

            return quantized_model, metrics
        except Exception as e:
            self.logger.error(f"Error during quantization and evaluation: {str(e)}")
            raise

    def _evaluate_model(self, fp32_model: nn.Module, int8_model: nn.Module, example_inputs: Any, eval_function: Optional[Callable]) -> Dict[str, Any]:
        metrics = {}
        self.logger.info("Comparing model sizes")
        metrics['size_reduction'] = ModelEvaluator.compare_model_sizes(fp32_model, int8_model)
        self.logger.info("Comparing model latencies")
        metrics['latency'] = ModelEvaluator.compare_model_latency(fp32_model, int8_model, example_inputs, self.device)
        self.logger.info("Comparing model accuracies")
        metrics['accuracy'] = ModelEvaluator.compare_model_accuracy(fp32_model, int8_model, example_inputs, self.device)

        if eval_function:
            self.logger.info("Evaluating models with provided evaluation function")
            fp32_accuracy = eval_function(fp32_model)
            int8_accuracy = eval_function(int8_model)
            metrics['eval_accuracy'] = {
                'fp32': fp32_accuracy,
                'int8': int8_accuracy,
                'difference': fp32_accuracy - int8_accuracy
            }

        return metrics

    def save_quantized_model(self, model: nn.Module, path: str) -> None:
        self.logger.info(f"Saving quantized model to {path}")
        try:
            torch.jit.save(torch.jit.script(model), path)
        except Exception as e:
            self.logger.error(f"Failed to save quantized model: {str(e)}")
            raise

    def load_quantized_model(self, path: str) -> nn.Module:
        self.logger.info(f"Loading quantized model from {path}")
        try:
            return torch.jit.load(path)
        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {str(e)}")
            raise
    
    def create_custom_backend_config(self, config: BackendConfig) -> BackendConfig:
        return self.backend_manager.create_custom_backend_config(config)

    def compare_weights(self, float_model: nn.Module, quantized_model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.quantizer.numeric_suite.compare_weights(float_model.state_dict(), quantized_model.state_dict())

    def compare_model_outputs(self, float_model: nn.Module, quantized_model: nn.Module, inputs: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.quantizer.numeric_suite.compare_model_outputs(float_model, quantized_model, inputs)

    def prepare_model_with_stubs(self, float_model: nn.Module, quantized_model: nn.Module, module_swap_list: List[Union[str, nn.Module]]) -> nn.Module:
        return self.quantizer.numeric_suite.prepare_model_with_stubs(float_model, quantized_model, module_swap_list)

    def get_matching_activations(self, float_model: nn.Module, quantized_model: nn.Module, example_inputs: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.quantizer.numeric_suite.get_matching_activations(float_model, quantized_model, example_inputs)

    def compare_accuracy_on_dataset(
        self,
        fp32_model: nn.Module,
        int8_model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        return ModelEvaluator.compare_accuracy_on_dataset(fp32_model, int8_model, dataloader, self.device)
    
    def visualize_model(self, model: nn.Module, save_path: Optional[str] = None) -> None:
        try:
            from torchviz import make_dot
            from graphviz import Source

            x = torch.randn(1, 3, 224, 224).to(self.device)
            y = model(x)
            dot = make_dot(y, params=dict(model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
                self.logger.info(f"Model visualization saved to {save_path}.png")
            else:
                dot.view()
        except ImportError:
            self.logger.warning("torchviz and graphviz are required for advanced model visualization. "
                                "Please install them using: pip install torchviz graphviz")
            ModelEvaluator.visualize_model(model)

    def determine_best_quantization_method(self, model: nn.Module, example_inputs: Any) -> str:
        return self.quantizer.determine_best_quantization_method(model, example_inputs)

    def visualize_quantization_effects(self, fp32_model: nn.Module, int8_model: nn.Module, example_inputs: Any) -> None:
        self.quantizer.visualize_quantization_effects(fp32_model, int8_model, example_inputs)

    def perform_sensitivity_analysis(self, model: nn.Module, example_inputs: Any) -> Dict[str, float]:
        return self.quantizer.perform_sensitivity_analysis(model, example_inputs)

# Example usage
if __name__ == "__main__":
    # Create a simple model with a non-traceable part
    class NonTraceableActivation(nn.Module):
        def forward(self, x):
            return torch.where(x > 0, x, torch.zeros_like(x))

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        NonTraceableActivation(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10)
    )

    # Create example inputs
    example_inputs = torch.randn(1, 3, 32, 32)

    # Create a configuration
    config = QuantizationConfig({
        'quantization_type': 'static',
        'calibration_batch_size': 32,
        'evaluation_batch_size': 64,
        'num_calibration_batches': 10,
        'num_evaluation_batches': 20,
    })

    # Initialize QuantizationWrapper
    qw = QuantizationWrapper(config, backend='x86', device='cpu')

    # Define a custom evaluation function
    def custom_eval_function(model):
        # This is a placeholder for a real evaluation function
        return 0.95  # Dummy accuracy

    # Create calibration data
    calibration_data = [(torch.randn(1, 3, 32, 32),) for _ in range(config.get('num_calibration_batches', 10))]

    # Determine the best quantization method
    best_method = qw.determine_best_quantization_method(model, example_inputs)
    print(f"Best quantization method: {best_method}")

    # Define a dummy QAT function and register it
    def dummy_qat_function(model):
        # This is a placeholder for a real QAT function
        pass

    qw.register_qat_method(dummy_qat_function)

    # Quantize and evaluate the model
    quantized_model, metrics = qw.quantize_and_evaluate(
        model,
        example_inputs,
        quantization_type=best_method,
        calibration_data=calibration_data,
        eval_function=custom_eval_function
    )

    print("Quantization complete. Metrics:", metrics)

    # Perform sensitivity analysis
    sensitivities = qw.perform_sensitivity_analysis(model, example_inputs)
    print("Layer sensitivities:", sensitivities)

    # Visualize quantization effects
    qw.visualize_quantization_effects(model, quantized_model, example_inputs)

    # Use NumericSuite to compare weights and activations
    weight_comparison = qw.compare_weights(model, quantized_model)
    print("Weight comparison:", weight_comparison)

    output_comparison = qw.compare_model_outputs(model, quantized_model, example_inputs)
    print("Output comparison:", output_comparison)

    # Prepare model with stubs
    stubbed_model = qw.prepare_model_with_stubs(model, quantized_model, [nn.Conv2d, nn.Linear])
    print("Model prepared with stubs")

    # Get matching activations
    activations = qw.get_matching_activations(model, quantized_model, example_inputs)
    print("Matching activations:", activations)

    # Create a custom backend config
    custom_config = BackendConfig(
        supported_ops=['conv2d', 'linear', 'relu'],
        dtype_configs={'int8': {'weight': True, 'activation': True}}
    )
    custom_backend_config = qw.create_custom_backend_config(custom_config)
    print("Custom backend config created:", custom_backend_config)

    # Visualize the model
    qw.visualize_model(quantized_model, save_path='quantized_model_viz')

    # Save and load the quantized model
    qw.save_quantized_model(quantized_model, "quantized_model.pt")
    loaded_model = qw.load_quantized_model("quantized_model.pt")
    print("Model saved and loaded successfully.")

    print("Quantization framework demonstration complete.")