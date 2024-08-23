# My code
import os
import time
import copy
import re
from loguru import logger
from typing import Dict, Any, List, Union, Optional, Tuple, Callable, Type
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict, field
from enum import Enum
import concurrent.futures
from tqdm import tqdm

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    custom_options: Dict[str, Any] = field(default_factory=dict)
    quantization_bit_width: int = 8
    quantization_scheme: str = 'symmetric'

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Config':
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_file: str) -> None:
        config_dict = asdict(self)
        config_dict['backend'] = self.backend.value
        config_dict['method'] = self.method.value
        with open(yaml_file, 'w') as f:
            yaml.dump(config_dict, f)

    @classmethod
    def load(cls, yaml_file: str) -> 'Config':
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        data['backend'] = Backend(data['backend'])
        data['method'] = QuantizationMethod(data['method'])
        return cls(**data)

    def validate(self) -> None:
        if self.calibration_batches <= 0:
            raise ConfigValidationError("calibration_batches must be positive")
        if self.evaluation_batches <= 0:
            raise ConfigValidationError("evaluation_batches must be positive")
        if len(self.input_shape) != 3:
            raise ConfigValidationError("input_shape must have 3 dimensions")
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        if self.num_samples <= 0:
            raise ConfigValidationError("num_samples must be positive")
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ConfigValidationError("num_epochs must be positive")
        if self.quantization_bit_width not in [4, 8]:
            raise ConfigValidationError("quantization_bit_width must be either 4 or 8")
        if self.quantization_scheme not in ['symmetric', 'asymmetric']:
            raise ConfigValidationError("quantization_scheme must be either 'symmetric' or 'asymmetric'")
        
        if self.backend == Backend.QNNPACK and self.device.startswith('cuda'):
            raise ConfigValidationError("QNNPACK backend is not compatible with CUDA devices")
        if self.method in [QuantizationMethod.PT2E_STATIC, QuantizationMethod.PT2E_QAT] and self.backend != Backend.X86:
            raise ConfigValidationError("PT2E quantization methods are only compatible with X86 backend")

    def update(self, **kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_options[key] = value
        self.validate()

    def __str__(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False)

class QuantizationError(Exception):
    """Base class for quantization-related errors."""

class CalibrationError(QuantizationError):
    """Raised when there's an error during model calibration."""

class BackendError(QuantizationError):
    """Raised when there's an error related to backend configuration."""

class ModelStorageError(Exception):
    """Raised when there's an error related to model storage or retrieval."""

class ConfigValidationError(Exception):
    """Raised when there's an error in configuration validation."""

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
    
    @staticmethod
    def compare_activation_distributions(float_model: nn.Module, quantized_model: nn.Module, inputs: Any) -> Dict[str, Dict[str, np.ndarray]]:
        float_activations = NumericSuite.get_activation_values(float_model, inputs)
        quant_activations = NumericSuite.get_activation_values(quantized_model, inputs)
        
        results = {}
        for name in float_activations.keys():
            if name in quant_activations:
                results[name] = {
                    'float_mean': np.mean(float_activations[name]),
                    'float_std': np.std(float_activations[name]),
                    'quant_mean': np.mean(quant_activations[name]),
                    'quant_std': np.std(quant_activations[name]),
                }
        return results

    @staticmethod
    def get_activation_values(model: nn.Module, inputs: Any) -> Dict[str, np.ndarray]:
        activations = {}
        def hook_fn(name):
            def fn(_, __, output):
                activations[name] = output.detach().cpu().numpy()
            return fn

        for name, module in model.named_modules():
            module.register_forward_hook(hook_fn(name))

        with torch.no_grad():
            model(inputs)

        return activations

    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module, example_inputs: Any) -> Dict[str, Any]:
        comparison_results = {}
        
        comparison_results["architecture_diff"] = NumericSuite._compare_architectures(model1, model2)
        comparison_results["output_diff"] = NumericSuite._compare_outputs(model1, model2, example_inputs)
        comparison_results["parameter_diff"] = NumericSuite._compare_parameters(model1, model2)
        
        return comparison_results
    
    @staticmethod
    def _compare_architectures(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        def get_architecture_string(model: nn.Module) -> str:
            return str(model)
        
        arch1 = get_architecture_string(model1)
        arch2 = get_architecture_string(model2)
        
        return {
            "model1_architecture": arch1,
            "model2_architecture": arch2,
            "architectures_match": arch1 == arch2
        }
    
    @staticmethod
    def _compare_outputs(model1: nn.Module, model2: nn.Module, example_inputs: Any) -> Dict[str, Any]:
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(example_inputs)
            output2 = model2(example_inputs)
        
        if isinstance(output1, tuple):
            output1 = output1[0]
        if isinstance(output2, tuple):
            output2 = output2[0]
        
        mae = torch.mean(torch.abs(output1 - output2)).item()
        mse = torch.mean((output1 - output2) ** 2).item()
        
        return {
            "mean_absolute_error": mae,
            "mean_squared_error": mse
        }
    
    @staticmethod
    def _compare_parameters(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        
        diff_stats = {}
        for name in params1.keys():
            if name in params2:
                param1 = params1[name]
                param2 = params2[name]
                diff = torch.abs(param1 - param2)
                diff_stats[name] = {
                    "mean_diff": torch.mean(diff).item(),
                    "max_diff": torch.max(diff).item(),
                    "min_diff": torch.min(diff).item()
                }
        
        return diff_stats

    @staticmethod
    def visualize_quantization_error_distribution(float_model: nn.Module, quantized_model: nn.Module, example_inputs: Any):
        import matplotlib.pyplot as plt
        
        float_activations = NumericSuite.get_activation_values(float_model, example_inputs)
        quant_activations = NumericSuite.get_activation_values(quantized_model, example_inputs)
        
        for name in float_activations.keys():
            if name in quant_activations:
                error = float_activations[name] - quant_activations[name]
                plt.figure(figsize=(10, 5))
                plt.hist(error.flatten(), bins=50)
                plt.title(f"Quantization Error Distribution: {name}")
                plt.xlabel("Error")
                plt.ylabel("Frequency")
                plt.show()
    
    @staticmethod
    def visualize_weight_distributions(float_model: nn.Module, quantized_model: nn.Module):
        import matplotlib.pyplot as plt

        for (name, float_param), (_, quant_param) in zip(float_model.named_parameters(), quantized_model.named_parameters()):
            if 'weight' in name:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(float_param.detach().numpy().flatten(), bins=50)
                plt.title(f"Float Weights: {name}")
                plt.subplot(1, 2, 2)
                plt.hist(quant_param.dequantize().detach().numpy().flatten(), bins=50)
                plt.title(f"Quantized Weights: {name}")
                plt.tight_layout()
                plt.show()

    @staticmethod
    def visualize_activation_distributions(float_model: nn.Module, quantized_model: nn.Module, example_input: torch.Tensor):
        import matplotlib.pyplot as plt

        def hook_fn(module, input, output, activations, name):
            activations[name] = output.detach()

        float_activations = {}
        quant_activations = {}

        for name, module in float_model.named_modules():
            module.register_forward_hook(lambda mod, inp, out, act=float_activations, n=name: hook_fn(mod, inp, out, act, n))

        for name, module in quantized_model.named_modules():
            module.register_forward_hook(lambda mod, inp, out, act=quant_activations, n=name: hook_fn(mod, inp, out, act, n))

        with torch.no_grad():
            float_model(example_input)
            quantized_model(example_input)

        for name in float_activations:
            if name in quant_activations:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(float_activations[name].numpy().flatten(), bins=50)
                plt.title(f"Float Activations: {name}")
                plt.subplot(1, 2, 2)
                plt.hist(quant_activations[name].dequantize().numpy().flatten(), bins=50)
                plt.title(f"Quantized Activations: {name}")
                plt.tight_layout()
                plt.show()

    @staticmethod
    def visualize_quantization_effects_on_architecture(float_model: nn.Module, quantized_model: nn.Module):
        import networkx as nx
        import matplotlib.pyplot as plt

        def create_graph(model: nn.Module) -> nx.DiGraph:
            G = nx.DiGraph()
            for name, module in model.named_modules():
                if name:
                    G.add_node(name, label=f"{name}\n{type(module).__name__}")
                    parent = '.'.join(name.split('.')[:-1])
                    if parent:
                        G.add_edge(parent, name)
            return G

        G_float = create_graph(float_model)
        G_quant = create_graph(quantized_model)

        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        pos = nx.spring_layout(G_float)
        nx.draw(G_float, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=8, font_weight='bold')
        nx.draw_networkx_labels(G_float, pos, {node: data['label'] for node, data in G_float.nodes(data=True)})
        plt.title("Float Model Architecture")

        plt.subplot(122)
        pos = nx.spring_layout(G_quant)
        nx.draw(G_quant, pos, with_labels=True, node_size=1000, node_color='lightgreen', font_size=8, font_weight='bold')
        nx.draw_networkx_labels(G_quant, pos, {node: data['label'] for node, data in G_quant.nodes(data=True)})
        plt.title("Quantized Model Architecture")

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
        def measure_latency(model: nn.Module, inputs: Any) -> float:
            model = model.to(device)
            inputs = inputs[0].to(device) if isinstance(inputs, tuple) else inputs.to(device)
            
            if device.startswith('cuda'):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            else:
                start = time.time()
            
            with torch.no_grad():
                _ = model(inputs)
            
            if device.startswith('cuda'):
                ender.record()
                torch.cuda.synchronize()
                elapsed_time = starter.elapsed_time(ender) / 1000  # convert to seconds
            else:
                elapsed_time = time.time() - start
            
            return elapsed_time

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fp32_latencies = list(executor.map(lambda _: measure_latency(fp32_model, inputs), range(n_iter)))
            int8_latencies = list(executor.map(lambda _: measure_latency(int8_model, inputs), range(n_iter)))

        fp32_latency = sum(fp32_latencies) / n_iter
        int8_latency = sum(int8_latencies) / n_iter
        
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
            for inputs, labels in tqdm(dataloader, desc="Evaluating models"):
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
            for inputs, labels in tqdm(test_loader, desc="Evaluating binary classification"):
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    @staticmethod
    def evaluate_multi_class_classification(model: nn.Module, test_loader: DataLoader, num_classes: int) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating multi-class classification"):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        accuracy = correct / total
        class_accuracies = {f"class_{i}_accuracy": class_correct[i] / class_total[i] for i in range(num_classes)}
        
        return {"overall_accuracy": accuracy, **class_accuracies}

    @staticmethod
    def evaluate_model_robustness(model: nn.Module, test_loader: DataLoader, perturbation_types: List[str], perturbation_magnitudes: List[float]) -> Dict[str, float]:
        results = {}
        
        for p_type in perturbation_types:
            for p_mag in perturbation_magnitudes:
                perturbed_accuracy = ModelEvaluator._evaluate_with_perturbation(model, test_loader, p_type, p_mag)
                results[f"{p_type}_{p_mag}"] = perturbed_accuracy
        
        return results

    @staticmethod
    def _evaluate_with_perturbation(model: nn.Module, test_loader: DataLoader, perturbation_type: str, magnitude: float) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Evaluating with {perturbation_type} perturbation"):
                perturbed_inputs = ModelEvaluator._apply_perturbation(inputs, perturbation_type, magnitude)
                outputs = model(perturbed_inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total

    @staticmethod
    def _apply_perturbation(inputs: torch.Tensor, perturbation_type: str, magnitude: float) -> torch.Tensor:
        if perturbation_type == 'gaussian_noise':
            return inputs + torch.randn_like(inputs) * magnitude
        elif perturbation_type == 'salt_and_pepper':
            mask = torch.rand_like(inputs) < magnitude
            inputs[mask] = torch.randint(2, size=inputs[mask].shape).float()
            return inputs
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

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

    @staticmethod
    def compare_quantized_model_performance_across_hardware(
        quantized_model: nn.Module,
        example_inputs: Any,
        hardware_devices: List[str]
    ) -> Dict[str, float]:
        performance_results = {}
        
        for device in hardware_devices:
            logger.info(f"Evaluating performance on {device}")
            quantized_model = quantized_model.to(device)
            example_inputs = example_inputs.to(device)
            
            start_time = time.time()
            with torch.no_grad():
                _ = quantized_model(example_inputs)
            end_time = time.time()
            
            inference_time = end_time - start_time
            performance_results[device] = inference_time
        
        return performance_results

class BaseQuantizationMethod(ABC):
    def __init__(self, quantizer: 'Quantizer'):
        self.quantizer = quantizer

    @abstractmethod
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        pass

    def prepare_model(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model = self.quantizer.handle_non_traceable(model)
        self.quantizer.qconfig_manager.update_qconfig_mapping(model)
        return self.quantizer.model_fuser.auto_fuse_modules(model)

class StaticQuantization(BaseQuantizationMethod):
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model = self.prepare_model(model, example_inputs)
        model.eval()
        logger.info("Preparing model for static quantization")
        prepared_model = tqfx.prepare_fx(model, self.quantizer.qconfig_manager.qconfig_mapping, example_inputs)
        logger.info("Calibrating model")
        self.quantizer.calibration_manager.calibrate(prepared_model, example_inputs)
        logger.info("Converting model to static quantized version")
        return tqfx.convert_fx(prepared_model)

class DynamicQuantization(BaseQuantizationMethod):
    def __init__(self, quantizer: 'Quantizer', custom_qconfig_spec: Optional[Dict[Type[nn.Module], Any]] = None):
        super().__init__(quantizer)
        self.custom_qconfig_spec = custom_qconfig_spec

    def get_default_qconfig_spec(self) -> Dict[Type[nn.Module], Any]:
        per_channel_dynamic_layers = [
            nn.Linear, nn.LSTM, nn.GRU,
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
        ]
        
        qconfig_spec = {layer_type: per_channel_dynamic_qconfig for layer_type in per_channel_dynamic_layers}
        qconfig_spec[nn.Embedding] = float_qparams_weight_only_qconfig
        
        return qconfig_spec

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model = self.prepare_model(model, example_inputs)
        logger.info("Starting dynamic quantization")
        
        qconfig_spec = self.get_default_qconfig_spec()
        if self.custom_qconfig_spec:
            qconfig_spec.update(self.custom_qconfig_spec)
        
        qconfig_mapping = tq.QConfigMapping().set_global(None)
        for module_type, qconfig in qconfig_spec.items():
            qconfig_mapping.set_object_type(module_type, qconfig)
        
        # Allow for specifying configs for specific layers by name
        for name, module in model.named_modules():
            if name in self.custom_qconfig_spec:
                qconfig_mapping.set_module_name(name, self.custom_qconfig_spec[name])
        
        logger.info("Preparing model for dynamic quantization using FX")
        prepared_model = tqfx.prepare_fx(model, qconfig_mapping, example_inputs)
        logger.info("Converting model to dynamic quantized version using FX")
        return tqfx.convert_fx(prepared_model)

class TrainerBase(ABC):
    def __init__(self, 
                 num_epochs: int,
                 learning_rate: float,
                 optimizer_class: Type[torch.optim.Optimizer],
                 loss_fn: Callable = F.cross_entropy,
                 scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
                 scheduler_params: Optional[Dict[str, Any]] = None):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

    @abstractmethod
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> nn.Module:
        pass

    @staticmethod
    def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

class QATTrainer(TrainerBase):
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> nn.Module:
        model.train()
        optimizer = self.optimizer_class(model.parameters(), lr=self.learning_rate)
        scheduler = self.scheduler_class(optimizer, **self.scheduler_params) if self.scheduler_class else None
        
        logger.info("Starting quantization-aware training")
        best_model = None
        best_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer)
            
            if scheduler:
                scheduler.step()
            
            if val_loader:
                val_accuracy = self.evaluate(model, val_loader)
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model = copy.deepcopy(model)
            else:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")
        
        return best_model if best_model is not None else model

class QuantizationAwareTraining(BaseQuantizationMethod):
    def __init__(self, quantizer: 'Quantizer', 
                 num_epochs: int,
                 learning_rate: float,
                 optimizer_class: Type[torch.optim.Optimizer],
                 loss_fn: Callable = F.cross_entropy,
                 scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
                 scheduler_params: Optional[Dict[str, Any]] = None):
        super().__init__(quantizer)
        self.trainer = QATTrainer(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            loss_fn=loss_fn,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params
        )

    def quantize(self, model: nn.Module, example_inputs: Any, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> nn.Module:
        model = self.prepare_model(model, example_inputs)
        logger.info("Preparing model for quantization-aware training")
        prepared_model = tqfx.prepare_qat_fx(model, self.quantizer.qconfig_manager.qconfig_mapping, example_inputs)
        
        logger.info("Starting quantization-aware training")
        trained_model = self.trainer.train(prepared_model, train_loader, val_loader)
        
        trained_model.eval()
        logger.info("Converting model to quantized version after QAT")
        return tqfx.convert_fx(trained_model)

class PT2EStaticQuantization(BaseQuantizationMethod):
    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model = self.prepare_model(model, example_inputs)
        logger.info("Starting PT2E static quantization")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_pt2e(exported_model, self.quantizer.quantizer)
        self.quantizer.calibration_manager.calibrate(prepared_model, example_inputs)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

class PT2EQuantizationAwareTraining(BaseQuantizationMethod):
    def __init__(self, quantizer: 'Quantizer', train_function: Callable):
        super().__init__(quantizer)
        self.train_function = train_function

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model = self.prepare_model(model, example_inputs)
        logger.info("Starting PT2E quantization-aware training")
        exported_model = capture_pre_autograd_graph(model, example_inputs)
        prepared_model = tqpt2e.prepare_qat_pt2e(exported_model, self.quantizer.quantizer)
        self.train_function(prepared_model)
        quantized_model = tqpt2e.convert_pt2e(prepared_model)
        tq.move_exported_model_to_eval(quantized_model)
        return quantized_model

class QConfigManager:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.qconfig_mapping = self._get_default_qconfig_mapping()

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

    def update_qconfig_mapping(self, model: nn.Module) -> None:
        logger.info("Updating qconfig mapping based on model architecture")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.set_module_qconfig(nn.Conv2d, per_channel_dynamic_qconfig)
            elif isinstance(module, nn.Linear):
                self.set_module_qconfig(nn.Linear, default_dynamic_qconfig)
        logger.info("QConfig mapping updated")

class ModelFuser:
    @staticmethod
    def fuse_modules(model: nn.Module, modules_to_fuse: List[List[str]]) -> nn.Module:
        logger.info(f"Fusing modules: {modules_to_fuse}")
        return tq.fuse_modules(model, modules_to_fuse)

    @staticmethod
    def auto_fuse_modules(model: nn.Module) -> nn.Module:
        logger.info("Automatically fusing modules")
        fusion_patterns = [
            (r'.*conv.*', r'.*bn.*'),
            (r'.*conv.*', r'.*bn.*', r'.*relu.*'),
            (r'.*conv.*', r'.*relu.*'),
            (r'.*linear.*', r'.*relu.*'),
        ]
        
        def get_modules_to_fuse(model: nn.Module, pattern: List[str]) -> List[List[str]]:
            modules_to_fuse = []
            for name, _ in model.named_modules():
                if len(pattern) == 2:
                    if re.match(pattern[0], name) and re.match(pattern[1], name.split('.')[-1]):
                        modules_to_fuse.append([name, name.split('.')[-1]])
                elif len(pattern) == 3:
                    module_names = name.split('.')
                    if len(module_names) >= 3:
                        if (re.match(pattern[0], module_names[-3]) and 
                            re.match(pattern[1], module_names[-2]) and 
                            re.match(pattern[2], module_names[-1])):
                            modules_to_fuse.append(['.'.join(module_names[:-2]), 
                                                    '.'.join(module_names[:-1]), 
                                                    '.'.join(module_names)])
            return modules_to_fuse

        for pattern in fusion_patterns:
            modules_to_fuse = get_modules_to_fuse(model, pattern)
            if modules_to_fuse:
                model = ModelFuser.fuse_modules(model, modules_to_fuse)
        
        return model

class CalibrationManager:
    def calibrate(self, prepared_model: nn.Module, calibration_data: Any) -> None:
        logger.info("Calibrating model")
        prepared_model.eval()
        try:
            with torch.no_grad():
                if isinstance(calibration_data, DataLoader):
                    for batch in tqdm(calibration_data, desc="Calibrating"):
                        prepared_model(*batch)
                elif isinstance(calibration_data, torch.Tensor):
                    prepared_model(calibration_data)
                else:
                    raise ValueError("Unsupported calibration data type. Expected DataLoader or Tensor.")
        except Exception as e:
            logger.error(f"Failed to calibrate model: {str(e)}")
            raise CalibrationError("Failed to calibrate model") from e

class Quantizer:
    def __init__(self, backend_manager: BackendManager, backend: Backend):
        self.backend_manager = backend_manager
        self.backend = backend
        self.qconfig_manager = QConfigManager(backend)
        self.model_fuser = ModelFuser()
        self.calibration_manager = CalibrationManager()
        self.quantizer = self.backend_manager.get_quantizer()
        self.numeric_suite = NumericSuite()

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
            if self.backend == Backend.X86:
                return QuantizationMethod.PT2E_STATIC
            else:
                return QuantizationMethod.STATIC
        elif any(isinstance(m, nn.Linear) for m in model.modules()):
            return QuantizationMethod.QAT
        else:
            return QuantizationMethod.DYNAMIC

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

    def auto_tune_quantization_params(self, model: nn.Module, example_inputs: Any, eval_function: Callable) -> nn.Module:
        logger.info("Auto-tuning quantization parameters")
        best_model = None
        best_accuracy = float('-inf')
        
        bit_widths = [4, 8]
        qschemes = [torch.per_tensor_affine, torch.per_channel_symmetric]
        
        for bits in tqdm(bit_widths, desc="Tuning bit widths"):
            for qscheme in tqdm(qschemes, desc="Tuning qschemes", leave=False):
                logger.info(f"Trying quantization with {bits} bits and {qscheme} scheme")
                
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.MinMaxObserver.with_args(qscheme=qscheme, dtype=torch.quint8),
                    weight=torch.quantization.MinMaxObserver.with_args(qscheme=qscheme, dtype=torch.qint8)
                )
                
                self.qconfig_manager.set_global_qconfig(qconfig)
                
                quantized_model = self.quantize(model, example_inputs)
                
                accuracy = eval_function(quantized_model)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = quantized_model
        
        logger.info(f"Auto-tuning complete. Best accuracy: {best_accuracy}")
        return best_model

    def quantize(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        method = self._heuristic_based_selection(model)
        quantization_method = QuantizationMethodFactory.create(method, self)
        return quantization_method.quantize(model, example_inputs)

class QuantizationMethodFactory:
    @staticmethod
    def create(method: QuantizationMethod, quantizer: Quantizer, train_function: Optional[Callable] = None) -> BaseQuantizationMethod:
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
        self.storage_manager = ModelStorageManager(config.save_path)

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

            # Save the quantized model
            metadata = {
                'quantization_method': quantization_type.value,
                'metrics': metrics,
                'config': asdict(self.config)
            }
            version = self.storage_manager.save_model(quantized_model, metadata)
            logger.info(f"Quantized model saved as version {version}")

            return quantized_model, metrics
        except Exception as e:
            logger.error(f"Error during quantization and evaluation: {str(e)}")
            raise QuantizationError("Quantization process failed") from e

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = self.quantizer.handle_non_traceable(model)
        self.quantizer.qconfig_manager.update_qconfig_mapping(model)
        return self.quantizer.model_fuser.auto_fuse_modules(model)

    def _quantize_model(self, model: nn.Module, example_inputs: Any, quantization_type: QuantizationMethod, train_function: Optional[Callable]) -> nn.Module:
        quantization_method = QuantizationMethodFactory.create(
            quantization_type, 
            self.quantizer, 
            train_function
        )
        return quantization_method.quantize(model, example_inputs)

    def _calibrate_model(self, model: nn.Module, calibration_data: Any) -> None:
        self.quantizer.calibration_manager.calibrate(model, calibration_data)

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
        for method in tqdm(methods, desc="Comparing quantization methods"):
            try:
                quantization_method = QuantizationMethodFactory.create(method, self.quantizer, train_function)
                quantized_model = quantization_method.quantize(model, example_inputs)
                metrics = self.model_evaluator.evaluate_model(model, quantized_model, example_inputs, self.config.device)
                results[method.value] = metrics
            except Exception as e:
                logger.error(f"Error during comparison of {method.value}: {str(e)}")
                results[method.value] = {"error": str(e)}
        return results

    def auto_select_best_method(self, results: Dict[str, Dict[str, Any]]) -> QuantizationMethod:
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if not valid_results:
            raise QuantizationError("No valid quantization methods found")
        best_method = max(valid_results, key=lambda x: valid_results[x]['size_reduction'] * valid_results[x]['latency']['fp32_latency'] / valid_results[x]['latency']['int8_latency'] / (1 + valid_results[x]['accuracy']['relative_difference']))
        logger.info(f"Automatically selected best method: {best_method}")
        return QuantizationMethod(best_method)

    def track_quantization_progress(self, total_steps: int):
        self.progress = 0
        self.total_steps = total_steps

    def update_progress(self):
        self.progress += 1
        progress_percentage = (self.progress / self.total_steps) * 100
        logger.info(f"Quantization progress: {progress_percentage:.2f}%")
    
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

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs) -> nn.Module:
        if model_type == "SimpleCNN":
            return SimpleCNN(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class SimpleCNN(nn.Module):
    def __init__(self, input_shape: List[int], num_classes: int = 1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (input_shape[1] // 4) * (input_shape[2] // 4), num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)

class DataManager:
    def __init__(self, config: Config):
        self.config = config

    def get_example_inputs(self) -> torch.Tensor:
        return torch.randn(1, *self.config.input_shape)

    def get_calibration_data(self) -> DataLoader:
        dummy_data = torch.randn(100, *self.config.input_shape)
        dummy_labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(dummy_data, dummy_labels)
        return DataLoader(dataset, batch_size=32)

    def create_binary_dataset(self) -> Tuple[TensorDataset, TensorDataset]:
        total_samples = self.config.num_samples
        train_samples = int(0.8 * total_samples)  # 80% for training
        test_samples = total_samples - train_samples

        def create_data(num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
            images = torch.randint(0, 2, (num_samples, *self.config.input_shape), dtype=torch.float32)
            labels = (images.view(num_samples, -1).sum(dim=1) > (images[0].numel() // 2)).float()
            return images, labels

        train_images, train_labels = create_data(train_samples)
        test_images, test_labels = create_data(test_samples)

        return TensorDataset(train_images, train_labels), TensorDataset(test_images, test_labels)

class ModelStorageManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.version_file = os.path.join(self.base_path, 'versions.yaml')
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, Any]:
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return yaml.safe_load(f)
        return {'latest': 0, 'models': {}}

    def _save_versions(self) -> None:
        with open(self.version_file, 'w') as f:
            yaml.dump(self.versions, f)

    def save_model(self, model: nn.Module, metadata: Dict[str, Any], use_jit: bool = True) -> int:
        version = self.versions['latest'] + 1
        model_path = os.path.join(self.base_path, f'model_v{version}.pth')
        
        try:
            if use_jit:
                torch.jit.save(torch.jit.script(model), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelStorageError("Failed to save model") from e
        
        self.versions['latest'] = version
        self.versions['models'][version] = {
            'path': model_path,
            'metadata': metadata,
            'timestamp': time.time()
        }
        self._save_versions()
        return version

    def load_model(self, version: Optional[int] = None, model_class: Type[nn.Module] = SimpleCNN, use_jit: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
        if version is None:
            version = self.versions['latest']
        
        if version not in self.versions['models']:
            raise ModelStorageError(f"Version {version} not found")
        
        model_info = self.versions['models'][version]
        
        try:
            if use_jit:
                model = torch.jit.load(model_info['path'])
            else:
                model = model_class()
                model.load_state_dict(torch.load(model_info['path']))
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelStorageError("Failed to load model") from e
        
        return model, model_info['metadata']

    def get_model_history(self) -> List[Dict[str, Any]]:
        return [
            {'version': v, **info}
            for v, info in self.versions['models'].items()
        ]

def main():
    # Load configuration
    config = Config.from_yaml('config.yaml')
    config.validate()

    # Initialize QuantizationWrapper and DataManager
    qw = QuantizationWrapper(config)
    data_manager = DataManager(config)

    # Create model
    model = ModelFactory.create_model(config.model_type, input_shape=config.input_shape, num_classes=1)

    # Create datasets
    train_dataset, test_dataset = data_manager.create_binary_dataset()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    # Get example inputs
    example_inputs = data_manager.get_example_inputs()

    # Define a simple training function for QAT
    def train_function(model: nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.BCELoss()
        model.train()
        for epoch in range(config.num_epochs):
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
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
    qw.track_quantization_progress(total_steps=5)  # Assuming 5 steps in the quantization process
    quantized_model, metrics = qw.quantize_and_evaluate(
        model,
        example_inputs,
        quantization_type=best_method,
        calibration_data=data_manager.get_calibration_data(),
        eval_function=lambda m: qw.model_evaluator.evaluate_binary_classification(m, test_loader),
        train_function=train_function
    )

    logger.info("Quantization complete. Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    # Initialize ModelStorageManager
    storage_manager = ModelStorageManager(config.save_path)

    # Save the quantized model with metadata
    metadata = {
        'quantization_method': best_method.value,
        'metrics': metrics,
        'config': asdict(config)
    }
    version = storage_manager.save_model(quantized_model, metadata)
    logger.info(f"Quantized model saved as version {version}")

    # Load the latest quantized model
    loaded_model, loaded_metadata = storage_manager.load_model()
    logger.info(f"Loaded quantized model: {loaded_metadata}")

    # Compare the loaded model with the original quantized model
    loaded_accuracy = qw.model_evaluator.evaluate_binary_classification(loaded_model, test_loader)
    original_accuracy = qw.model_evaluator.evaluate_binary_classification(quantized_model, test_loader)
    logger.info(f"Original quantized model accuracy: {original_accuracy:.4f}")
    logger.info(f"Loaded quantized model accuracy: {loaded_accuracy:.4f}")

    # Visualize quantization effects
    NumericSuite.visualize_quantization_effects(model, quantized_model)
    NumericSuite.visualize_quantization_effects_on_architecture(model, quantized_model)

    # Profile quantization
    profiling_results = ModelEvaluator.profile_quantization(
        model, 
        example_inputs, 
        QuantizationMethodFactory.create(best_method, qw.quantizer, train_function)
    )
    logger.info("Quantization profiling results:")
    for key, value in profiling_results.items():
        logger.info(f"{key}: {value}")

    # Evaluate model robustness
    robustness_results = ModelEvaluator.evaluate_model_robustness(
        quantized_model,
        test_loader,
        perturbation_types=['gaussian_noise', 'salt_and_pepper'],
        perturbation_magnitudes=[0.1, 0.2, 0.3]
    )
    logger.info("Model robustness results:")
    for key, value in robustness_results.items():
        logger.info(f"{key}: {value}")

    # Compare quantized model performance across different hardware
    hardware_devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    performance_results = ModelEvaluator.compare_quantized_model_performance_across_hardware(
        quantized_model,
        example_inputs,
        hardware_devices
    )
    logger.info("Quantized model performance across hardware:")
    for device, inference_time in performance_results.items():
        logger.info(f"{device}: {inference_time:.6f} seconds")

    # Get model version history
    model_history = storage_manager.get_model_history()
    logger.info("Model version history:")
    for entry in model_history:
        logger.info(f"Version {entry['version']}: {entry['metadata']}")

    logger.info("Quantization process completed successfully.")

if __name__ == "__main__":
    main()