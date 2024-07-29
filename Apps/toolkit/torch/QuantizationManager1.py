# My code starts from here
import copy
import io
import time
from typing import Dict, Any, Optional, Tuple, List, Callable
import torch
from torch.utils.data import DataLoader
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import (
    QConfigMapping, float16_dynamic_qconfig, default_per_channel_qconfig, 
    default_qconfig, QConfig, get_default_qconfig, get_default_qat_qconfig, 
    get_default_qconfig_mapping, propagate_qconfig_, default_dynamic_qconfig, 
    prepare_qat, prepare, convert, fuse_modules, 
    MinMaxObserver, default_observer, default_weight_observer
)
from torch.ao.quantization.observer import default_per_channel_weight_observer

class QuantizationManager:
    def __init__(self, backend: str = 'x86', use_fx: bool = False):
        self.backend = backend
        self.use_fx = use_fx
        self.qconfig = get_default_qconfig(self.backend)
        self.qat_qconfig = get_default_qat_qconfig(self.backend)
        torch.backends.quantized.engine = self.backend

        if self.backend == 'onednn':
            self.qconfig_mapping = get_default_qconfig_mapping('onednn')
        else:
            self.qconfig_mapping = QConfigMapping().set_global(self.qconfig)

        # New attributes for FX-specific configurations
        self.skip_symbolic_trace_modules = []
        self.prepare_custom_config_dict = {}
        self.convert_custom_config_dict = {}

    def prepare_model(self, model: torch.nn.Module, example_inputs: Optional[torch.Tensor] = None, 
                      is_qat: bool = False, is_dynamic: bool = False, 
                      quantizable_ops: Optional[List[torch.nn.Module]] = None) -> torch.nn.Module:
        """
        Prepare the model for quantization with enhanced flexibility.
        """
        model = copy.deepcopy(model)
        
        if is_dynamic:
            return self._prepare_dynamic(model, quantizable_ops)
        
        if self.use_fx:
            return self._prepare_fx(model, example_inputs, is_qat)
        else:
            return self._prepare_eager(model, is_qat, quantizable_ops)

    def _prepare_fx(self, model: torch.nn.Module, example_inputs: torch.Tensor, is_qat: bool) -> torch.nn.Module:
        """
        Prepare the model using FX Graph Mode Quantization.
        """
        if not self._is_traceable(model):
            raise ValueError("Model is not symbolically traceable. Please check the model architecture.")

        self.prepare_custom_config_dict["non_traceable_module_name"] = self.skip_symbolic_trace_modules

        if is_qat:
            prepared_model = quantize_fx.prepare_qat_fx(model, self.qconfig_mapping, example_inputs,
                                                        prepare_custom_config_dict=self.prepare_custom_config_dict)
        else:
            prepared_model = quantize_fx.prepare_fx(model, self.qconfig_mapping, example_inputs,
                                                    prepare_custom_config_dict=self.prepare_custom_config_dict)
        return prepared_model

    def _prepare_eager(self, model: torch.nn.Module, is_qat: bool, quantizable_ops: Optional[List[torch.nn.Module]]) -> torch.nn.Module:
        """
        Prepare the model using eager mode quantization.
        """
        model.eval() if not is_qat else model.train()
        model.qconfig = self.qat_qconfig if is_qat else self.qconfig
        
        if quantizable_ops:
            propagate_qconfig_(model, qconfig_dict={op: self.qconfig for op in quantizable_ops})
        else:
            propagate_qconfig_(model)
        
        model = fuse_modules(model, self._get_fusable_modules(model))
        
        return prepare_qat(model) if is_qat else prepare(model)

    def _prepare_dynamic(self, model: torch.nn.Module, quantizable_ops: Optional[List[torch.nn.Module]] = None) -> torch.nn.Module:
        """
        Prepare the model for dynamic quantization with enhanced techniques.
        """
        if quantizable_ops:
            qconfig_dict = {op: default_dynamic_qconfig for op in quantizable_ops}
        else:
            qconfig_dict = {
                torch.nn.Linear: default_dynamic_qconfig,
                torch.nn.LSTM: default_dynamic_qconfig,
                torch.nn.GRU: default_dynamic_qconfig,
                torch.nn.RNN: default_dynamic_qconfig,
            }
        
        model.qconfig = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.qint8),
            weight=default_per_channel_weight_observer.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        )
        
        return prepare(model, qconfig_dict=qconfig_dict)

    def _is_traceable(self, model: torch.nn.Module) -> bool:
        """
        Check if the model is symbolically traceable.
        """
        try:
            torch.jit.script(model)
            return True
        except Exception:
            return False

    def quantize_model(self, prepared_model: torch.nn.Module, is_dynamic: bool = False) -> torch.nn.Module:
        """
        Convert the prepared model to a quantized model.
        """
        if is_dynamic:
            return convert(prepared_model)
        
        if self.use_fx:
            quantized_model = quantize_fx.convert_fx(prepared_model, convert_custom_config_dict=self.convert_custom_config_dict)
        else:
            quantized_model = convert(prepared_model)
        return quantized_model

    def handle_non_traceable_module(self, module: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        """
        Handle non-traceable modules by applying custom quantization techniques.
        """
        # Implementation depends on the specific non-traceable module
        # This is a placeholder for custom handling logic
        return module

    def _get_fusable_modules(self, model: torch.nn.Module) -> List[List[str]]:
        """
        Get a list of fusable modules in the model.
        """
        fusable_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                module_sequence = [name]
                if hasattr(module, 'bias') and module.bias is not None:
                    module_sequence.append(name + '.bias')
                next_module = list(module.children())[0] if list(module.children()) else None
                if isinstance(next_module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module_sequence.append(name.rsplit('.', 1)[0] + '.' + list(module.named_children())[0][0])
                if isinstance(next_module, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU)):
                    module_sequence.append(name.rsplit('.', 1)[0] + '.' + list(module.named_children())[0][0])
                if len(module_sequence) > 1:
                    fusable_modules.append(module_sequence)
        return fusable_modules

    def analyze_quantization(self, float_model: torch.nn.Module, quant_model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze the quantization results.
        """
        analysis = {}
        for (name, float_module), (_, quant_module) in zip(float_model.named_modules(), quant_model.named_modules()):
            if isinstance(quant_module, torch.ao.quantization.QuantizedModule):
                analysis[name] = {
                    'weight_range': (float_module.weight.min().item(), float_module.weight.max().item()),
                    'activation_range': (quant_module.activation_post_process.min_val.item(), 
                                         quant_module.activation_post_process.max_val.item()),
                    'weight_scale': quant_module.weight_scale,
                    'weight_zero_point': quant_module.weight_zero_point,
                }
            elif hasattr(quant_module, 'weight_fake_quant'):
                analysis[name] = {
                    'weight_range': (float_module.weight.min().item(), float_module.weight.max().item()),
                    'weight_scale': quant_module.weight_fake_quant.scale,
                    'weight_zero_point': quant_module.weight_fake_quant.zero_point,
                }
        return analysis

    def auto_select_qconfig(self, model: torch.nn.Module, example_inputs: torch.Tensor) -> QConfigMapping:
        """
        Automatically select the best quantization configuration based on model analysis.
        """
        qconfig_mapping = QConfigMapping()
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if module.in_features < 256:
                    qconfig_mapping.set_module_name(name, default_per_channel_qconfig)
                else:
                    qconfig_mapping.set_module_name(name, default_qconfig)
        return qconfig_mapping

    def benchmark_model(self, model: torch.nn.Module, input_data: torch.Tensor, 
                        num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark the model's performance with detailed metrics.
        """
        model.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_data)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(input_data)
            end_time = time.time()
            
        avg_time = (end_time - start_time) / num_runs
        throughput = 1 / avg_time  # inferences per second
        model_size = self.get_model_size(model)
        
        return {
            'avg_inference_time': avg_time * 1000,  # ms
            'throughput': throughput,
            'model_size_mb': model_size,
            'inferences_per_mb': throughput / model_size,
        }

    def calibrate_model(self, prepared_model: torch.nn.Module, 
                        calibration_data: torch.Tensor) -> None:
        """
        Calibrate the prepared model using the provided calibration data.
        """
        prepared_model.eval()
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)

    def save_quantized_model(self, model: torch.nn.Module, path: str) -> None:
        """
        Save the quantized model.
        """
        torch.save(model.state_dict(), path)

    def load_quantized_model(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        """
        Load a quantized model.
        """
        model.load_state_dict(torch.load(path))
        return model

    def save_scripted_quantized_model(self, model: torch.nn.Module, path: str) -> None:
        """
        Save the quantized model as a TorchScript model.
        """
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path)

    def load_scripted_quantized_model(self, path: str) -> torch.jit.ScriptModule:
        """
        Load a TorchScript quantized model.
        """
        return torch.jit.load(path)

    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """
        Get the size of the model in MB.
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size = buffer.getbuffer().nbytes / 1e6  # Size in MB
        return size

    def compare_accuracy(self, float_model: torch.nn.Module, quant_model: torch.nn.Module, 
                         test_data: torch.Tensor, target_data: torch.Tensor,
                         metric_fn: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple[float, float]:
        """
        Compare the accuracy of float and quantized models.
        """
        float_model.eval()
        quant_model.eval()
        
        with torch.no_grad():
            float_output = float_model(test_data)
            quant_output = quant_model(test_data)
        
        float_accuracy = metric_fn(float_output, target_data)
        quant_accuracy = metric_fn(quant_output, target_data)
        
        return float_accuracy, quant_accuracy

    def quantization_aware_training(self, model: torch.nn.Module, train_loader: DataLoader, 
                                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, 
                                    num_epochs: int) -> torch.nn.Module:
        """
        Perform Quantization-Aware Training (QAT) on the model.
        """
        qat_model = self.prepare_model(model, is_qat=True)
        qat_model.train()

        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = qat_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self.quantize_model(qat_model)

    def set_custom_qconfig(self, qconfig: QConfig) -> None:
        """
        Set a custom quantization configuration.
        """
        self.qconfig = qconfig
        self.qconfig_mapping = QConfigMapping().set_global(self.qconfig)

    def apply_mixed_precision_quantization(self, model: torch.nn.Module, 
                                           example_inputs: torch.Tensor) -> torch.nn.Module:
        """
        Apply mixed precision quantization to the model.
        """
        qconfig_mapping = QConfigMapping()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.out_features > 1000:
                    qconfig_mapping.set_module_name(name, float16_dynamic_qconfig)
                else:
                    qconfig_mapping.set_module_name(name, default_qconfig)
            elif isinstance(module, torch.nn.Conv2d):
                qconfig_mapping.set_module_name(name, default_qconfig)

        prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        quantized_model = quantize_fx.convert_fx(prepared_model)
        return quantized_model

    def quantize_custom_module(self, module: torch.nn.Module, 
                               quantization_config: Dict[str, Any]) -> torch.nn.Module:
        """
        Quantize a custom module using the provided configuration.
        """
        class QuantizedCustomModule(torch.nn.Module):
            def __init__(self, orig_module, qconfig):
                super().__init__()
                self.orig_module = orig_module
                self.qconfig = qconfig
                self.weight_fake_quant = qconfig.weight()
                self.activation_post_process = qconfig.activation()

            def forward(self, x):
                weight_quant = self.weight_fake_quant(self.orig_module.weight)
                out = self.orig_module._conv_forward(x, weight_quant, self.orig_module.bias)
                return self.activation_post_process(out)

        qconfig = QConfig(
            activation=quantization_config.get('activation', default_observer),
            weight=quantization_config.get('weight', default_weight_observer)
        )
        return QuantizedCustomModule(module, qconfig)

    def set_skip_symbolic_trace_modules(self, module_list: List[str]):
        """
        Set the list of modules to skip during symbolic tracing.
        """
        self.skip_symbolic_trace_modules = module_list

    def set_prepare_custom_config(self, config: Dict[str, Any]):
        """
        Set custom configuration for the prepare step in FX Graph Mode Quantization.
        """
        self.prepare_custom_config_dict = config

    def set_convert_custom_config(self, config: Dict[str, Any]):
        """
        Set custom configuration for the convert step in FX Graph Mode Quantization.
        """
        self.convert_custom_config_dict = config

    def get_qconfig_mapping(self) -> QConfigMapping:
        """
        Get the current QConfigMapping.
        """
        return self.qconfig_mapping

    def set_qconfig_mapping(self, qconfig_mapping: QConfigMapping):
        """
        Set a custom QConfigMapping.
        """
        self.qconfig_mapping = qconfig_mapping

    def fuse_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Fuse modules in the model for improved performance.
        """
        model.eval()
        model = torch.quantization.fuse_modules(model, self._get_fusable_modules(model))
        return model

    def _get_observed_module(self, module: torch.nn.Module, qconfig: QConfig) -> torch.nn.Module:
        """
        Get the observed version of a module for a given QConfig.
        """
        if isinstance(module, torch.nn.Conv2d):
            return torch.ao.quantization.QuantizedConv2d.from_float(module)
        elif isinstance(module, torch.nn.Linear):
            return torch.ao.quantization.QuantizedLinear.from_float(module)
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")

    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize the quantized model for inference.
        """
        model.eval()
        if self.use_fx:
            model = torch.ao.quantization.quantize_fx.convert_fx(model)
        else:
            model = torch.quantization.convert(model)
        return torch.jit.script(model)

    def quantize_per_channel(self, model: torch.nn.Module, example_inputs: torch.Tensor) -> torch.nn.Module:
        """
        Apply per-channel quantization to the model.
        """
        qconfig_mapping = QConfigMapping().set_global(default_per_channel_qconfig)
        prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        quantized_model = quantize_fx.convert_fx(prepared_model)
        return quantized_model

    def quantize_dynamic(self, model: torch.nn.Module, example_inputs: torch.Tensor) -> torch.nn.Module:
        """
        Apply dynamic quantization to the model.
        """
        qconfig_mapping = QConfigMapping().set_global(default_dynamic_qconfig)
        if self.use_fx:
            prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
            quantized_model = quantize_fx.convert_fx(prepared_model)
        else:
            quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec=qconfig_mapping)
        return quantized_model

    def export_torchscript(self, model: torch.nn.Module, example_inputs: torch.Tensor, path: str):
        """
        Export the quantized model to TorchScript format.
        """
        model.eval()
        traced_model = torch.jit.trace(model, example_inputs)
        torch.jit.save(traced_model, path)

    def export_onnx(self, model: torch.nn.Module, example_inputs: torch.Tensor, path: str):
        """
        Export the quantized model to ONNX format.
        """
        model.eval()
        torch.onnx.export(model, example_inputs, path, opset_version=11)

    def set_qat_learning_rate(self, optimizer: torch.optim.Optimizer, lr: float):
        """
        Set the learning rate for Quantization-Aware Training.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def quantize_embedding(self, embedding: torch.nn.Embedding, num_bits: int = 8) -> torch.nn.Embedding:
        """
        Quantize an embedding layer.
        """
        embedding.weight.data = torch.quantize_per_tensor(embedding.weight.data, 1 / 2**(num_bits-1), 0, torch.qint8)
        return embedding

    def apply_cross_layer_equalization(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply Cross-Layer Equalization (CLE) to improve quantization accuracy.
        """
        # This is a placeholder implementation. CLE requires a more complex implementation
        # that analyzes and adjusts weights across multiple layers.
        return model

    def apply_bias_correction(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply bias correction to compensate for quantization errors.
        """
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and module.bias is not None:
                # This is a simplified bias correction. A more accurate implementation
                # would involve analyzing the quantization error and adjusting accordingly.
                module.bias.data += 0.5 * module.weight.data.mean(dim=0)
        return model

    def visualize_quantization_effects(self, float_model: torch.nn.Module, quant_model: torch.nn.Module, 
                                       example_inputs: torch.Tensor):
        """
        Visualize the effects of quantization on model outputs.
        """
        float_model.eval()
        quant_model.eval()
        
        with torch.no_grad():
            float_output = float_model(example_inputs)
            quant_output = quant_model(example_inputs)
        
        diff = (float_output - quant_output).abs()
        
        print(f"Max absolute difference: {diff.max().item()}")
        print(f"Mean absolute difference: {diff.mean().item()}")
        
        # Here you could add code to create histograms or other visualizations
        # of the differences between float and quantized outputs

    def get_memory_footprint(self, model: torch.nn.Module) -> float:
        """
        Get the memory footprint of the model in MB.
        """
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem_total = mem_params + mem_bufs
        return mem_total / (1024 * 1024)  # Convert to MB

    def compare_models(self, model1: torch.nn.Module, model2: torch.nn.Module) -> Dict[str, Any]:
        """
        Compare two models (e.g., float vs quantized) and return various metrics.
        """
        comparison = {
            'param_count1': sum(p.numel() for p in model1.parameters()),
            'param_count2': sum(p.numel() for p in model2.parameters()),
            'memory_footprint1': self.get_memory_footprint(model1),
            'memory_footprint2': self.get_memory_footprint(model2),
        }
        
        comparison['param_count_diff'] = comparison['param_count1'] - comparison['param_count2']
        comparison['memory_footprint_diff'] = comparison['memory_footprint1'] - comparison['memory_footprint2']
        comparison['memory_reduction_percent'] = (1 - comparison['memory_footprint2'] / comparison['memory_footprint1']) * 100
        
        return comparison

# End of QuantizationManager class