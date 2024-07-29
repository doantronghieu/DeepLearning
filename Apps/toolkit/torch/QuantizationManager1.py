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

    def prepare_model(self, model: torch.nn.Module, example_inputs: Optional[torch.Tensor] = None, 
                      is_qat: bool = False, is_dynamic: bool = False, 
                      quantizable_ops: Optional[List[torch.nn.Module]] = None) -> torch.nn.Module:
        """
        Prepare the model for quantization with enhanced flexibility.
        
        Args:
            model (torch.nn.Module): The model to prepare for quantization.
            example_inputs (Optional[torch.Tensor]): Example inputs for the model (needed for FX mode).
            is_qat (bool): Whether to prepare for quantization-aware training.
            is_dynamic (bool): Whether to use dynamic quantization.
            quantizable_ops (Optional[List[torch.nn.Module]]): List of module types to quantize.
        
        Returns:
            torch.nn.Module: The prepared model.
        """
        model = copy.deepcopy(model)
        
        if is_dynamic:
            return self._prepare_dynamic(model, quantizable_ops)
        
        if self.use_fx:
            if is_qat:
                prepared_model = quantize_fx.prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
            else:
                prepared_model = quantize_fx.prepare_fx(model, self.qconfig_mapping, example_inputs)
        else:
            model.eval() if not is_qat else model.train()
            model.qconfig = self.qat_qconfig if is_qat else self.qconfig
            
            if quantizable_ops:
                propagate_qconfig_(model, qconfig_dict={op: self.qconfig for op in quantizable_ops})
            else:
                propagate_qconfig_(model)
            
            # Fuse modules if applicable
            model = fuse_modules(model, self._get_fusable_modules(model))
            
            prepared_model = prepare_qat(model) if is_qat else prepare(model)
        
        return prepared_model

    def _prepare_dynamic(self, model: torch.nn.Module, quantizable_ops: Optional[List[torch.nn.Module]] = None) -> torch.nn.Module:
        """
        Prepare the model for dynamic quantization with enhanced techniques.
        
        Args:
            model (torch.nn.Module): The model to prepare for dynamic quantization.
            quantizable_ops (Optional[List[torch.nn.Module]]): List of module types to quantize.
        
        Returns:
            torch.nn.Module: The prepared model.
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

    def quantize_model(self, prepared_model: torch.nn.Module, is_dynamic: bool = False) -> torch.nn.Module:
        """
        Convert the prepared model to a quantized model.
        
        Args:
            prepared_model (torch.nn.Module): The prepared model to quantize.
            is_dynamic (bool): Whether to use dynamic quantization.
        
        Returns:
            torch.nn.Module: The quantized model.
        """
        if is_dynamic:
            return convert(prepared_model)
        
        if self.use_fx:
            quantized_model = quantize_fx.convert_fx(prepared_model)
        else:
            quantized_model = convert(prepared_model)
        return quantized_model

    def _get_fusable_modules(self, model: torch.nn.Module) -> List[List[str]]:
        """
        Get a list of fusable modules in the model.
        
        Args:
            model (torch.nn.Module): The model to analyze.
        
        Returns:
            List[List[str]]: List of fusable module sequences.
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
        
        Args:
            float_model (torch.nn.Module): The original floating-point model.
            quant_model (torch.nn.Module): The quantized model.
        
        Returns:
            Dict[str, Any]: Analysis results including weight ranges, activation ranges, and quantization parameters.
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
        
        Args:
            model (torch.nn.Module): The model to analyze.
            example_inputs (torch.Tensor): Example inputs for the model.
        
        Returns:
            QConfigMapping: The selected quantization configuration mapping.
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
        
        Args:
            model (torch.nn.Module): The model to benchmark.
            input_data (torch.Tensor): Input data for the model.
            num_runs (int): Number of inference runs to average over.
        
        Returns:
            Dict[str, float]: Dictionary containing various performance metrics.
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
        
        Args:
            prepared_model (torch.nn.Module): The prepared model to calibrate.
            calibration_data (torch.Tensor): The data to use for calibration.
        """
        prepared_model.eval()
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)

    def save_quantized_model(self, model: torch.nn.Module, path: str) -> None:
        """
        Save the quantized model.
        
        Args:
            model (torch.nn.Module): The quantized model to save.
            path (str): The path to save the model to.
        """
        torch.save(model.state_dict(), path)

    def load_quantized_model(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        """
        Load a quantized model.
        
        Args:
            model (torch.nn.Module): An instance of the model architecture.
            path (str): The path to load the model from.
        
        Returns:
            torch.nn.Module: The loaded quantized model.
        """
        model.load_state_dict(torch.load(path))
        return model

    def save_scripted_quantized_model(self, model: torch.nn.Module, path: str) -> None:
        """
        Save the quantized model as a TorchScript model.
        
        Args:
            model (torch.nn.Module): The quantized model to save.
            path (str): The path to save the model to.
        """
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path)

    def load_scripted_quantized_model(self, path: str) -> torch.jit.ScriptModule:
        """
        Load a TorchScript quantized model.
        
        Args:
            path (str): The path to load the model from.
        
        Returns:
            torch.jit.ScriptModule: The loaded quantized model.
        """
        return torch.jit.load(path)

    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """
        Get the size of the model in MB.
        
        Args:
            model (torch.nn.Module): The model to get the size of.
        
        Returns:
            float: The size of the model in MB.
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size = buffer.getbuffer().nbytes / 1e6  # Size in MB
        return size

    def compare_accuracy(self, float_model: torch.nn.Module, quant_model: torch.nn.Module, 
                         test_data: torch.Tensor, target_data: torch.Tensor,
                         metric_fn: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple[float, float]:
        float_model.eval()
        quant_model.eval()
        
        with torch.no_grad():
            float_output = float_model(test_data)
            quant_output = quant_model(test_data)
        
        float_accuracy = metric_fn(float_output, target_data)
        quant_accuracy = metric_fn(quant_output, target_data)
        
        return float_accuracy, quant_accuracy

    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        # Implement accuracy calculation based on your specific task
        # This is a placeholder implementation
        return ((output - target).abs() < 1e-5).float().mean().item()
    
    def quantization_aware_training(self, model: torch.nn.Module, train_loader: DataLoader, 
                                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, 
                                    num_epochs: int) -> torch.nn.Module:
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
        self.qconfig = qconfig
        self.qconfig_mapping = QConfigMapping().set_global(self.qconfig)

    def apply_mixed_precision_quantization(self, model: torch.nn.Module, 
                                           example_inputs: torch.Tensor) -> torch.nn.Module:
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