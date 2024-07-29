# My code starts from here
import torch
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
import copy
import io
import time
from typing import Dict, Any, Optional, Tuple, List

class QuantizationManager:
    def __init__(self, backend: str = 'x86', use_fx: bool = False):
        self.backend = backend
        self.use_fx = use_fx
        self.qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
        self.qat_qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)
        torch.backends.quantized.engine = self.backend

        if self.backend == 'onednn':
            self.qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('onednn')
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
                torch.ao.quantization.propagate_qconfig_(model, qconfig_dict={op: self.qconfig for op in quantizable_ops})
            else:
                torch.ao.quantization.propagate_qconfig_(model)
            
            # Fuse modules if applicable
            model = torch.ao.quantization.fuse_modules(model, self._get_fusable_modules(model))
            
            prepared_model = torch.ao.quantization.prepare_qat(model) if is_qat else torch.ao.quantization.prepare(model)
        
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
            qconfig_dict = {op: torch.ao.quantization.default_dynamic_qconfig for op in quantizable_ops}
        else:
            qconfig_dict = {
                torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.LSTM: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.GRU: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.RNN: torch.ao.quantization.default_dynamic_qconfig,
            }
        
        model.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            weight=torch.ao.quantization.per_channel_weight_observer.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        )
        
        return torch.ao.quantization.prepare(model, qconfig_dict=qconfig_dict)

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
            return torch.ao.quantization.convert(prepared_model)
        
        if self.use_fx:
            quantized_model = quantize_fx.convert_fx(prepared_model)
        else:
            quantized_model = torch.ao.quantization.convert(prepared_model)
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
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module_sequence = [name]
                if hasattr(module, 'bias') and module.bias is not None:
                    module_sequence.append(name + '.bias')
                next_module = list(module.children())[0] if list(module.children()) else None
                if isinstance(next_module, torch.nn.BatchNorm2d):
                    module_sequence.append(name.rsplit('.', 1)[0] + '.' + list(module.named_children())[0][0])
                if isinstance(next_module, (torch.nn.ReLU, torch.nn.ReLU6)):
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
        # Implement logic to analyze model and select appropriate qconfig
        # This is a placeholder implementation
        return self.qconfig_mapping

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
            prepared_model(calibration_data)

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
                         test_data: torch.Tensor, target_data: torch.Tensor) -> Tuple[float, float]:
        """
        Compare the accuracy of float and quantized models.
        
        Args:
            float_model (torch.nn.Module): The floating-point model.
            quant_model (torch.nn.Module): The quantized model.
            test_data (torch.Tensor): Test input data for accuracy comparison.
            target_data (torch.Tensor): Target data for accuracy comparison.
        
        Returns:
            Tuple[float, float]: Accuracy of float model and quantized model.
        """
        float_model.eval()
        quant_model.eval()
        
        with torch.no_grad():
            float_output = float_model(test_data)
            quant_output = quant_model(test_data)
        
        float_accuracy = self._calculate_accuracy(float_output, target_data)
        quant_accuracy = self._calculate_accuracy(quant_output, target_data)
        
        return float_accuracy, quant_accuracy

    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        # Implement accuracy calculation based on your specific task
        # This is a placeholder implementation
        return ((output - target).abs() < 1e-5).float().mean().item()