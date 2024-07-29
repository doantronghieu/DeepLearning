# My code starts from here
import torch
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
import copy
import io
import os
from typing import Dict, Any, Optional, Tuple

class QuantizationManager:
    def __init__(self, backend: str = 'x86', use_fx: bool = False):
        """
        Initialize the QuantizationManager.
        
        Args:
            backend (str): The backend to use for quantization. Options are 'x86', 'qnnpack', or 'onednn'.
            use_fx (bool): Whether to use FX Graph Mode Quantization (True) or Eager Mode Quantization (False).
        """
        self.backend = backend
        self.use_fx = use_fx
        self.qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
        self.qat_qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)
        torch.backends.quantized.engine = self.backend

        if self.backend == 'onednn':
            self.qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('onednn')
        else:
            self.qconfig_mapping = QConfigMapping().set_global(self.qconfig)

    def prepare_model(self, model: torch.nn.Module, example_inputs: torch.Tensor, 
                      is_qat: bool = False) -> torch.nn.Module:
        """
        Prepare the model for quantization.
        
        Args:
            model (torch.nn.Module): The model to prepare for quantization.
            example_inputs (torch.Tensor): Example inputs for the model (needed for FX mode).
            is_qat (bool): Whether to prepare for quantization-aware training.
        
        Returns:
            torch.nn.Module: The prepared model.
        """
        model = copy.deepcopy(model)
        
        if self.use_fx:
            if is_qat:
                prepared_model = quantize_fx.prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
            else:
                prepared_model = quantize_fx.prepare_fx(model, self.qconfig_mapping, example_inputs)
        else:
            model.eval()  # Set to eval mode for static quantization
            if is_qat:
                model.train()  # Set to train mode for QAT
                model.qconfig = self.qat_qconfig
            else:
                model.qconfig = self.qconfig
            
            # Fuse modules
            model = torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
            
            # Prepare the model
            if is_qat:
                prepared_model = torch.ao.quantization.prepare_qat(model)
            else:
                prepared_model = torch.ao.quantization.prepare(model)
        
        return prepared_model

    def quantize_model(self, prepared_model: torch.nn.Module) -> torch.nn.Module:
        """
        Convert the prepared model to a quantized model.
        
        Args:
            prepared_model (torch.nn.Module): The prepared model to quantize.
        
        Returns:
            torch.nn.Module: The quantized model.
        """
        if self.use_fx:
            quantized_model = quantize_fx.convert_fx(prepared_model)
        else:
            quantized_model = torch.ao.quantization.convert(prepared_model)
        return quantized_model

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

    def benchmark_model(self, model: torch.nn.Module, input_data: torch.Tensor, 
                        num_runs: int = 100) -> Tuple[float, float]:
        """
        Benchmark the model's inference time and throughput.
        
        Args:
            model (torch.nn.Module): The model to benchmark.
            input_data (torch.Tensor): Input data for the model.
            num_runs (int): Number of inference runs to average over.
        
        Returns:
            Tuple[float, float]: Average inference time (ms) and throughput (inferences/second).
        """
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_data)
            
            # Benchmark
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(num_runs):
                _ = model(input_data)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
        avg_time = elapsed_time / num_runs
        throughput = 1000 / avg_time  # inferences per second
        
        return avg_time, throughput