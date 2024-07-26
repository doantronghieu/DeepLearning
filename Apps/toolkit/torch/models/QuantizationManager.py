import torch
import torch.nn as nn
import torch.quantization
import os
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping

class QuantizationManager:
    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the QuantizationManager with a PyTorch model.
        
        Args:
            model (nn.Module): The PyTorch model to be quantized.
        """
        self.float_model = model
        self.quantized_model = None
        self.logger = self._setup_logger()
        self.prepare_custom_config = {}
        self.convert_custom_config = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the QuantizationManager."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_example_inputs(self) -> Tuple:
        """Get example inputs for the model. Implement this method based on your model's input requirements."""
        raise NotImplementedError("Example inputs method needs to be implemented")

    def quantize(self, qconfig_mapping: Optional[QConfigMapping] = None, 
                 dtype: torch.dtype = torch.qint8, 
                 static: bool = False,
                 use_fx: bool = True) -> nn.Module:
        """
        Apply quantization to the model.
        
        Args:
            qconfig_mapping (Optional[QConfigMapping]): Quantization configuration mapping.
            dtype (torch.dtype): Data type for quantization (default: torch.qint8).
            static (bool): Whether to use static quantization (default: False).
            use_fx (bool): Whether to use FX Graph Mode Quantization (default: True).
        
        Returns:
            nn.Module: The quantized model.
        """
        try:
            if use_fx:
                self.logger.info("Applying FX Graph Mode Quantization...")
                if qconfig_mapping is None:
                    qconfig_mapping = get_default_qconfig_mapping("x86")
                if static:
                    self.quantized_model = self._static_quantize_fx(qconfig_mapping, dtype)
                else:
                    self.quantized_model = self._dynamic_quantize_fx(qconfig_mapping, dtype)
            else:
                if static:
                    self.logger.info("Applying static quantization...")
                    self.quantized_model = self._static_quantize(qconfig_mapping, dtype)
                else:
                    self.logger.info("Applying dynamic quantization...")
                    self.quantized_model = torch.quantization.quantize_dynamic(
                        self.float_model, qconfig_mapping, dtype=dtype
                    )
            return self.quantized_model
        except Exception as e:
            self.logger.error(f"Quantization failed: {str(e)}")
            raise
    
    def _static_quantize(self, qconfig_dict: Dict[str, torch.quantization.QConfig], 
                     dtype: torch.dtype) -> nn.Module:
        """Apply static quantization to the model with calibration."""
        model_to_quantize = self.float_model
        model_to_quantize.eval()
        
        # Use the provided qconfig_dict
        torch.quantization.propagate_qconfig_(model_to_quantize, qconfig_dict)
        
        # Prepare the model for static quantization
        torch.quantization.prepare(model_to_quantize, inplace=True)
        
        # Calibration process
        self.logger.info("Performing calibration for static quantization...")
        calibration_data = self._get_calibration_data()
        with torch.no_grad():
            for data in calibration_data:
                model_to_quantize(data)
        
        # Convert the model to quantized version
        torch.quantization.convert(model_to_quantize, inplace=True, dtype=dtype)
        return model_to_quantize

    def _static_quantize_fx(self, qconfig_mapping: QConfigMapping, 
                            dtype: torch.dtype) -> nn.Module:
        """Apply static quantization using FX Graph Mode."""
        model_to_quantize = self.float_model
        model_to_quantize.eval()
        
        example_inputs = self._get_example_inputs()
        prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs, 
                                    prepare_custom_config=self.prepare_custom_config)
        
        # Calibration process
        self.logger.info("Performing calibration for static quantization...")
        calibration_data = self._get_calibration_data()
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        # Use dtype in convert_fx
        quantized_model = convert_fx(prepared_model, convert_custom_config=self.convert_custom_config, dtype=dtype)
        return quantized_model

    def _dynamic_quantize_fx(self, qconfig_mapping: QConfigMapping, 
                            dtype: torch.dtype) -> nn.Module:
        """Apply dynamic quantization using FX Graph Mode."""
        model_to_quantize = self.float_model
        model_to_quantize.eval()
        
        example_inputs = self._get_example_inputs()
        prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs,
                                    prepare_custom_config=self.prepare_custom_config)
        
        # Use dtype in convert_fx
        quantized_model = convert_fx(prepared_model, convert_custom_config=self.convert_custom_config, dtype=dtype)
        return quantized_model
      
    def _get_calibration_data(self) -> List[torch.Tensor]:
        """Get representative data for calibration."""
        # Implement this method to return a list of representative input tensors
        raise NotImplementedError("Calibration data method needs to be implemented")

    def handle_non_traceable_modules(self, prepare_custom_config: Dict = None, 
                                     convert_custom_config: Dict = None) -> None:
        """
        Configure handling of non-traceable modules for FX Graph Mode Quantization.
        
        Args:
            prepare_custom_config (Dict): Custom configuration for prepare_fx.
            convert_custom_config (Dict): Custom configuration for convert_fx.
        """
        self.prepare_custom_config = prepare_custom_config or {}
        self.convert_custom_config = convert_custom_config or {}

    def compare_model_sizes(self) -> Tuple[float, float, float]:
        """
        Compare the sizes of the floating-point and quantized models.
        
        Returns:
            Tuple[float, float, float]: FP32 size, INT8 size, and reduction factor.
        """
        def get_model_size(model: nn.Module) -> float:
            torch.save(model.state_dict(), "temp.p")
            size = os.path.getsize("temp.p") / 1e6  # Size in MB
            os.remove("temp.p")
            return size

        fp32_size = get_model_size(self.float_model)
        int8_size = get_model_size(self.quantized_model) if self.quantized_model else 0
        reduction_factor = fp32_size / int8_size if int8_size > 0 else 0

        self.logger.info(f"FP32 model size: {fp32_size:.2f} MB")
        self.logger.info(f"INT8 model size: {int8_size:.2f} MB")
        self.logger.info(f"Size reduction factor: {reduction_factor:.2f}x")

        return fp32_size, int8_size, reduction_factor

    def measure_latency(self, input_data: torch.Tensor, 
                        num_runs: int = 100) -> Tuple[float, float]:
        """
        Measure and compare the latency of the floating-point and quantized models.
        
        Args:
            input_data (torch.Tensor): Input data for the model.
            num_runs (int): Number of runs for averaging latency.
        
        Returns:
            Tuple[float, float]: FP32 latency and INT8 latency in milliseconds.
        """
        def run_model(model: nn.Module) -> float:
            model.eval()
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    _ = model(input_data)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
            return (end_time - start_time) * 1000 / num_runs  # Convert to ms

        fp32_latency = run_model(self.float_model)
        int8_latency = run_model(self.quantized_model) if self.quantized_model else 0

        self.logger.info(f"FP32 model latency: {fp32_latency:.2f} ms")
        self.logger.info(f"INT8 model latency: {int8_latency:.2f} ms")
        self.logger.info(f"Speedup factor: {fp32_latency / int8_latency:.2f}x")

        return fp32_latency, int8_latency

    def compare_outputs(self, input_data: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compare the outputs of the floating-point and quantized models.
        
        Args:
            input_data (torch.Tensor): Input data for the model.
        
        Returns:
            Tuple[float, float, float]: FP32 magnitude, INT8 magnitude, and difference magnitude.
        """
        with torch.no_grad():
            fp32_out = self.float_model(input_data)
            int8_out = self.quantized_model(input_data) if self.quantized_model else torch.tensor([])

        fp32_mag = torch.mean(torch.abs(fp32_out)).item()
        int8_mag = torch.mean(torch.abs(int8_out)).item() if self.quantized_model else 0
        diff_mag = torch.mean(torch.abs(fp32_out - int8_out)).item() if self.quantized_model else 0

        self.logger.info(f"FP32 output magnitude: {fp32_mag:.6f}")
        self.logger.info(f"INT8 output magnitude: {int8_mag:.6f}")
        self.logger.info(f"Output difference magnitude: {diff_mag:.6f}")

        return fp32_mag, int8_mag, diff_mag

    def evaluate_accuracy(self, eval_fn: Callable[[nn.Module], float]) -> Tuple[float, float]:
        """
        Evaluate and compare the accuracy of the floating-point and quantized models.
        
        Args:
            eval_fn (Callable): Function to evaluate model accuracy.
        
        Returns:
            Tuple[float, float]: FP32 accuracy and INT8 accuracy.
        """
        fp32_accuracy = eval_fn(self.float_model)
        int8_accuracy = eval_fn(self.quantized_model) if self.quantized_model else 0

        self.logger.info(f"FP32 model accuracy: {fp32_accuracy:.4f}")
        self.logger.info(f"INT8 model accuracy: {int8_accuracy:.4f}")

        return fp32_accuracy, int8_accuracy

    def export_quantized_model(self, path: str) -> None:
        """
        Export the quantized model to a file.
        
        Args:
            path (str): Path to save the quantized model.
        """
        if self.quantized_model is None:
            self.logger.error("No quantized model available to export.")
            return

        try:
            torch.jit.save(torch.jit.script(self.quantized_model), path)
            self.logger.info(f"Quantized model exported to {path}")
        except Exception as e:
            self.logger.error(f"Failed to export quantized model: {str(e)}")
            raise

    def analyze_performance(self, input_data: torch.Tensor, 
                            num_runs: int = 100) -> Dict[str, float]:
        """
        Perform detailed performance analysis of the models.
        
        Args:
            input_data (torch.Tensor): Input data for the model.
            num_runs (int): Number of runs for averaging latency.
        
        Returns:
            Dict[str, float]: Dictionary containing various performance metrics.
        """
        results = {}
        results['fp32_latency'], results['int8_latency'] = self.measure_latency(input_data, num_runs)
        results['fp32_size'], results['int8_size'], results['size_reduction'] = self.compare_model_sizes()
        
        # Memory usage analysis
        results['fp32_memory'], results['int8_memory'] = self._analyze_memory_usage(input_data)
        
        return results

    def _analyze_memory_usage(self, input_data: torch.Tensor) -> Tuple[float, float]:
        """
        Analyze memory usage of both models.
        
        Args:
            input_data (torch.Tensor): Input data for the model.
        
        Returns:
            Tuple[float, float]: FP32 memory usage and INT8 memory usage in MB.
        """
        def get_memory_usage(model: nn.Module) -> float:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model(input_data)
            return torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

        fp32_memory = get_memory_usage(self.float_model)
        int8_memory = get_memory_usage(self.quantized_model) if self.quantized_model else 0

        self.logger.info(f"FP32 model peak memory usage: {fp32_memory:.2f} MB")
        self.logger.info(f"INT8 model peak memory usage: {int8_memory:.2f} MB")

        return fp32_memory, int8_memory

    def compare_outputs_detailed(self, input_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform a detailed comparison of the outputs from both models.
        
        Args:
            input_data (torch.Tensor): Input data for the model.
        
        Returns:
            Dict[str, float]: Dictionary containing various output comparison metrics.
        """
        with torch.no_grad():
            fp32_out = self.float_model(input_data)
            int8_out = self.quantized_model(input_data) if self.quantized_model else torch.tensor([])

        results = {}
        results['fp32_mean'] = torch.mean(fp32_out).item()
        results['fp32_std'] = torch.std(fp32_out).item()
        results['int8_mean'] = torch.mean(int8_out).item() if self.quantized_model else 0
        results['int8_std'] = torch.std(int8_out).item() if self.quantized_model else 0
        results['mse'] = torch.mean((fp32_out - int8_out) ** 2).item() if self.quantized_model else 0
        results['max_diff'] = torch.max(torch.abs(fp32_out - int8_out)).item() if self.quantized_model else 0

        self.logger.info(f"FP32 output mean: {results['fp32_mean']:.6f}, std: {results['fp32_std']:.6f}")
        self.logger.info(f"INT8 output mean: {results['int8_mean']:.6f}, std: {results['int8_std']:.6f}")
        self.logger.info(f"Mean Squared Error: {results['mse']:.6f}")
        self.logger.info(f"Maximum Absolute Difference: {results['max_diff']:.6f}")

        return results