# My code starts from here
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
import os
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.backend_config import BackendConfig, get_native_backend_config
from torch.ao.quantization import (
  get_default_qconfig, QConfigMapping, prepare, convert, fuse_modules,
  get_default_qat_qconfig, prepare_qat
)
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
import torch.quantization._numeric_suite as ns
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_qat_pt2e

from torch.ao.quantization.quantizer import Quantizer, QuantizationAnnotation, QuantizationSpec
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch._export import capture_pre_autograd_graph
from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap

class QuantizationManager:
    def __init__(self, model: nn.Module) -> None:
        self.float_model = model
        self.quantized_model = None
        self.logger = self._setup_logger()
        self.prepare_custom_config = {}
        self.convert_custom_config = {}
        self.use_pt2e = False
        self.backend_config = get_native_backend_config()
        self.quantizer = None

    def set_quantizer(self, quantizer: Quantizer) -> None:
        """Set a custom Quantizer for PyTorch 2 Export Quantization."""
        self.quantizer = quantizer
        self.logger.info(f"Custom Quantizer set: {type(quantizer).__name__}")

    def _setup_logger(
        self
    ) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_example_inputs(
        self
    ) -> Tuple:
        raise NotImplementedError("Example inputs method needs to be implemented")

    def set_backend_config(self, backend_config: BackendConfig) -> None:
        """Set a custom BackendConfig for quantization."""
        self.backend_config = backend_config
        self.logger.info("Custom BackendConfig set.")
    
    def quantize(
        self, 
        quantizer: Optional[Quantizer] = None,
        backend: str = 'x86',
        static: bool = True,
        per_channel: bool = False,
        dtype: torch.dtype = torch.qint8,
        use_pt2e: bool = False
    ) -> nn.Module:
        try:
            self.use_pt2e = use_pt2e
            if use_pt2e:
                if quantizer is None:
                    quantizer = self.quantizer or XNNPACKQuantizer()
                    if per_channel:
                        quantizer.set_global(get_symmetric_quantization_config())
                    else:
                        quantizer.set_global(get_symmetric_quantization_config())

                self.logger.info("Applying PyTorch 2 Export Quantization...")
                example_inputs = self._get_example_inputs()
                exported_model = capture_pre_autograd_graph(self.float_model, example_inputs)

                if static:
                    self.quantized_model = self._static_quantize_pt2e(exported_model, quantizer)
                else:
                    self.quantized_model = self._dynamic_quantize_pt2e(exported_model, quantizer)
            else:
                # Existing quantization logic
                qconfig_mapping = get_default_qconfig_mapping(backend)
                if static:
                    self.quantized_model = self._static_quantize_fx(qconfig_mapping, dtype, backend)
                else:
                    self.quantized_model = self._dynamic_quantize_fx(qconfig_mapping, dtype, backend)

            return self.quantized_model
        except Exception as e:
            self.logger.error(f"Quantization failed: {str(e)}")
            raise

    def _static_quantize_pt2e(self, exported_model: nn.Module, quantizer: Quantizer) -> nn.Module:
        prepared_model = prepare_pt2e(exported_model, quantizer)

        self.logger.info("Performing calibration for static quantization...")
        calibration_data = self._get_calibration_data()
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)

        quantized_model = convert_pt2e(prepared_model)
        return quantized_model

    def _dynamic_quantize_pt2e(self, exported_model: nn.Module, quantizer: Quantizer) -> nn.Module:
        prepared_model = prepare_pt2e(exported_model, quantizer)
        quantized_model = convert_pt2e(prepared_model)
        return quantized_model

    def _static_quantize_fx(
        self, 
        qconfig_mapping: QConfigMapping, 
        dtype: torch.dtype, 
        backend: str
    ) -> nn.Module:
        model_to_quantize = self.float_model
        model_to_quantize.eval()
        
        model_to_quantize = self.fuse_modules(model_to_quantize)
        
        example_inputs = self._get_example_inputs()
        prepared_model = prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs, prepare_custom_config=self.prepare_custom_config,
            backend_config=self.backend_config
        )
        
        self.logger.info("Performing calibration for static quantization...")
        calibration_data = self._get_calibration_data()
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        quantized_model = convert_fx(prepared_model, convert_custom_config=self.convert_custom_config)
        return quantized_model

    def _dynamic_quantize_fx(
        self, 
        qconfig_mapping: QConfigMapping, 
        dtype: torch.dtype, 
        backend: str
    ) -> nn.Module:
        model_to_quantize = self.float_model
        model_to_quantize.eval()
        
        example_inputs = self._get_example_inputs()
        prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs,
                                    prepare_custom_config=self.prepare_custom_config,
                                    backend_config=self.backend_config)
        
        quantized_model = convert_fx(prepared_model, convert_custom_config=self.convert_custom_config, dtype=dtype)
        return quantized_model
  
    def _get_calibration_data(
        self
    ) -> List[torch.Tensor]:
        raise NotImplementedError("Calibration data method needs to be implemented")

    def handle_non_traceable_modules(
        self, 
        prepare_custom_config: Dict = None, 
        convert_custom_config: Dict = None
    ) -> None:
        self.prepare_custom_config = prepare_custom_config or {}
        self.convert_custom_config = convert_custom_config or {}

    def fuse_modules(
        self, 
        model: nn.Module
    ) -> nn.Module:
        model.eval()
        model = fuse_modules(model, [['conv', 'bn', 'relu']])
        return model

    def quantization_aware_training(
        self, 
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        backend: str = 'x86'
    ) -> nn.Module:
        if self.use_pt2e:
            example_inputs = self._get_example_inputs()
            exported_model = capture_pre_autograd_graph(self.float_model, example_inputs)
            quantizer = XNNPACKQuantizer()
            quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
            prepared_model = prepare_qat_pt2e(exported_model, quantizer)
        else:
            qconfig = get_default_qat_qconfig(backend)
            prepared_model = prepare_qat(self.float_model, {'': qconfig})
        
        for epoch in range(num_epochs):
            # Training loop
            prepared_model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = prepared_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluation loop
            prepared_model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    output = prepared_model(data)
                    # Compute and log evaluation metrics
        
        if self.use_pt2e:
            quantized_model = convert_pt2e(prepared_model)
        else:
            quantized_model = convert(prepared_model.eval(), inplace=False)
        
        return quantized_model

    def compute_error(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        Compute the Signal-to-Quantization-Noise Ratio (SQNR) between two tensors.
        """
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn).item()

    def match_pattern(self, pattern_fn: Callable, model: Optional[nn.Module] = None) -> List[Dict[str, nn.Module]]:
        """
        Match a pattern in the model using SubgraphMatcherWithNameNodeMap.
        
        Args:
            pattern_fn (Callable): Function defining the pattern to match.
            model (Optional[nn.Module]): Model to search for the pattern. If None, uses self.float_model.
        
        Returns:
            List[Dict[str, nn.Module]]: List of matched subgraphs.
        """
        if model is None:
            model = self.float_model
        
        matcher = SubgraphMatcherWithNameNodeMap(pattern_fn)
        matches = matcher.match(model)
        
        return [match.name_node_map for match in matches]

    def annotate_model(self, annotations: Dict[str, QuantizationAnnotation]) -> None:
        """
        Annotate the model with QuantizationAnnotations.
        
        Args:
            annotations (Dict[str, QuantizationAnnotation]): Dictionary mapping node names to QuantizationAnnotations.
        """
        for name, module in self.float_model.named_modules():
            if name in annotations:
                module.meta["quantization_annotation"] = annotations[name]
        
        self.logger.info(f"Model annotated with {len(annotations)} annotations.")

    def create_quantization_spec(
        self,
        dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
        qscheme: torch.qscheme,
        observer_cls: Optional[Callable] = None
    ) -> QuantizationSpec:
        """
        Create a QuantizationSpec object.
        
        Args:
            dtype (torch.dtype): Quantization data type.
            quant_min (int): Minimum quantization value.
            quant_max (int): Maximum quantization value.
            qscheme (torch.qscheme): Quantization scheme.
            observer_cls (Optional[Callable]): Observer class to use.
        
        Returns:
            QuantizationSpec: Created QuantizationSpec object.
        """
        return QuantizationSpec(
            dtype=dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=qscheme,
            observer_or_fake_quant_ctr=observer_cls
        )

    def benchmark(
        self,
        input_data: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        results = super().benchmark(input_data, num_runs)
        
        # Add Numeric Suite comparisons
        wt_compare_dict = self.compare_weights_numeric_suite()
        act_compare_dict = self.compare_model_outputs_numeric_suite(input_data)
        
        # Compute average weight and activation errors
        avg_weight_error = sum(self.compute_error(v['float'], v['quantized'].dequantize()) for v in wt_compare_dict.values()) / len(wt_compare_dict)
        avg_act_error = sum(self.compute_error(v['float'][0], v['quantized'][0].dequantize()) for v in act_compare_dict.values()) / len(act_compare_dict)
        
        results['avg_weight_error'] = avg_weight_error
        results['avg_activation_error'] = avg_act_error
        
        self.logger.info(f"Average weight error: {avg_weight_error:.2f} dB")
        self.logger.info(f"Average activation error: {avg_act_error:.2f} dB")
        
        return results

    def measure_latency(
        self, 
        input_data: torch.Tensor, 
        num_runs: int = 100
    ) -> Tuple[float, float]:
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

    def evaluate_accuracy(
        self, 
        eval_fn: Callable[[nn.Module], float]
    ) -> Tuple[float, float]:
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
        if self.quantized_model is None:
            self.logger.error("No quantized model available to export.")
            return

        try:
            example_inputs = self._get_example_inputs()
            quantized_ep = torch.export.export(self.quantized_model, example_inputs)
            torch.export.save(quantized_ep, path)
            self.logger.info(f"Quantized model exported to {path}")
        except Exception as e:
            self.logger.error(f"Failed to export quantized model: {str(e)}")
            raise
    
    def load_quantized_model(self, path: str) -> nn.Module:
        try:
            loaded_quantized_ep = torch.export.load(path)
            self.quantized_model = loaded_quantized_ep.module()
            self.logger.info(f"Quantized model loaded from {path}")
            return self.quantized_model
        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {str(e)}")
            raise

    def compare_outputs_detailed(
        self, 
        input_data: torch.Tensor
    ) -> Dict[str, Union[float, torch.Tensor]]:
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

        # Histogram analysis
        results['fp32_histogram'] = torch.histogram(fp32_out.float(), bins=100)
        results['int8_histogram'] = torch.histogram(int8_out.float(), bins=100) if self.quantized_model else None

        self.logger.info(f"FP32 output mean: {results['fp32_mean']:.6f}, std: {results['fp32_std']:.6f}")
        self.logger.info(f"INT8 output mean: {results['int8_mean']:.6f}, std: {results['int8_std']:.6f}")
        self.logger.info(f"Mean Squared Error: {results['mse']:.6f}")
        self.logger.info(f"Maximum Absolute Difference: {results['max_diff']:.6f}")

        return results

    def compare_model_sizes(
      self
    ) -> Tuple[float, float, float]:
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

    def compare_outputs(
        self, 
        input_data: torch.Tensor
    ) -> Tuple[float, float, float]:
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

    def compare_outputs_detailed(
        self,
        input_data: torch.Tensor
    ) -> Dict[str, float]:
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

    def compare_weights_numeric_suite(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.quantized_model is None:
            self.logger.error("No quantized model available for comparison.")
            return {}
        
        wt_compare_dict = ns.compare_weights(self.float_model.state_dict(), self.quantized_model.state_dict())
        
        for key, value in wt_compare_dict.items():
            if 'float' in value and 'quantized' in value:
                error = self.compute_error(value['float'], value['quantized'].dequantize())
                self.logger.info(f"{key} weight error: {error:.2f} dB")
        
        return wt_compare_dict

    def compare_model_outputs_numeric_suite(self, input_data: torch.Tensor) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        if self.quantized_model is None:
            self.logger.error("No quantized model available for comparison.")
            return {}
        
        act_compare_dict = ns.compare_model_outputs(self.float_model, self.quantized_model, input_data)
        
        for key, value in act_compare_dict.items():
            if 'float' in value and 'quantized' in value:
                error = self.compute_error(value['float'][0], value['quantized'][0].dequantize())
                self.logger.info(f"{key} activation error: {error:.2f} dB")
        
        return act_compare_dict

    def compare_model_stub_numeric_suite(
        self,
        input_data: torch.Tensor,
        module_swap_list: List[nn.Module]
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """
        Compare quantized modules with their float counterparts using Numeric Suite.
        """
        if self.quantized_model is None:
            self.logger.error("No quantized model available for comparison.")
            return {}
        
        ob_dict = ns.compare_model_stub(self.float_model, self.quantized_model, module_swap_list, input_data)
        
        for key, value in ob_dict.items():
            if 'float' in value and 'quantized' in value:
                error = self.compute_error(value['float'][0], value['quantized'][0].dequantize())
                self.logger.info(f"{key} module error: {error:.2f} dB")
        
        return ob_dict

    def compare_model_representations(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {}
        with torch.no_grad():
            fp32_out = self.float_model(input_data)
            int8_out = self.quantized_model(input_data) if self.quantized_model else torch.tensor([])

        results['fp32_out'] = fp32_out
        results['int8_out'] = int8_out
        results['q_dq_representation'] = self._get_q_dq_representation(self.quantized_model)
        results['reference_quantized_representation'] = self._get_reference_quantized_representation(self.quantized_model)

        return results

    def _get_q_dq_representation(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        # Implementation for Q/DQ representation
        pass

    def _get_reference_quantized_representation(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        # Implementation for Reference Quantized Model representation
        pass

    def analyze_performance(
        self, 
        input_data: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        results = super().analyze_performance(input_data, num_runs)
        
        # Add Numeric Suite analyses
        wt_compare_dict = self.compare_weights_numeric_suite()
        act_compare_dict = self.compare_model_outputs_numeric_suite(input_data)
        
        results['avg_weight_error'] = sum(self.compute_error(v['float'], v['quantized'].dequantize()) for v in wt_compare_dict.values()) / len(wt_compare_dict)
        results['avg_activation_error'] = sum(self.compute_error(v['float'][0], v['quantized'][0].dequantize()) for v in act_compare_dict.values()) / len(act_compare_dict)
        
        return results

    def _analyze_memory_usage(
        self, 
        input_data: torch.Tensor
    ) -> Tuple[float, float]:
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

    def analyze_scalability(
        self,
        input_sizes: List[Tuple[int,
        ...]],
        num_runs: int = 10
    ) -> Dict[str, List[float]]:
        results = {'float_latency': [], 'quantized_latency': [], 'input_sizes': []}
        
        for size in input_sizes:
            input_data = torch.randn(*size)
            results['input_sizes'].append(str(size))
            
            for model_name, model in [("float", self.float_model), ("quantized", self.quantized_model)]:
                model.eval()
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    for _ in range(num_runs):
                        _ = model(input_data)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000 / num_runs  # Convert to ms
                results[f'{model_name}_latency'].append(latency)
        
        self.logger.info("Scalability analysis results:")
        for i, size in enumerate(results['input_sizes']):
            self.logger.info(f"Input size: {size}")
            self.logger.info(f"  Float model latency: {results['float_latency'][i]:.2f} ms")
            self.logger.info(f"  Quantized model latency: {results['quantized_latency'][i]:.2f} ms")
        
        return results
