from abc import ABC, abstractmethod
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import yaml
from tqdm import tqdm
from loguru import logger
import copy
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import QConfigMapping, default_dynamic_qconfig, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.ao.quantization import quantize_dynamic, get_default_qat_qconfig

class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def modify_for_quantization(self):
        pass

class AbstractDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None):
        return self.config.get(key, default)

config = Config()

torch.backends.quantized.engine = config.get('quantization_backend', 'qnnpack')

# Setup logger
logger.add(config.get('log_file', 'quantization.log'), rotation=config.get('log_rotation', '500 MB'))

class QuantizationWrapper:
    """Wrapper for PyTorch quantization APIs"""
    
    def __init__(self):
        self.qconfig_mapping = QConfigMapping()
        self.quantizer = None
        
    def set_global_qconfig(self, backend: str = 'qnnpack') -> None:
        try:
            qconfig = get_default_qconfig(backend)
            self.qconfig_mapping.set_global(qconfig)
        except ValueError as e:
            logger.error(f"Error setting global qconfig: {str(e)}")
            raise
        
    def set_qconfig_for_module(self, module_type: Type[nn.Module], qconfig: Any) -> None:
        self.qconfig_mapping.set_object_type(module_type, qconfig)
        
    def set_quantizer(self, backend: str = 'xnnpack') -> None:
        if backend == 'xnnpack':
            self.quantizer = XNNPACKQuantizer()
            self.quantizer.set_global(get_symmetric_quantization_config())
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
    def prepare_fx(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.eval()
        prepared_model = prepare_fx(model, self.qconfig_mapping, example_inputs)
        return prepared_model
    
    def prepare_qat_fx(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.train()
        prepared_model = prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
        return prepared_model
    
    def convert_fx(self, prepared_model: nn.Module) -> nn.Module:
        quantized_model = convert_fx(prepared_model)
        return quantized_model
    
    def quantize_dynamic(self, model: nn.Module, qconfig_spec: Optional[Dict[Type[nn.Module], Any]] = None) -> nn.Module:
        if qconfig_spec is None:
            qconfig_spec = {nn.Linear: default_dynamic_qconfig}
        quantized_model = quantize_dynamic(model, qconfig_spec)
        return quantized_model

class QuantizationCalibrator:
    def __init__(self, model: nn.Module, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader
        
    def calibrate(self, num_batches: Optional[int] = None) -> None:
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(tqdm(self.data_loader, desc="Calibrating")):
                if num_batches is not None and i >= num_batches:
                    break
                self.model(inputs)
        logger.info(f"Calibration completed using {i+1} batches.")

class QuantizationDebugger:
    def __init__(self, float_model: nn.Module, quantized_model: nn.Module):
        self.float_model = float_model
        self.quantized_model = quantized_model
        
    def compare_outputs(self, inputs: torch.Tensor) -> None:
        with torch.no_grad():
            float_output = self.float_model(inputs)
            quantized_output = self.quantized_model(inputs)
        
        diff = torch.abs(float_output - quantized_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        logger.info(f"Max absolute difference: {max_diff.item()}")
        logger.info(f"Mean absolute difference: {mean_diff.item()}")
        
    def print_model_size(self) -> None:
        def get_size(model: nn.Module) -> float:
            torch.save(model.state_dict(), "temp.p")
            size = os.path.getsize("temp.p") / (1024 * 1024)
            os.remove("temp.p")
            return size
        
        float_size = get_size(self.float_model)
        quantized_size = get_size(self.quantized_model)
        
        logger.info(f"Float model size: {float_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - quantized_size/float_size)*100:.2f}%")

class QuantizationRunner:
    def __init__(self, model: AbstractModel, example_inputs: Any, data_loader: DataLoader):
        self.model = model
        self.example_inputs = example_inputs
        self.data_loader = data_loader
        self.wrapper = QuantizationWrapper()
        
    def run_static_quantization(self) -> nn.Module:
        logger.info("Starting static quantization...")
        self.wrapper.set_global_qconfig(config.get('quantization_backend', 'qnnpack'))
        self.wrapper.set_quantizer()
        
        self.model.modify_for_quantization()
        prepared_model = self.wrapper.prepare_fx(self.model, self.example_inputs)
        
        calibrator = QuantizationCalibrator(prepared_model, self.data_loader)
        calibrator.calibrate(num_batches=config.get('static_quantization_calibration_batches', 10))
        
        quantized_model = self.wrapper.convert_fx(prepared_model)
        
        logger.info("Static quantization completed.")
        return quantized_model
    
    def run_qat(self, num_epochs: int = 5) -> nn.Module:
        logger.info("Starting quantization-aware training...")
        self.wrapper.set_global_qconfig(config.get('quantization_backend', 'qnnpack'))
        
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.modify_for_quantization()
        model_to_quantize.train()
        
        qconfig = get_default_qat_qconfig(config.get('quantization_backend', 'qnnpack'))
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        prepared_model = self.wrapper.prepare_qat_fx(model_to_quantize, self.example_inputs)
        
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=config.get('qat_learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for inputs, targets in tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                outputs = prepared_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            logger.info(f"Completed epoch {epoch+1}/{num_epochs}")
        
        quantized_model = self.wrapper.convert_fx(prepared_model.eval())
        logger.info("Quantization-aware training completed.")
        return quantized_model
    
    def run_dynamic_quantization(self) -> nn.Module:
        logger.info("Starting dynamic quantization...")
        self.model.modify_for_quantization()
        quantized_model = self.wrapper.quantize_dynamic(self.model)
        logger.info("Dynamic quantization completed.")
        return quantized_model

    def save_quantized_model(self, model: nn.Module, path: str) -> None:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path)
        logger.info(f"Quantized model saved to {path}")
      
    def load_quantized_model(self, path: str) -> nn.Module:
        model = torch.jit.load(path)
        logger.info(f"Quantized model loaded from {path}")
        return model
  
    def benchmark_performance(self, model: nn.Module, num_runs: int = 100) -> Tuple[float, float]:
        model.eval()
        times = []
        
        # Warm-up run
        _ = model(self.example_inputs)
        
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            with torch.no_grad():
                _ = model(self.example_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        logger.info(f"Standard deviation: {std_dev*1000:.2f} ms")
        
        return avg_time, std_dev

    def compare_models(self, float_model: nn.Module, quantized_model: nn.Module) -> None:
        debugger = QuantizationDebugger(float_model, quantized_model)
        debugger.print_model_size()
        
        logger.info("Benchmarking float model:")
        float_time, _ = self.benchmark_performance(float_model)
        
        logger.info("Benchmarking quantized model:")
        quant_time, _ = self.benchmark_performance(quantized_model)
        
        speedup = float_time / quant_time
        logger.info(f"Speedup: {speedup:.2f}x")

    def validate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Validating"):
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        logger.info(f"Validation Accuracy: {accuracy*100:.2f}%")
        return accuracy

class YourModel(AbstractModel):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def modify_for_quantization(self):
        # Add any specific modifications needed for quantization
        # For example, you might want to replace certain layers or fuse operations
        pass

class YourDataset(AbstractDataset):
    def __init__(self, size: int = 1000, img_size: int = 224):
        self.size = size
        self.img_size = img_size
        self.data = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for _ in range(size):
            if torch.rand(1).item() > 0.5:
                img = Image.fromarray(np.uint8(np.ones((img_size, img_size, 3)) * 255))
                label = 0
            else:
                img = Image.fromarray(np.uint8(np.zeros((img_size, img_size, 3))))
                label = 1
            
            self.data.append(img)
            self.labels.append(label)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Example usage:
if __name__ == "__main__":

    # Create datasets and data loaders
    train_dataset = YourDataset(size=config.get('train_dataset_size', 1000), img_size=config.get('img_size', 224))
    val_dataset = YourDataset(size=config.get('val_dataset_size', 200), img_size=config.get('img_size', 224))
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

    # Create your model
    model = YourModel(num_classes=config.get('num_classes', 2), pretrained=config.get('pretrained', True))

    # Prepare example inputs
    example_inputs = torch.randn(1, 3, config.get('img_size', 224), config.get('img_size', 224))

    # Initialize the QuantizationRunner
    runner = QuantizationRunner(model, example_inputs, train_loader)

    # Run static quantization
    quantized_model = runner.run_static_quantization()

    # Save the quantized model
    runner.save_quantized_model(quantized_model, config.get('quantized_model_path', 'quantized_model.pt'))

    # Load the quantized model
    loaded_quantized_model = runner.load_quantized_model(config.get('quantized_model_path', 'quantized_model.pt'))

    # Compare float and quantized models
    runner.compare_models(model, loaded_quantized_model)

    # Validate accuracy
    logger.info("Validating float model:")
    float_accuracy = runner.validate_accuracy(model, val_loader)
    logger.info("Validating quantized model:")
    quant_accuracy = runner.validate_accuracy(loaded_quantized_model, val_loader)

    logger.info(f"Float model accuracy: {float_accuracy*100:.2f}%")
    logger.info(f"Quantized model accuracy: {quant_accuracy*100:.2f}%")

    # Run dynamic quantization
    dynamic_quantized_model = runner.run_dynamic_quantization()

    # Compare float and dynamic quantized models
    runner.compare_models(model, dynamic_quantized_model)

    # Validate accuracy of dynamic quantized model
    logger.info("Validating dynamic quantized model:")
    dynamic_quant_accuracy = runner.validate_accuracy(dynamic_quantized_model, val_loader)
    logger.info(f"Dynamic quantized model accuracy: {dynamic_quant_accuracy*100:.2f}%")

    # Run quantization-aware training (QAT)
    qat_model = runner.run_qat(num_epochs=config.get('qat_num_epochs', 5))

    # Compare float and QAT models
    runner.compare_models(model, qat_model)

    # Validate accuracy of QAT model
    logger.info("Validating QAT model:")
    qat_accuracy = runner.validate_accuracy(qat_model, val_loader)
    logger.info(f"QAT model accuracy: {qat_accuracy*100:.2f}%")

    # Print final comparison
    logger.info("\nFinal Comparison:")
    logger.info(f"Float model accuracy: {float_accuracy*100:.2f}%")
    logger.info(f"Static quantized model accuracy: {quant_accuracy*100:.2f}%")
    logger.info(f"Dynamic quantized model accuracy: {dynamic_quant_accuracy*100:.2f}%")
    logger.info(f"QAT model accuracy: {qat_accuracy*100:.2f}%")