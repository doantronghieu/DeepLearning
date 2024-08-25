import os
import time
import logging
from typing import Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig, 
    QConfigMapping,
    default_dynamic_qconfig,
    default_qconfig,
    get_default_qconfig,
    prepare_fx, 
    convert_fx,
    prepare_qat_fx,
    quantize_dynamic,
    quantize_static,
    quantize_qat,
    fuse_modules
)
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    FakeQuantize
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationConfig:
    def __init__(self,
                 qconfig: Optional[QConfig] = None,
                 dtype: Optional[torch.dtype] = None,
                 backend: str = 'fbgemm',
                 calibration: bool = True,
                 qat: bool = False,
                 per_channel: bool = False,
                 observer: Optional[Callable] = None):
        self.qconfig = qconfig or get_default_qconfig(backend)
        self.dtype = dtype or torch.qint8
        self.backend = backend
        self.calibration = calibration
        self.qat = qat
        self.per_channel = per_channel
        self.observer = observer

class QuantizableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def fuse_model(self):
        # To be implemented by subclasses
        pass
    
    def configure_qat(self, qconfig: QConfig):
        # Default implementation
        self.qconfig = qconfig

class QuantizationEngine:
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config
        self.prepared_model = None
        self.quantized_model = None
        
    def prepare(self):
        logger.info("Preparing model for quantization")
        if isinstance(self.model, QuantizableModule):
            self.model.fuse_model()
        
        qconfig_mapping = QConfigMapping().set_global(self.config.qconfig)
        
        if self.config.qat:
            self.prepared_model = prepare_qat_fx(self.model, qconfig_mapping)
        else:
            self.prepared_model = prepare_fx(self.model, qconfig_mapping)
        
    def calibrate(self, data_loader):
        if not self.prepared_model:
            raise ValueError("Model must be prepared before calibration")
        
        logger.info("Calibrating model")
        if self.config.calibration:
            with torch.no_grad():
                for input, _ in data_loader:
                    self.prepared_model(input)
        
    def convert(self):
        if not self.prepared_model:
            raise ValueError("Model must be prepared before conversion")
        
        logger.info("Converting model to quantized version")
        self.quantized_model = convert_fx(self.prepared_model)
        
    def quantize_dynamic(self):
        logger.info("Performing dynamic quantization")
        self.quantized_model = quantize_dynamic(
            self.model, 
            qconfig_spec={'': default_dynamic_qconfig},
            dtype=self.config.dtype
        )
        
    def quantize_static(self):
        logger.info("Performing static quantization")
        self.prepare()
        self.convert()
        
    def quantize_qat(self, train_func: Callable):
        logger.info("Performing quantization-aware training")
        self.prepare()
        train_func(self.prepared_model)
        self.convert()
        
    def get_quantized_model(self) -> nn.Module:
        if not self.quantized_model:
            raise ValueError("Model has not been quantized yet")
        return self.quantized_model

    def save_quantized_model(self, path: str):
        if not self.quantized_model:
            raise ValueError("Model has not been quantized yet")
        torch.save(self.quantized_model.state_dict(), path)

    def load_quantized_model(self, path: str):
        if not self.quantized_model:
            raise ValueError("Model has not been quantized yet")
        self.quantized_model.load_state_dict(torch.load(path))

class QuantizationProfiler:
    def __init__(self, model: nn.Module):
        self.model = model
        
    def print_model_size(self):
        torch.save(self.model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")/1e6
        os.remove("temp.p")
        logger.info(f'Model Size: {size:.2f} MB')
        
    def benchmark(self, input_data, num_runs=100):
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(num_runs):
                self.model(input_data)
            end = time.time()
        avg_time = (end-start)/num_runs*1000
        logger.info(f'Average inference time: {avg_time:.2f} ms')

    def per_layer_analysis(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.quantization.QuantizedModule):
                logger.info(f'Layer: {name}, Type: {type(module).__name__}')

def quantize_model(model: nn.Module, 
                   config: QuantizationConfig,
                   calibration_data: Optional[torch.utils.data.DataLoader] = None,
                   train_func: Optional[Callable] = None) -> nn.Module:
    engine = QuantizationEngine(model, config)
    
    if config.qat:
        if not train_func:
            raise ValueError("train_func must be provided for QAT")
        engine.quantize_qat(train_func)
    elif config.calibration:
        engine.quantize_static()
        if calibration_data:
            engine.calibrate(calibration_data)
    else:
        engine.quantize_dynamic()
        
    return engine.get_quantized_model()

# Usage example:
class MyQuantizableModel(QuantizableModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        fuse_modules(self, ['conv', 'relu'], inplace=True)

model = MyQuantizableModel()
config = QuantizationConfig(backend='qnnpack', qat=True, per_channel=True)
quantized_model = quantize_model(model, config, train_func=lambda m: None)  # placeholder train func

profiler = QuantizationProfiler(quantized_model)
profiler.print_model_size()
profiler.benchmark(torch.randn(1, 3, 224, 224))
profiler.per_layer_analysis()