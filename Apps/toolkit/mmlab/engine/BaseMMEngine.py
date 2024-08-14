from typing import Dict, Any, Callable, Type
from loguru import logger

from mmengine.model import BaseModel
from mmengine.dataset import BaseDataset
from mmengine.evaluator import BaseMetric


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def register_module(self, name: str = None, module: Type = None, force: bool = False) -> Callable:
        def _register(module):
            if not name:
                _name = module.__name__
            else:
                _name = name
            if not force and _name in self._module_dict:
                raise KeyError(f"{_name} is already registered in {self.name}")
            self._module_dict[_name] = module
            return module

        if module is not None:
            return _register(module)
        return _register

    def get(self, name: str) -> Type:
        return self._module_dict.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

class BaseMMEngine:
    def __init__(self, config_file: str) -> None:
        self.config = self.load_config(config_file)
        self.registries = {
            'models': Registry('models'),
            'datasets': Registry('datasets'),
            'transforms': Registry('transforms'),
            'optimizers': Registry('optimizers'),
            'schedulers': Registry('schedulers'),
            'hooks': Registry('hooks')
        }
        self.validate_config()
        self.parse_config()

        # Initialize model, optimizer, and other components
        self.model: BaseModel = self.build_from_cfg(self.config, 'models')
        self.train_dataset: BaseDataset = self.build_from_cfg(self.config, 'datasets')
        self.val_dataset: BaseDataset = self.build_from_cfg(self.config, 'datasets')
        self.optimizer = self.build_from_cfg(self.config, 'optimizers')

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            config_str = f.read()
        config_dict = {}
        exec(config_str, config_dict)
        return {k: v for k, v in config_dict.items() if not k.startswith('__')}

    def validate_config(self) -> None:
        required_keys = [
            'model', 'work_dir', 'train_dataloader', 'train_cfg',
            'optim_wrapper', 'val_dataloader', 'val_cfg'
        ]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Required configuration key '{key}' is missing.")

    def parse_config(self) -> None:
        for key, value in self.config.items():
            setattr(self, key, value)

    def register_module(self, registry_name: str, name: str = None, module: Type = None, force: bool = False) -> Callable:
        if registry_name not in self.registries:
            raise KeyError(f"Registry {registry_name} does not exist.")
        return self.registries[registry_name].register_module(name, module, force)

    def build_from_cfg(self, cfg: Dict[str, Any], registry_name: str, **kwargs: Any) -> Any:
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

        cfg_ = cfg.copy()
        module_type = cfg_.pop('type')
        if registry_name not in self.registries:
            raise KeyError(f"Registry {registry_name} does not exist.")

        if module_type not in self.registries[registry_name]:
            raise KeyError(f'{module_type} is not in the {registry_name} registry')

        module_cls = self.registries[registry_name].get(module_type)
        module = module_cls(**cfg_, **kwargs)

        return module

    def get_runner_config(self) -> Dict[str, Any]:
        return self.config

    def train(self) -> None:
        logger.info("Starting training...")
        # Here you would set up your training loop using the initialized components
        logger.info("Training completed.")

    def test(self) -> None:
        logger.info("Starting testing...")
        # Implement test logic here
        logger.info("Testing completed.")

    def infer(self, input_data: Any) -> Any:
        logger.info("Starting inference...")
        # Implement inference logic here
        logger.info("Inference completed.")
        return None  # Replace with actual inference result

    def export(self, format: str) -> None:
        logger.info(f"Exporting model to {format} format...")
        # Implement export logic here
        logger.info("Export completed.")


###

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from mmengine.model import BaseDataPreprocessor

class BaseModel(nn.Module, ABC):
    def __init__(self, data_preprocessor: Dict = None, init_cfg: Dict = None):
        super().__init__()
        self.data_preprocessor = BaseDataPreprocessor(data_preprocessor) if data_preprocessor else None
        self.init_cfg = init_cfg

    @abstractmethod
    def forward(self, inputs: torch.Tensor, data_samples=None, mode: str = 'tensor') -> Union[Dict, Tuple, torch.Tensor]:
        # Sample implementation of forward method
        if mode == 'loss':
            return {'loss': F.cross_entropy(inputs, data_samples)}
        elif mode == 'predict':
            return torch.argmax(inputs, dim=1)
        elif mode == 'tensor':
            return inputs

    def parse_losses(self, losses: Dict) -> Tuple[torch.Tensor, Dict]:
        # Sample implementation of parse_losses method
        loss = sum(loss for loss in losses.values())
        return loss, losses

    def train_step(self, data, optim_wrapper) -> Dict[str, torch.Tensor]:
        # Sample implementation of train_step method
        data = self.data_preprocessor(data, training=True)
        losses = self(**data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data) -> list:
        # Sample implementation of val_step method
        data = self.data_preprocessor(data, training=False)
        outputs = self(**data, mode='predict')
        return outputs

    def test_step(self, data) -> list:
        # Sample implementation of test_step method
        return self.val_step(data)
