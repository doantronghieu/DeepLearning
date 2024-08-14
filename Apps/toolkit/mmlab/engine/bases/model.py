from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List, Type

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from argparse import ArgumentParser

class MyBaseModel(BaseModel, ABC):
    def __init__(self, 
                 init_cfg: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                 data_preprocessor: Optional[Union[Dict[str, Any], nn.Module]] = None):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.init_cfg = init_cfg

    @abstractmethod
    def forward(self, 
                inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                data_samples: Optional[Any] = None, 
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], torch.Tensor, Tuple[Any, ...], List[Any]]:
        pass

    def train_step(self, data: Dict[str, Any], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        data = self.data_preprocessor(data, training=True)
        losses = self(**data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = self.data_preprocessor(data, training=False)
        outputs = self(**data, mode='predict')
        return outputs

    def test_step(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.val_step(data)

    def extract_feat(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self(inputs, mode='tensor')

    def _parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parsed_loss, log_vars = super()._parse_losses(losses)
        return parsed_loss, log_vars

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("MyBaseModel")
        parser.add_argument('--custom_arg', type=str, help='Custom argument for the model')
        return parent_parser

    def init_weights(self) -> None:
        if self.init_cfg:
            if isinstance(self.init_cfg, dict):
                self.init_cfg = [self.init_cfg]
            for cfg in self.init_cfg:
                if cfg['type'] == 'Pretrained':
                    self._load_pretrained(cfg['checkpoint'])
                elif cfg['type'] in ['Kaiming', 'Xavier', 'Normal', 'Uniform']:
                    self._initialize_weights(cfg)
        else:
            super().init_weights()

    def _load_pretrained(self, checkpoint: str) -> None:
        self.load_state_dict(torch.load(checkpoint), strict=False)

    def _initialize_weights(self, cfg: Dict[str, Any]) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if cfg['type'] == 'Kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif cfg['type'] == 'Xavier':
                    nn.init.xavier_normal_(m.weight)
                elif cfg['type'] == 'Normal':
                    nn.init.normal_(m.weight, mean=cfg.get('mean', 0), std=cfg.get('std', 0.01))
                elif cfg['type'] == 'Uniform':
                    nn.init.uniform_(m.weight, a=cfg.get('a', 0), b=cfg.get('b', 1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, cfg.get('bias', 0))

    @torch.no_grad()
    def to_ema(self, momentum: float = 0.999) -> None:
        if not hasattr(self, 'ema_model'):
            # Create a new instance of the same class
            self.ema_model = type(self)(init_cfg=self.init_cfg, data_preprocessor=self.data_preprocessor)
            self.ema_model.eval()
            self.ema_model.load_state_dict(self.state_dict())
        else:
            for param, ema_param in zip(self.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)


