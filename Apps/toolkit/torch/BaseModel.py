# My code starts from here
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC, nn.Module):
  def __init__(self, model: nn.Module, loss_fn: nn.Module) -> None:
    pass
  
  @abstractmethod()
  def forward(self, x):
    pass

  @abstractmethod()
  def compute_loss(self, **kwargs):
    pass