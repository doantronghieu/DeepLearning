# My code starts from here
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Tuple
import time
import logging
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device: str = 'cuda'
    use_amp: bool = True
    grad_clip: Optional[float] = None
    grad_penalty: Optional[float] = None
    iters_to_accumulate: int = 1

class TrainingManager:
    def __init__(self, models: List[nn.Module], optimizers: List[torch.optim.Optimizer],
                 loss_fns: List[nn.Module], config: TrainingConfig) -> None:
        self.models = [model.to(config.device) for model in models]
        self.optimizers = optimizers
        self.loss_fns = loss_fns
        self.config = config
        self.scaler = GradScaler(enabled=self.config.use_amp)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def train(self, dataloaders: List[torch.utils.data.DataLoader], epochs: int) -> None:
        for model in self.models:
            model.train()
        
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch+1}")
            start_time = time.time()
            
            for batch_idx, batches in enumerate(zip(*dataloaders)):
                if batch_idx % self.config.iters_to_accumulate == 0:
                    for optimizer in self.optimizers:
                        optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=self.config.device, dtype=torch.float16, enabled=self.config.use_amp):
                    losses = self._compute_losses(batches)
                    total_loss = sum(losses) / self.config.iters_to_accumulate

                self.scaler.scale(total_loss).backward()

                if (batch_idx + 1) % self.config.iters_to_accumulate == 0:
                    self._update_models()
            
            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
            self._log_memory_stats()

    def _compute_losses(self, batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        losses = []
        for i, (inputs, targets) in enumerate(batches):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            outputs = self.models[i](inputs)
            loss = self.loss_fns[i](outputs, targets)
            losses.append(loss)
        
        if self.config.grad_penalty:
            penalties = self._compute_gradient_penalty(losses)
            losses = [loss + penalty for loss, penalty in zip(losses, penalties)]
        
        return losses

    def _update_models(self) -> None:
        if self.config.grad_clip:
            self._clip_gradients()
        
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()

    def evaluate(self, dataloaders: List[torch.utils.data.DataLoader]) -> List[float]:
        for model in self.models:
            model.eval()
        
        total_losses = [0.0 for _ in range(len(self.models))]
        with torch.no_grad():
            for batches in zip(*dataloaders):
                with autocast(device_type=self.config.device, dtype=torch.float16, enabled=self.config.use_amp):
                    for i, (inputs, targets) in enumerate(batches):
                        inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                        outputs = self.models[i](inputs)
                        loss = self.loss_fns[i](outputs, targets)
                        total_losses[i] += loss.item()
        
        return [total_loss / len(dataloader) for total_loss, dataloader in zip(total_losses, dataloaders)]

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "models": [model.state_dict() for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "scaler": self.scaler.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.config.device)
        for model, state_dict in zip(self.models, checkpoint["models"]):
            model.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state_dict)
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.config = checkpoint["config"]
        self.logger.info(f"Checkpoint loaded from {path}")

    def _log_memory_stats(self) -> None:
        memory_stats = self.get_memory_stats()
        self.logger.info(f"Memory stats: {memory_stats}")

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "current_memory": torch.cuda.memory_allocated(),
            "max_memory": torch.cuda.max_memory_allocated(),
            "reserved_memory": torch.cuda.memory_reserved(),
            "max_reserved_memory": torch.cuda.max_memory_reserved()
        }

    def _clip_gradients(self) -> None:
        self.scaler.unscale_(self.optimizers[0])
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

    def _compute_gradient_penalty(self, losses: List[torch.Tensor]) -> List[torch.Tensor]:
        penalties = []
        for i, loss in enumerate(losses):
            grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss),
                                              inputs=self.models[i].parameters(),
                                              create_graph=True)
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            penalties.append(self.config.grad_penalty * grad_norm)
        return penalties

    @staticmethod
    def check_tensor_core_compatibility(model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                sizes = param.size()
                if any(size % 8 != 0 for size in sizes):
                    logging.warning(f"Parameter {name} with shape {sizes} may not be optimal for Tensor Core usage.")