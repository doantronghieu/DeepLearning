import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type
from loguru import logger
from torch.utils.data import DataLoader
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
import os
from mmengine.dist import init_dist

class BaseMMengine(ABC):
    def __init__(self) -> None:
        """
        Base class for MMEngine-based machine learning pipelines.
        More information at: https://mmengine.readthedocs.io/en/latest/
        """
        super().__init__()
        self.model: Optional[BaseModel] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.metric: Optional[BaseMetric] = None
        self.runner: Optional[Runner] = None

    @abstractmethod
    def build_model(self) -> BaseModel:
        pass
    
    @abstractmethod
    def build_dataset(self) -> Tuple[DataLoader, DataLoader]:
        pass
    
    @abstractmethod
    def build_metric(self) -> BaseMetric:
        pass
    
    def create_strategy(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """
        Factory method to create a strategy configuration based on the specified type.
        
        Args:
            strategy_type (str): The type of strategy to create ('default', 'deepspeed', 'fsdp', 'colossalai')
            **kwargs: Additional keyword arguments for strategy configuration
        
        Returns:
            Dict[str, Any]: The strategy configuration
        """
        if strategy_type == 'default':
            return {}
        elif strategy_type == 'deepspeed':
            return {
                'type': 'DeepSpeedStrategy',
                'fp16': kwargs.get('fp16', {'enabled': True}),
                'zero_optimization': kwargs.get('zero_optimization', {'stage': 3})
            }
        elif strategy_type == 'fsdp':
            return {
                'type': 'FSDPStrategy',
                'model_wrapper': kwargs.get('model_wrapper', {})
            }
        elif strategy_type == 'colossalai':
            return {'type': 'ColossalAIStrategy'}
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

    def create_optimizer(self, optimizer_type: str, **kwargs) -> Dict[str, Any]:
        """
        Factory method to create an optimizer configuration based on the specified type.
        
        Args:
            optimizer_type (str): The type of optimizer to create
            **kwargs: Additional keyword arguments for optimizer configuration
        
        Returns:
            Dict[str, Any]: The optimizer configuration
        """
        base_optimizers = {
            'SGD': dict(type='SGD', lr=0.01, momentum=0.9),
            'Adam': dict(type='Adam', lr=0.001, betas=(0.9, 0.999)),
            'AdamW': dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
        }
        
        advanced_optimizers = {
            'DAdaptAdaGrad': dict(type='DAdaptAdaGrad', lr=0.01),
            'DAdaptAdam': dict(type='DAdaptAdam', lr=0.01),
            'DAdaptSGD': dict(type='DAdaptSGD', lr=0.01),
            'Lion': dict(type='Lion', lr=1e-4, weight_decay=1e-2),
            'Sophia': dict(type='Sophia', lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1),
            'SophiaG': dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1),
            'AdamW8bit': dict(type='AdamW8bit', lr=1e-4, weight_decay=1e-2),
            'Adafactor': dict(type='Adafactor', lr=1e-5, weight_decay=1e-2, scale_parameter=False, relative_step=False),
        }
        
        all_optimizers = {**base_optimizers, **advanced_optimizers}
        
        if optimizer_type not in all_optimizers:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        optimizer_config = all_optimizers[optimizer_type].copy()
        optimizer_config.update(kwargs)
        
        return optimizer_config

    
    def configure_runner(
        self, work_dir: str, max_epochs: int, val_interval: int, 
        resume: bool = False, load_from: Optional[str] = None,
        distributed: bool = False, launcher: Optional[str] = None,
        use_amp: bool = False, accumulative_counts: int = 1,
        use_grad_checkpoint: bool = False, compile_model: bool = False,
        efficient_conv_bn_eval: bool = False,
        strategy_type: str = 'default', optimizer_type: str = 'SGD',
        **kwargs: Any
    ) -> None:
        """
        Configure the MMEngine Runner with enhanced options for optimization and memory saving.

        Args:
            ... (previous arguments remain unchanged)
            optimizer_type (str): Type of optimizer to use (default: 'SGD')
            **kwargs: Additional keyword arguments for optimizer and strategy configuration
        """
        if not all([self.model, self.train_dataloader, self.val_dataloader, self.metric]):
            raise ValueError("Model, dataloaders, and metric must be set before configuring the runner.")

        optimizer_cfg = self.create_optimizer(optimizer_type, **kwargs.get('optimizer_kwargs', {}))
        optimizer_wrapper_cfg = {
            'type': 'AmpOptimWrapper' if use_amp else 'OptimWrapper',
            'optimizer': optimizer_cfg,
            'accumulative_counts': accumulative_counts
        }

        if strategy_type == 'deepspeed':
            optimizer_wrapper_cfg = {'type': 'DeepSpeedOptimWrapper', 'optimizer': optimizer_cfg}
        elif strategy_type == 'colossalai':
            optimizer_wrapper_cfg = {'optimizer': dict(type='HybridAdam', lr=1e-3)}

        strategy = self.create_strategy(strategy_type, **kwargs.get('strategy_kwargs', {}))

        runner_config = dict(
            model=self.model,
            work_dir=work_dir,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval),
            val_cfg=dict(),
            val_evaluator=dict(type=type(self.metric)),
            optim_wrapper=optimizer_wrapper_cfg,
            default_scope='mmengine',
            resume=resume,
            strategy=strategy
        )
        
    def train(self) -> None:
        if not self.runner:
            raise ValueError("Runner must be configured before starting training.")
        
        logger.info("Starting training process")
        try:
            self.runner.train()
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise

    def setup(
        self, work_dir: str, max_epochs: int, val_interval: int, 
        resume: bool = False, load_from: Optional[str] = None,
        distributed: bool = False, launcher: Optional[str] = None,
        use_amp: bool = False, accumulative_counts: int = 1,
        use_grad_checkpoint: bool = False, compile_model: bool = False,
        efficient_conv_bn_eval: bool = False,
        strategy_type: str = 'default', optimizer_type: str = 'SGD',
        **kwargs: Any
    ) -> None:
        """
        Set up the entire pipeline with enhanced options for optimization and memory saving.

        Args:
            work_dir (str): Working directory for saving checkpoints and logs
            max_epochs (int): Maximum number of epochs for training
            val_interval (int): Validation interval
            resume (bool): Whether to resume training from a checkpoint
            load_from (Optional[str]): Path to load model checkpoint
            distributed (bool): Whether to use distributed training
            launcher (Optional[str]): Launcher for distributed training
            use_amp (bool): Whether to use automatic mixed precision training
            accumulative_counts (int): Number of gradient accumulation steps
            use_grad_checkpoint (bool): Whether to use gradient checkpointing
            compile_model (bool): Whether to use torch.compile (PyTorch 2.0+)
            efficient_conv_bn_eval (bool): Whether to use the experimental Efficient Conv BN Eval feature
            strategy_type (str): Type of training strategy to use ('default', 'deepspeed', 'fsdp', 'colossalai')
            **strategy_kwargs: Additional keyword arguments for strategy configuration
        """
        logger.info(f"Setting up MMEngine pipeline with strategy: {strategy_type} and optimizer: {optimizer_type}")
        try:
            self.model = self.build_model()
            self.train_dataloader, self.val_dataloader = self.build_dataset()
            self.metric = self.build_metric()
            self.configure_runner(
                work_dir, max_epochs, val_interval, resume, load_from, distributed, launcher,
                use_amp, accumulative_counts, use_grad_checkpoint, compile_model, efficient_conv_bn_eval,
                strategy_type, optimizer_type, **kwargs
            )
            logger.info("Enhanced pipeline setup completed successfully")
        except Exception as e:
            logger.error(f"An error occurred during setup: {str(e)}")
            raise

    def run_experiment(
        self, work_dir: str, max_epochs: int, val_interval: int, 
        resume: bool = False, load_from: Optional[str] = None,
        distributed: bool = False, launcher: Optional[str] = None,
        use_amp: bool = False, accumulative_counts: int = 1,
        use_grad_checkpoint: bool = False, compile_model: bool = False,
        efficient_conv_bn_eval: bool = False,
        strategy_type: str = 'default', **strategy_kwargs: Any
    ) -> None:
        """
        Run a complete experiment with enhanced options for optimization and memory saving.

        Args:
            work_dir (str): Working directory for saving checkpoints and logs
            max_epochs (int): Maximum number of epochs for training
            val_interval (int): Validation interval
            resume (bool): Whether to resume training from a checkpoint
            load_from (Optional[str]): Path to load model checkpoint
            distributed (bool): Whether to use distributed training
            launcher (Optional[str]): Launcher for distributed training
            use_amp (bool): Whether to use automatic mixed precision training
            accumulative_counts (int): Number of gradient accumulation steps
            use_grad_checkpoint (bool): Whether to use gradient checkpointing
            compile_model (bool): Whether to use torch.compile (PyTorch 2.0+)
            efficient_conv_bn_eval (bool): Whether to use the experimental Efficient Conv BN Eval feature
            strategy_type (str): Type of training strategy to use ('default', 'deepspeed', 'fsdp', 'colossalai')
            **strategy_kwargs: Additional keyword arguments for strategy configuration
        """
        logger.info(f"Starting experiment with strategy: {strategy_type}")
        self.setup(
            work_dir, max_epochs, val_interval, resume, load_from, distributed, launcher,
            use_amp, accumulative_counts, use_grad_checkpoint, compile_model, efficient_conv_bn_eval,
            strategy_type, **strategy_kwargs
        )
        self.train()
        logger.info("Enhanced experiment completed")