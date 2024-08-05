import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type, List, Union
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import mmengine
from mmengine.runner import Runner, set_random_seed
from mmengine.analysis import get_model_complexity_info
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.dist import init_dist
from mmengine.config import Config
from mmengine.registry import RUNNERS, DATASETS, TRANSFORMS, FUNCTIONS, METRICS
from mmengine.dataset import DefaultSampler, pseudo_collate, default_collate, BaseDataset
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine.optim.scheduler import (
    MultiStepLR, LinearLR, ExponentialLR, CosineAnnealingLR, LinearParamScheduler
)
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

class DebugOptions(BaseModel):
    dataset_length: Optional[int] = Field(None, description="Set a fixed dataset length for debugging")
    num_batch_per_epoch: Optional[int] = Field(None, description="Set a fixed number of batches per epoch")
    find_unused_parameters: bool = Field(False, description="Enable unused parameter detection")
    detect_anomalous_params: bool = Field(False, description="Enable detection of anomalous parameters")

class OptimizerConfig(BaseModel):
    type: str = Field(..., description="Type of optimizer to use")
    lr: float = Field(..., description="Learning rate")
    weight_decay: Optional[float] = Field(None, description="Weight decay factor")
    paramwise_cfg: Optional[Dict[str, Any]] = Field(None, description="Parameter-wise learning rate configuration")
    constructor: Optional[str] = Field(None, description="Custom optimizer wrapper constructor")

class StrategyConfig(BaseModel):
    type: str = Field(..., description="Type of training strategy to use")
    fp16: Optional[Dict[str, Any]] = Field(None, description="FP16 configuration for DeepSpeed")
    zero_optimization: Optional[Dict[str, Any]] = Field(None, description="ZeRO optimization configuration for DeepSpeed")
    model_wrapper: Optional[Dict[str, Any]] = Field(None, description="Model wrapper configuration for FSDP")

class VisualizerConfig(BaseModel):
    wandb_init: Optional[Dict[str, Any]] = Field(None, description="Weights & Biases initialization parameters")
    neptune_init: Optional[Dict[str, Any]] = Field(None, description="Neptune initialization parameters")

class DatasetConfig(BaseModel):
    type: str = Field(..., description="Type of dataset to use")
    data_root: str = Field(..., description="Root directory of the dataset")
    ann_file: Optional[str] = Field(None, description="Path to annotation file")
    data_prefix: Optional[Dict[str, str]] = Field(None, description="Prefix for data files")
    pipeline: Optional[List[Dict[str, Any]]] = Field(None, description="Data processing pipeline")

class DataLoaderConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for dataloader")
    num_workers: int = Field(2, description="Number of worker processes for data loading")
    shuffle: bool = Field(True, description="Whether to shuffle the data")
    sampler: Optional[Dict[str, Any]] = Field(None, description="Sampler configuration")
    collate_fn: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Collate function configuration")

class EvaluatorConfig(BaseModel):
    type: str = Field(..., description="Type of evaluator to use")
    metrics: List[Dict[str, Any]] = Field(..., description="List of metrics to evaluate")

class ParamSchedulerConfig(BaseModel):
    type: str = Field(..., description="Type of parameter scheduler")
    by_epoch: bool = Field(True, description="Whether to update by epoch or iteration")
    begin: int = Field(0, description="Starting point of the scheduler")
    end: Optional[int] = Field(None, description="Ending point of the scheduler")
    convert_to_iter_based: bool = Field(False, description="Convert epoch-based config to iteration-based")
    param_name: Optional[str] = Field(None, description="Name of the parameter to schedule (for generic schedulers)")


class HookConfig(mmengine.model.BaseModel):
    type: str = Field(..., description="Type of hook to use")
    priority: str = Field("NORMAL", description="Priority of the hook")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional arguments for the hook")

class BaseMMengineConfig(BaseModel):
    work_dir: str = Field(..., description="Working directory for saving checkpoints and logs")
    max_epochs: int = Field(..., description="Maximum number of epochs for training")
    val_interval: int = Field(..., description="Validation interval")
    resume: bool = Field(False, description="Whether to resume training from a checkpoint")
    load_from: Optional[str] = Field(None, description="Path to load model checkpoint")
    distributed: bool = Field(False, description="Whether to use distributed training")
    launcher: Optional[str] = Field(None, description="Launcher for distributed training")
    use_amp: bool = Field(False, description="Whether to use automatic mixed precision training")
    accumulative_counts: int = Field(1, description="Number of gradient accumulation steps")
    use_grad_checkpoint: bool = Field(False, description="Whether to use gradient checkpointing")
    compile_model: bool = Field(False, description="Whether to use torch.compile (PyTorch 2.0+)")
    efficient_conv_bn_eval: bool = Field(False, description="Whether to use the experimental Efficient Conv BN Eval feature")
    strategy_type: str = Field("default", description="Type of training strategy to use")
    optimizer_type: str = Field("SGD", description="Type of optimizer to use")
    vis_backends: List[str] = Field([], description="List of visualization backends to use")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    diff_rank_seed: bool = Field(False, description="Whether to use different seeds for different ranks")
    deterministic: bool = Field(False, description="Whether to set deterministic mode for reproducibility")
    calculate_complexity: bool = Field(False, description="Whether to calculate model complexity")
    input_shape: Optional[Tuple[int, ...]] = Field(None, description="Input shape for model complexity calculation")
    
    debug_options: DebugOptions = Field(default_factory=DebugOptions, description="Debugging options")
    optimizer_kwargs: OptimizerConfig = Field(..., description="Optimizer configuration")
    strategy_kwargs: StrategyConfig = Field(..., description="Strategy configuration")
    visualizer_kwargs: VisualizerConfig = Field(default_factory=VisualizerConfig, description="Visualizer configuration")
    
    train_dataset: DatasetConfig = Field(..., description="Training dataset configuration")
    val_dataset: DatasetConfig = Field(..., description="Validation dataset configuration")
    train_dataloader: DataLoaderConfig = Field(..., description="Training dataloader configuration")
    val_dataloader: DataLoaderConfig = Field(..., description="Validation dataloader configuration")
    
    param_schedulers: Optional[List[ParamSchedulerConfig]] = Field(None, description="Parameter scheduler configurations")
    
    default_hooks: Dict[str, HookConfig] = Field(
        default_factory=lambda: {
            "runtime_info": HookConfig(type="RuntimeInfoHook", priority="VERY_HIGH"),
            "timer": HookConfig(type="IterTimerHook"),
            "sampler_seed": HookConfig(type="DistSamplerSeedHook"),
            "logger": HookConfig(type="LoggerHook", priority="BELOW_NORMAL"),
            "param_scheduler": HookConfig(type="ParamSchedulerHook", priority="LOW"),
            "checkpoint": HookConfig(type="CheckpointHook", priority="VERY_LOW"),
        },
        description="Default hooks configuration"
    )
    custom_hooks: List[HookConfig] = Field(default_factory=list, description="Custom hooks configuration")

    class Config:
        extra = "allow"  # Allows for additional fields not explicitly defined in the model

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
        self.data_preprocessor: Optional[torch.nn.Module] = None

    @abstractmethod
    def build_model(self) -> BaseModel:
        pass
    
    def build_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """Build dataset based on the provided configuration."""
        try:
            dataset_class = DATASETS.get(dataset_config.type)
            if issubclass(dataset_class, BaseDataset):
                return dataset_class(
                    data_root=dataset_config.data_root,
                    ann_file=dataset_config.ann_file,
                    data_prefix=dataset_config.data_prefix,
                    pipeline=dataset_config.pipeline
                )
            else:
                # Fallback for other dataset types (e.g., torchvision datasets)
                return dataset_class(root=dataset_config.data_root, **dataset_config.dict(exclude={'type', 'data_root'}))
        except Exception as e:
            logger.error(f"Failed to build dataset: {str(e)}")
            raise
    
    def build_dataloader(self, dataset: Dataset, dataloader_config: DataLoaderConfig) -> DataLoader:
        """Build dataloader based on the provided configuration."""
        try:
            sampler = self._build_sampler(dataset, dataloader_config.sampler)
            collate_fn = self._build_collate_fn(dataloader_config.collate_fn)
            
            return DataLoader(
                dataset,
                batch_size=dataloader_config.batch_size,
                num_workers=dataloader_config.num_workers,
                sampler=sampler,
                collate_fn=collate_fn
            )
        except Exception as e:
            logger.error(f"Failed to build dataloader: {str(e)}")
            raise
    
    def _build_sampler(self, dataset: Dataset, sampler_config: Optional[Dict[str, Any]]) -> Optional[torch.utils.data.Sampler]:
        """Build sampler based on the provided configuration."""
        if sampler_config is None:
            return DefaultSampler(dataset, shuffle=True)
        
        sampler_type = sampler_config.pop('type', 'DefaultSampler')
        return RUNNERS.build(dict(type=sampler_type, dataset=dataset, **sampler_config))

    def _build_collate_fn(self, collate_config: Optional[Union[str, Dict[str, Any]]]) -> callable:
        """Build collate function based on the provided configuration."""
        if collate_config is None:
            return pseudo_collate
        
        if isinstance(collate_config, str):
            if collate_config == 'default_collate':
                return default_collate
            elif collate_config == 'pseudo_collate':
                return pseudo_collate
        
        return FUNCTIONS.build(collate_config)

    def create_strategy(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """
        Factory method to create a strategy configuration based on the specified type.
        
        Args:
            strategy_type (str): The type of strategy to create ('default', 'deepspeed', 'fsdp', 'colossalai')
            **kwargs: Additional keyword arguments for strategy configuration
        
        Returns:
            Dict[str, Any]: The strategy configuration
        """
        strategies = {
            'default': lambda: {},
            'deepspeed': lambda: {
                'type': 'DeepSpeedStrategy',
                'fp16': kwargs.get('fp16', {'enabled': True}),
                'zero_optimization': kwargs.get('zero_optimization', {'stage': 3})
            },
            'fsdp': lambda: {
                'type': 'FSDPStrategy',
                'model_wrapper': kwargs.get('model_wrapper', {})
            },
            'colossalai': lambda: {'type': 'ColossalAIStrategy'}
        }
        
        strategy_func = strategies.get(strategy_type)
        if not strategy_func:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        
        return strategy_func()

    def _configure_optimizer(self, optimizer_type: str, use_amp: bool, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """Helper method to configure the optimizer wrapper."""
        optimizer_cfg = self.create_optimizer(optimizer_type, **kwargs.get('optimizer_kwargs', {}))
        
        wrapper_cfg = {
            'optimizer': optimizer_cfg,
            'accumulative_counts': kwargs.get('accumulative_counts', 1)
        }
        
        if kwargs.get('clip_grad'):
            wrapper_cfg['clip_grad'] = kwargs['clip_grad']
        
        if strategy_type == 'deepspeed':
            return {'type': 'DeepSpeedOptimWrapper', **wrapper_cfg}
        elif strategy_type == 'colossalai':
            return {'optimizer': dict(type='HybridAdam', lr=1e-3)}
        else:
            wrapper_type = 'AmpOptimWrapper' if use_amp else 'OptimWrapper'
            return {'type': wrapper_type, **wrapper_cfg}

    def create_optimizer(self, optimizer_type: str, **kwargs) -> Dict[str, Any]:
        """
        Factory method to create an optimizer configuration based on the specified type.
        
        Args:
            optimizer_type (str): The type of optimizer to create
            **kwargs: Additional keyword arguments for optimizer configuration
        
        Returns:
            Dict[str, Any]: The optimizer configuration
        """
        optimizers = {
            'SGD': dict(type='SGD', lr=0.01, momentum=0.9),
            'Adam': dict(type='Adam', lr=0.001, betas=(0.9, 0.999)),
            'AdamW': dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
            'DAdaptAdaGrad': dict(type='DAdaptAdaGrad', lr=0.01),
            'DAdaptAdam': dict(type='DAdaptAdam', lr=0.01),
            'DAdaptSGD': dict(type='DAdaptSGD', lr=0.01),
            'Lion': dict(type='Lion', lr=1e-4, weight_decay=1e-2),
            'Sophia': dict(type='Sophia', lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1),
            'SophiaG': dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1),
            'AdamW8bit': dict(type='AdamW8bit', lr=1e-4, weight_decay=1e-2),
            'Adafactor': dict(type='Adafactor', lr=1e-5, weight_decay=1e-2, scale_parameter=False, relative_step=False),
        }
        
        if optimizer_type not in optimizers:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        optimizer_config = optimizers[optimizer_type].copy()
        optimizer_config.update(kwargs)
        
        # Add support for parameter-wise learning rates
        if 'paramwise_cfg' in kwargs:
            optimizer_config['paramwise_cfg'] = kwargs['paramwise_cfg']
        
        return optimizer_config

    def _configure_model_wrapper(self, debug_options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to configure the model wrapper."""
        return {
            'type': 'MMDistributedDataParallel',
            'find_unused_parameters': debug_options.get('find_unused_parameters', False),
            'detect_anomalous_params': debug_options.get('detect_anomalous_params', False)
        }
    
    def create_visualizer(self, vis_backends: List[str], **kwargs) -> Dict[str, Any]:
        """
        Factory method to create a visualizer configuration based on the specified backends.
        
        Args:
            vis_backends (List[str]): List of visualization backends to use
            **kwargs: Additional keyword arguments for visualizer configuration
        
        Returns:
            Dict[str, Any]: The visualizer configuration
        """
        vis_backends_config = []
        backend_configs = {
            'TensorBoard': lambda: dict(type='TensorboardVisBackend'),
            'WandB': lambda: dict(type='WandbVisBackend', init_kwargs=kwargs.get('wandb_init', {})),
            'ClearML': lambda: dict(type='ClearMLVisBackend'),
            'Neptune': lambda: dict(type='NeptuneVisBackend', init_kwargs=kwargs.get('neptune_init', {})),
            'DVCLive': lambda: dict(type='DVCLiveVisBackend'),
            'Aim': lambda: dict(type='AimVisBackend')
        }

        for backend in vis_backends:
            if backend in backend_configs:
                vis_backends_config.append(backend_configs[backend]())
            else:
                logger.warning(f"Unsupported visualization backend: {backend}")

        return dict(type='Visualizer', vis_backends=vis_backends_config)
    
    @abstractmethod
    def build_metric(self) -> List[BaseMetric]:
        pass
    
    def build_evaluator(self, config: EvaluatorConfig) -> Evaluator:
        """Build evaluator based on the provided configuration."""
        metrics = []
        for metric_config in config.metrics:
            metric_type = metric_config.pop('type')
            metric_class = METRICS.get(metric_type)
            metrics.append(metric_class(**metric_config))
        
        return Evaluator(metrics)
    
    def build_param_scheduler(self, scheduler_config: ParamSchedulerConfig) -> Any:
        """Build parameter scheduler based on the provided configuration."""
        scheduler_cls = self._get_scheduler_class(scheduler_config.type)
        scheduler_kwargs = scheduler_config.dict(exclude={'type', 'param_name'})
        
        if scheduler_config.param_name:
            scheduler_kwargs['param_name'] = scheduler_config.param_name
        
        return scheduler_cls(**scheduler_kwargs)

    def _get_scheduler_class(self, scheduler_type: str) -> Type:
        """Get the scheduler class based on the scheduler type."""
        scheduler_map = {
            'MultiStepLR': MultiStepLR,
            'LinearLR': LinearLR,
            'ExponentialLR': ExponentialLR,
            'CosineAnnealingLR': CosineAnnealingLR,
            'LinearParamScheduler': LinearParamScheduler,
        }
        
        if scheduler_type not in scheduler_map:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler_map[scheduler_type]
    
    def _configure_hooks(self, config: BaseMMengineConfig) -> Dict[str, List[Dict[str, Any]]]:
        """Configure default and custom hooks."""
        default_hooks = []
        for hook_name, hook_config in config.default_hooks.items():
            hook_cfg = {"type": hook_config.type, "priority": hook_config.priority, **hook_config.kwargs}
            default_hooks.append(hook_cfg)

        custom_hooks = []
        for hook_config in config.custom_hooks:
            hook_cfg = {"type": hook_config.type, "priority": hook_config.priority, **hook_config.kwargs}
            custom_hooks.append(hook_cfg)

        return {"default_hooks": default_hooks, "custom_hooks": custom_hooks}
    
    def configure_runner(self, config: Dict[str, Any]) -> None:
        """
        Configure the MMEngine Runner with enhanced options for optimization and visualization.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all necessary parameters
        """
        if not all([self.model, self.train_dataloader, self.val_dataloader, self.data_preprocessor]):
            raise ValueError("Model, dataloaders, and data preprocessor must be set before configuring the runner.")

        optimizer_wrapper_cfg = self._configure_optimizer(
            config['optimizer_type'], config['use_amp'], config['strategy_type'], **config
        )

        # Add support for custom optimizer wrapper constructor
        if config.get('optimizer_kwargs', {}).get('constructor'):
            optimizer_wrapper_cfg['constructor'] = config['optimizer_kwargs']['constructor']

        strategy = self.create_strategy(config['strategy_type'], **config.get('strategy_kwargs', {}))
        visualizer = self.create_visualizer(config['vis_backends'], **config)

        randomness = {
            'seed': config.get('seed'),
            'diff_rank_seed': config.get('diff_rank_seed', False),
            'deterministic': config.get('deterministic', False)
        }

        # Apply debugging options to dataloaders
        for dataloader in [self.train_dataloader, self.val_dataloader]:
            if config['debug_options'].get('dataset_length'):
                dataloader.dataset.indices = config['debug_options']['dataset_length']

        train_dataloader_cfg = dict(self.train_dataloader.cfg)
        val_dataloader_cfg = dict(self.val_dataloader.cfg)

        if config['debug_options'].get('num_batch_per_epoch'):
            train_dataloader_cfg['num_batch_per_epoch'] = config['debug_options']['num_batch_per_epoch']
            val_dataloader_cfg['num_batch_per_epoch'] = config['debug_options']['num_batch_per_epoch']

        model_wrapper_cfg = self._configure_model_wrapper(config['debug_options'])

        evaluator = self.build_evaluator(config['evaluator'])

        param_schedulers = None
        if config.get('param_schedulers'):
            param_schedulers = [self.build_param_scheduler(sc) for sc in config['param_schedulers']]
        
        hooks_config = self._configure_hooks(config)
        
        runner_config = {
            'model': self.model,
            'work_dir': config['work_dir'],
            'train_dataloader': train_dataloader_cfg,
            'val_dataloader': val_dataloader_cfg,
            'train_cfg': dict(by_epoch=True, max_epochs=config['max_epochs'], val_interval=config['val_interval']),
            'val_cfg': dict(),
            'val_evaluator': evaluator,
            'optim_wrapper': optimizer_wrapper_cfg,
            'default_scope': 'mmengine',
            'resume': config.get('resume', False),
            'strategy': strategy,
            'visualizer': visualizer,
            'randomness': randomness,
            'model_wrapper_cfg': model_wrapper_cfg,
            'data_preprocessor': self.data_preprocessor,
            'param_scheduler': param_schedulers,
            "default_hooks": hooks_config["default_hooks"],
            "custom_hooks": hooks_config["custom_hooks"],
        }
        
        self.runner = Runner(**runner_config)
        
        if config.get('seed') is not None:
            logger.info(f"Setting random seed to {config['seed']}, "
                        f"deterministic: {config['deterministic']}, "
                        f"diff_rank_seed: {config['diff_rank_seed']}")
            set_random_seed(config['seed'], deterministic=config['deterministic'], diff_rank_seed=config['diff_rank_seed'])
    
    def train(self) -> None:
        """Start the training process."""
        if not self.runner:
            raise ValueError("Runner must be configured before starting training.")
        
        logger.info("Starting training process")
        try:
            self.runner.train()
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise
    
    def calculate_model_complexity(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Calculate the FLOPs and parameters of the model.

        Args:
            input_shape (Tuple[int, ...]): The input shape for the model.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results.
        """
        if not self.model:
            raise ValueError("Model must be set before calculating complexity.")

        logger.info(f"Calculating model complexity for input shape: {input_shape}")
        try:
            analysis_results = get_model_complexity_info(self.model, input_shape)
            
            logger.info("Model complexity calculation completed")
            logger.info(f"Model FLOPs: {analysis_results['flops_str']}")
            logger.info(f"Model Parameters: {analysis_results['params_str']}")
            
            return analysis_results
        except Exception as e:
            logger.error(f"An error occurred during model complexity calculation: {str(e)}")
            raise
    
    def setup(self, config: BaseMMengineConfig) -> None:
        """Set up the entire pipeline with enhanced options for optimization and memory saving."""
        logger.info(f"Setting up MMEngine pipeline with strategy: {config.strategy_type}, "
                    f"optimizer: {config.optimizer_type}, and visualization backends: {config.vis_backends}")
        try:
            self.model = self.build_model()
            self.data_preprocessor = self.build_data_preprocessor()
            
            train_dataset = self.build_dataset(config.train_dataset)
            val_dataset = self.build_dataset(config.val_dataset)
            
            self.train_dataloader = self.build_dataloader(train_dataset, config.train_dataloader)
            self.val_dataloader = self.build_dataloader(val_dataset, config.val_dataloader)

            if config.calculate_complexity:
                if not config.input_shape:
                    raise ValueError("input_shape must be provided when calculate_complexity is True")
                complexity_results = self.calculate_model_complexity(config.input_shape)
                logger.info(f"Model complexity results:\n{complexity_results['out_table']}")

            if config.param_schedulers:
                logger.info(f"Configuring {len(config.param_schedulers)} parameter scheduler(s)")
                for scheduler_config in config.param_schedulers:
                    logger.info(f"  - {scheduler_config.type} for {scheduler_config.param_name or 'learning rate'}")

            self.configure_runner(config.dict())
            logger.info("Enhanced pipeline setup completed successfully")
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during setup: {str(e)}")
            raise

    def run_experiment(self, config: BaseMMengineConfig) -> None:
        """
        Run a complete experiment with enhanced options for optimization and memory saving.

        Args:
            config (BaseMMengineConfig): Configuration object containing all necessary parameters
        """
        logger.info(f"Starting experiment with strategy: {config.strategy_type} "
                    f"and visualization backends: {config.vis_backends}")
        self.setup(config)
        self.train()
        logger.info("Enhanced experiment completed")

