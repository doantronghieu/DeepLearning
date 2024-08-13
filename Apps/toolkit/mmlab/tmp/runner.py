import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, Literal, Optional, Union, List, Callable, Type, TypeVar, Sequence
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator
import asyncio
from functools import partial
import copy
import warnings
import os.path as osp

from mmengine.registry import Registry, MODELS, DATASETS, METRICS, RUNNERS, HOOKS, OPTIMIZERS, DATA_SAMPLERS
from mmengine.model import BaseModel as MMBaseModel
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper, AmpOptimWrapper, OptimWrapperDict
from mmengine.visualization import Visualizer
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed
from mmengine.dataset import BaseDataset, ConcatDataset, RepeatDataset, ClassBalancedDataset
from mmengine.dist import is_distributed
from mmengine.dataset import DefaultSampler, InfiniteSampler
from mmengine.logging import MMLogger
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# Type variables for generics
T = TypeVar('T')
D = TypeVar('D')

# Create registries with proper scope, hierarchy, and locations
CUSTOM_MODELS = Registry('custom_model', parent=MODELS, scope='mmengine_example', locations=['models'])
CUSTOM_DATASETS = Registry('custom_dataset', parent=DATASETS, scope='mmengine_example', locations=['datasets'])
CUSTOM_METRICS = Registry('custom_metric', parent=METRICS, scope='mmengine_example', locations=['metrics'])
CUSTOM_HOOKS = Registry('custom_hook', parent=HOOKS, scope='mmengine_example', locations=['hooks'])
CUSTOM_FUNCTIONS = Registry('custom_function', scope='mmengine_example', locations=['functions'])

class ParamSchedulerConfig(BaseModel):
    type: str
    by_epoch: bool = True
    begin: Optional[int] = None
    end: Optional[int] = None
    param_name: str = 'lr'
    init_args: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    type: str
    init_args: Dict[str, Any] = Field(default_factory=dict)
    pretrained: Optional[str] = None
    backbone: Optional[Dict[str, Any]] = None
    neck: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    data_preprocessor: Optional[Dict[str, Any]] = None
    feature_flags: Dict[str, bool] = Field(default_factory=dict)

    @field_validator('type')
    def check_model_type(cls, v):
        if v not in CUSTOM_MODELS.module_dict:
            raise ValueError(f"Model type {v} is not registered")
        return v

class DatasetConfig(BaseModel):
    type: str
    data_root: str
    ann_file: Optional[str] = None
    data_prefix: Optional[Dict[str, str]] = None
    pipeline: List[Dict[str, Any]]
    test_mode: bool = False

class DataLoaderConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 4
    sampler: Dict[str, Any] = Field(default_factory=lambda: {"type": "DefaultSampler", "shuffle": True})
    collate_fn: Dict[str, Any] = Field(default_factory=lambda: {"type": "default_collate"})
    pin_memory: bool = True
    persistent_workers: bool = True

class OptimizerConfig(BaseModel):
    type: str
    lr: float = 0.01
    weight_decay: float = 0.0001

class HookConfig(BaseModel):
    type: str
    priority: Optional[int] = None
    init_args: Dict[str, Any] = Field(default_factory=dict)

class RunnerConfig(BaseModel):
    type: str = 'EpochBasedRunner'
    max_epochs: int = 10
    val_interval: int = 1
    log_interval: int = 50

class VisualizerConfig(BaseModel):
    type: str = 'LocalVisBackend'
    save_dir: str = 'visual_results'
    vis_backends: List[Dict[str, Any]] = Field(default_factory=lambda: [{"type": "LocalVisBackend"}])

class MetricConfig(BaseModel):
    type: str
    init_args: Dict[str, Any] = Field(default_factory=dict)

class FeatureFlags(BaseModel):
    use_amp: bool = False
    use_gradient_accumulation: bool = False
    use_multi_optimizer: bool = False
    use_parameter_scheduling: bool = False
    use_custom_hooks: bool = False

class MMEngineConfig(BaseModel):
    project_name: str
    work_dir: str
    seed: int = 42
    
    dataset_wrappers: Optional[Dict[str, Dict[str, Any]]] = None
    custom_imports: Optional[List[str]] = None
    
    model: ModelConfig
    dataset: Dict[str, DatasetConfig]
    dataloader: Dict[str, DataLoaderConfig]
    
    optimizer: Union[OptimizerConfig, Dict[str, OptimizerConfig]]
    param_schedulers: List[ParamSchedulerConfig] = Field(default_factory=list)
    
    grad_accumulator: Optional[Dict[str, Any]] = None
    
    runner: RunnerConfig
    
    hooks: List[HookConfig] = Field(default_factory=list)
    custom_hooks: List[HookConfig] = Field(default_factory=list)
    
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    
    log_config: Dict[str, Any] = Field(default_factory=lambda: {
        "interval": 50,
        "hooks": [
            {"type": "TextLoggerHook"},
            {"type": "TensorboardLoggerHook"}
        ]
    })
    
    checkpoint_config: Dict[str, Any] = Field(default_factory=lambda: {
        "interval": 1
    })
    
    evaluation: Dict[str, Any] = Field(default_factory=lambda: {
        "interval": 1,
        "metrics": [{"type": "Accuracy"}]
    })
    
    visualizer: Optional[VisualizerConfig] = None
    metrics: List[MetricConfig] = Field(default_factory=list)
    
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    
    cudnn_benchmark: bool = False
    mp_start_method: str = 'fork'
    
    dist_params: Dict[str, Any] = Field(default_factory=lambda: {
        "backend": "nccl"
    })
    
    log_level: str = 'INFO'
    log_processor: Dict[str, Any] = Field(default_factory=dict)
    
    default_scope: str = 'mmengine'
    default_hooks: Dict[str, Any] = Field(default_factory=lambda: {
        "timer": {"type": "IterTimerHook"},
        "logger": {"type": "LoggerHook", "interval": 50},
        "param_scheduler": {"type": "ParamSchedulerHook"},
        "checkpoint": {"type": "CheckpointHook", "interval": 1},
        "sampler_seed": {"type": "DistSamplerSeedHook"},
        "visualization": {"type": "DetVisualizationHook"}
    })
    
    launcher: str = 'none'
    distributed: bool = False
    env_cfg: Dict[str, Any] = Field(default_factory=lambda: {
        "cudnn_benchmark": False,
        "mp_cfg": {
            "mp_start_method": "fork",
            "opencv_num_threads": 0
        },
        "dist_cfg": {
            "backend": "nccl"
        }
    })

    @field_validator('default_scope')
    def check_default_scope(cls, v):
        if v not in ['mmengine', 'mmengine_example']:
            raise ValueError(f"Invalid default_scope: {v}")
        return v

    def model_dump(self):
        return {k: v.model_dump() if isinstance(v, BaseModel) else v 
                for k, v in self.__dict__.items() if v is not None}

class BaseMMEngine:
    def __init__(self, cfg: MMEngineConfig) -> None:
        self.cfg = cfg
        self.model: Optional[MMBaseModel] = None
        self.dataset: Optional[Dict[str, BaseDataset]] = None
        self.metrics: Optional[List[BaseMetric]] = None
        self.runner: Optional[Runner] = None
        self.visualizer: Optional[Visualizer] = None
        self.logger = MMLogger.get_current_instance()

    async def lazy_init(self, build_func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(build_func, *args, **kwargs)
        
    async def build_model(self) -> None:
        try:
            model_cfg = self.cfg.model.model_dump()
            if 'data_preprocessor' in model_cfg:
                data_preprocessor = await self.lazy_init(MODELS.build, model_cfg['data_preprocessor'])
                model_cfg['data_preprocessor'] = data_preprocessor
            self.model = await self.lazy_init(CUSTOM_MODELS.build, model_cfg)
        except Exception as e:
            self.logger.error(f"Failed to build model: {str(e)}", exc_info=True)
            raise
        
    async def build_dataset(self) -> None:
        self.dataset = {}
        for split, cfg in self.cfg.dataset.items():
            dataset = await self.lazy_init(CUSTOM_DATASETS.build, cfg.model_dump())
            if self.cfg.dataset_wrappers:
                for wrapper_name, wrapper_cfg in self.cfg.dataset_wrappers.items():
                    if wrapper_name == 'ConcatDataset':
                        dataset = ConcatDataset([dataset] + [await self.lazy_init(CUSTOM_DATASETS.build, d) for d in wrapper_cfg['datasets']])
                    elif wrapper_name == 'RepeatDataset':
                        dataset = RepeatDataset(dataset, wrapper_cfg['times'])
                    elif wrapper_name == 'ClassBalancedDataset':
                        dataset = ClassBalancedDataset(dataset, wrapper_cfg['oversample_thr'])
            self.dataset[split] = dataset

    async def build_metrics(self) -> None:
        self.metrics = [await self.lazy_init(CUSTOM_METRICS.build, metric_cfg.model_dump()) for metric_cfg in self.cfg.metrics]
    
    async def _build_dataloader(self, split: str) -> DataLoader:
        dataset = self.dataset[split]
        dataloader_cfg = self.cfg.dataloader[split].model_dump()
        
        if is_distributed():
            if split == 'train' and self.cfg.runner.type == 'IterBasedRunner':
                sampler = InfiniteSampler(dataset, shuffle=True)
            else:
                sampler = await self.lazy_init(DATA_SAMPLERS.build, 
                                               dict(type='DistributedSampler', dataset=dataset, shuffle=(split == 'train')))
        else:
            if split == 'train' and self.cfg.runner.type == 'IterBasedRunner':
                sampler = InfiniteSampler(dataset, shuffle=True)
            else:
                sampler = DefaultSampler(dataset, shuffle=(split == 'train'))
        
        dataloader_cfg['sampler'] = sampler
        dataloader_cfg['collate_fn'] = await self.lazy_init(DATASETS.build, dataloader_cfg['collate_fn'])
        
        init_fn = CUSTOM_FUNCTIONS.get('init_fn')
        if init_fn:
            worker_init_fn = lambda worker_id: init_fn(worker_id, self.cfg.seed)
        else:
            worker_init_fn = None
        
        return DataLoader(dataset, **dataloader_cfg, worker_init_fn=worker_init_fn)

    async def build_optimizer(self) -> Union[OptimWrapper, OptimWrapperDict]:
        if self.cfg.feature_flags.use_multi_optimizer:
            optimizers = {}
            for name, optim_cfg in self.cfg.optimizer.items():
                optimizer = await self.lazy_init(OPTIMIZERS.build, optim_cfg.model_dump())
                if self.cfg.feature_flags.use_amp:
                    optimizers[name] = AmpOptimWrapper(optimizer=optimizer)
                else:
                    optimizers[name] = OptimWrapper(optimizer=optimizer)
            return OptimWrapperDict(**optimizers)
        else:
            optimizer = await self.lazy_init(OPTIMIZERS.build, self.cfg.optimizer.model_dump())
            if self.cfg.feature_flags.use_amp:
                return AmpOptimWrapper(optimizer=optimizer)
            else:
                return OptimWrapper(optimizer=optimizer)

    async def build_param_scheduler(self) -> List[Any]:
        if self.cfg.feature_flags.use_parameter_scheduling:
            schedulers = []
            for scheduler_cfg in self.cfg.param_schedulers:
                scheduler = await self.lazy_init(HOOKS.build, scheduler_cfg.model_dump())
                schedulers.append(scheduler)
            return schedulers
        else:
            return []

    async def build_runner(self) -> None:
        runner_cfg = self.cfg.runner.model_dump()
        runner_cfg.update({
            'model': self.model,
            'train_dataloader': await self._build_dataloader('train'),
            'val_dataloader': await self._build_dataloader('val'),
            'val_evaluator': self.metrics,
            'optim_wrapper': await self.build_optimizer(),
            'param_scheduler': await self.build_param_scheduler(),
        })
        if self.cfg.feature_flags.use_gradient_accumulation and self.cfg.grad_accumulator:
            runner_cfg['grad_accumulator'] = await self.lazy_init(HOOKS.build, self.cfg.grad_accumulator)
        if self.cfg.feature_flags.use_custom_hooks and self.cfg.hooks:
            runner_cfg['custom_hooks'] = [await self.lazy_init(CUSTOM_HOOKS.build, hook.model_dump()) for hook in self.cfg.hooks]
        if self.cfg.visualizer:
            self.visualizer = await self.lazy_init(Visualizer.from_config, self.cfg.visualizer.model_dump())
            runner_cfg['visualizer'] = self.visualizer
        self.runner: Runner = await self.lazy_init(RUNNERS.build, runner_cfg)

    async def run(self) -> None:
        if self.runner is None:
            raise ValueError("Runner is not built. Call build_runner() first.")
        await self.runner.train()

    async def offline_evaluate(self, data_file: str, prediction_file: str, chunk_size: int = 128) -> Dict[str, float]:
        """Perform offline evaluation using saved data and predictions."""
        evaluator = Evaluator(metrics=self.metrics)
        data = await self.lazy_init(load, data_file)
        data_samples = await self.lazy_init(load, prediction_file)
        results = await self.lazy_init(evaluator.offline_evaluate, data, data_samples, chunk_size=chunk_size)
        return results

@CUSTOM_MODELS.register_module()
class GenericModel(MMBaseModel):
    def __init__(self, components: Dict[str, nn.Module], forward_sequence: List[str]):
        super().__init__()
        self.components = nn.ModuleDict({name: MODELS.build(comp) for name, comp in components.items()})
        self.forward_sequence = forward_sequence

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None, mode: str = 'loss') -> Dict[str, Any]:
        x = inputs
        for component_name in self.forward_sequence:
            x = self.components[component_name](x)
        
        if mode == "loss":
            if 'labels' not in inputs:
                raise ValueError("Labels must be provided for loss computation")
            loss = self.compute_loss(x, inputs['labels'])
            return {'loss': loss}
        elif mode == 'predict':
            return {'predictions': x, 'labels': labels}
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # This method can be overridden by subclasses for custom loss computation
        return F.cross_entropy(predictions, labels)

@CUSTOM_DATASETS.register_module()
class GenericDataset(BaseDataset):
    METAINFO = {}  # Default meta information

    def __init__(
        self, 
        data_root: str, 
        ann_file: Optional[str] = None, 
        data_prefix: Optional[Dict[str, str]] = None,
        pipeline: List[Dict[str, Any]] = None, 
        test_mode: bool = False,
        lazy_init: bool = False,
        serialize_data: bool = True,
        data_loader: Optional[Callable] = None,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.data_prefix = data_prefix or {}
        self.test_mode = test_mode
        self.serialize_data = serialize_data
        self.data_loader = data_loader
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            serialize_data=serialize_data
        )

    def load_data_list(self) -> List[Dict[str, Any]]:
        """Load annotation file to get data information."""
        if self.data_loader is not None:
            return self.data_loader(self.ann_file)
        
        data_list = load(self.ann_file)
        if isinstance(data_list, dict):
            if 'metainfo' in data_list:
                self._metainfo.update(data_list['metainfo'])
            data_list = data_list.get('data_list', [])
        return [self.parse_data_info(data_info) for data_info in data_list]

    def parse_data_info(self, raw_data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw annotation to target format."""
        data_info = raw_data_info.copy()
        for key, prefix in self.data_prefix.items():
            if prefix is not None and key in data_info:
                data_info[key] = osp.join(prefix, data_info[key])
        return data_info

    def get_subset(self, indices: Union[Sequence[int], int]) -> 'GenericDataset':
        """Get a subset of the dataset."""
        if isinstance(indices, int):
            indices = list(range(indices))
        
        dataset = copy.deepcopy(self)
        dataset.data_list = [self.get_data_info(idx) for idx in indices]
        dataset.cumulative_sizes = None
        return dataset

    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        """Get a subset of the dataset in place."""
        if isinstance(indices, int):
            indices = list(range(indices))
        
        self.data_list = [self.get_data_info(idx) for idx in indices]
        self.cumulative_sizes = None

@CUSTOM_METRICS.register_module()
class GenericMetric(BaseMetric):
    def __init__(self, metric_fn: Callable, collect_device: str = 'cpu', prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric_fn = metric_fn

    def process(self, data_batch: Any, data_samples: Any) -> None:
        self.results.append(self.metric_fn(data_batch, data_samples))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if isinstance(results[0], dict):
            metrics = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
        else:
            metrics = {self.__class__.__name__: np.mean(results)}
        return metrics

@CUSTOM_HOOKS.register_module()
class GenericHook(Hook):
    def __init__(self, priority: int = 50):
        self._priority = priority

    @property
    def priority(self):
        return self._priority

    def before_run(self, runner: Runner) -> None:
        self.logger.info("Starting run")

    def after_run(self, runner: Runner) -> None:
        self.logger.info("Run completed")

    def before_train(self, runner: Runner) -> None:
        self.logger.info("Starting training")

    def after_train(self, runner: Runner) -> None:
        self.logger.info("Training completed")

    def before_train_epoch(self, runner: Runner) -> None:
        self.logger.info(f"Starting epoch {runner.epoch}")

    def after_train_epoch(self, runner: Runner) -> None:
        self.logger.info(f"Epoch {runner.epoch} completed")

    def before_train_iter(self, runner: Runner) -> None:
        if runner.iter % runner.log_interval == 0:
            self.logger.info(f"Starting iteration {runner.iter}")

    def after_train_iter(self, runner: Runner) -> None:
        if runner.iter % runner.log_interval == 0:
            self.logger.info(f"Iteration {runner.iter} completed")

class VisualizerWrapper(Visualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_vis_functions = {}

    def add_custom_vis_function(self, name: str, func: Callable):
        self.custom_vis_functions[name] = func

    def visualize(self, name: str, *args, **kwargs):
        if name in self.custom_vis_functions:
            fig = self.custom_vis_functions[name](*args, **kwargs)
            self.add_figure(name, fig)
        else:
            raise ValueError(f"Unsupported visualization: {name}")

@CUSTOM_FUNCTIONS.register_module()
def custom_build_function(cfg: dict, registry: Registry, *args, **kwargs) -> Any:
    """Example of a custom build function"""
    return registry.get(cfg['type'])(*args, **kwargs, **cfg.get('init_args', {}))

@CUSTOM_FUNCTIONS.register_module()
def init_fn(worker_id: int, seed: int) -> None:
    """Initialize worker for DataLoader"""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class VisualizerWrapper(Visualizer):
    """Wrapper class for Visualizer to add domain-specific visualization methods."""

    def visualize_predictions(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        """Visualize predictions and ground truth labels."""
        # Implement visualization logic here
        pass

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting MMEngine pipeline")
    
    # Convert Hydra config to a regular dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize MMEngine config
    config = MMEngineConfig(**cfg_dict)
    
    # Import custom modules
    if config.custom_imports:
        for module_path in config.custom_imports:
            try:
                __import__(module_path)
            except ImportError as e:
                logger.error(f"Failed to import {module_path}: {str(e)}")
    
    init_default_scope(config.default_scope)
    set_random_seed(config.seed)
    
    engine = BaseMMEngine(config)
    
    try:
        asyncio.run(engine.build_model())
        asyncio.run(engine.build_dataset())
        asyncio.run(engine.build_metrics())
        asyncio.run(engine.build_runner())
        
        if cfg.get('offline_evaluation'):
            results = asyncio.run(engine.offline_evaluate(cfg.data_file, cfg.prediction_file))
            logger.info(f"Offline evaluation results: {results}")
        else:
            asyncio.run(engine.run())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        logger.info("MMEngine pipeline completed")

if __name__ == "__main__":
    main()