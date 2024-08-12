# My code starts from here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, Union, List, Callable, Type
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator
import asyncio

from mmengine.registry import Registry, MODELS, DATASETS, METRICS, RUNNERS, HOOKS, OPTIMIZERS
from mmengine.model import BaseModel as MMBaseModel
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper
from mmengine.visualization import Visualizer
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed

# Create registries with proper scope, hierarchy, and locations
CUSTOM_MODELS = Registry('custom_model', parent=MODELS, scope='mmengine_example', locations=['models'])
CUSTOM_DATASETS = Registry('custom_dataset', parent=DATASETS, scope='mmengine_example', locations=['datasets'])
CUSTOM_METRICS = Registry('custom_metric', parent=METRICS, scope='mmengine_example', locations=['metrics'])
CUSTOM_HOOKS = Registry('custom_hook', parent=HOOKS, scope='mmengine_example', locations=['hooks'])
CUSTOM_FUNCTIONS = Registry('custom_function', scope='mmengine_example', locations=['functions'])


class ModelConfig(BaseModel):
    type: str
    init_args: Dict[str, Any] = Field(default_factory=dict)
    pretrained: Optional[str] = None
    backbone: Optional[Dict[str, Any]] = None
    neck: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    data_preprocessor: Optional[Dict[str, Any]] = None

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
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False

class OptimizerConfig(BaseModel):
    type: str
    lr: float = 0.01
    weight_decay: float = 0.0001

class LRSchedulerConfig(BaseModel):
    type: str
    by_epoch: bool = True
    warmup: Optional[Dict[str, Any]] = None

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

class MetricConfig(BaseModel):
    type: str
    init_args: Dict[str, Any] = Field(default_factory=dict)

class MMEngineConfig(BaseModel):
    project_name: str
    work_dir: str
    seed: int = 42
    
    model: ModelConfig
    dataset: Dict[str, DatasetConfig]
    dataloader: Dict[str, DataLoaderConfig]
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    runner: RunnerConfig
    
    hooks: List[HookConfig] = Field(default_factory=list)
    custom_hooks: List[HookConfig] = Field(default_factory=list)
    
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
        "metric": "accuracy"
    })
    
    visualizer: Optional[VisualizerConfig] = None
    metric: MetricConfig
    
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    
    cudnn_benchmark: bool = False
    mp_start_method: str = 'fork'
    
    dist_params: Dict[str, Any] = Field(default_factory=lambda: {
        "backend": "nccl"
    })
    
    log_level: str = 'INFO'
    log_processor: Dict[str, Any] = Field(default_factory=dict)
    
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
    
    vis_backends: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"type": "LocalVisBackend"}
    ])
    
    default_scope: str = 'mmengine'
    default_hooks: Dict[str, Any] = Field(default_factory=lambda: {
        "timer": {"type": "IterTimerHook"},
        "logger": {"type": "LoggerHook", "interval": 50},
        "param_scheduler": {"type": "ParamSchedulerHook"},
        "checkpoint": {"type": "CheckpointHook", "interval": 1},
        "sampler_seed": {"type": "DistSamplerSeedHook"},
        "visualization": {"type": "DetVisualizationHook"}
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
        self.dataset: Optional[Dict[str, Dataset]] = None
        self.metric: Optional[BaseMetric] = None
        self.runner: Optional[Runner] = None
        self.visualizer: Optional[Visualizer] = None

    def build_model(self) -> None:
        try:
            model_cfg = self.cfg.model.model_dump()
            if 'data_preprocessor' in model_cfg:
                data_preprocessor = MODELS.build(model_cfg['data_preprocessor'])
                model_cfg['data_preprocessor'] = data_preprocessor
            self.model = CUSTOM_MODELS.build(model_cfg)
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise
        
    def build_dataset(self) -> None:
        self.dataset = {split: CUSTOM_DATASETS.build(cfg.model_dump()) 
                        for split, cfg in self.cfg.dataset.items()}

    def build_metric(self) -> None:
        self.metric = CUSTOM_METRICS.build(self.cfg.metric.model_dump())
    
    def build_runner(self) -> None:
        runner_cfg = self.cfg.runner.model_dump()
        runner_cfg.update({
            'model': self.model,
            'train_dataloader': DataLoader(self.dataset['train'], **self.cfg.dataloader['train'].model_dump()),
            'val_dataloader': DataLoader(self.dataset['val'], **self.cfg.dataloader['val'].model_dump()),
            'val_evaluator': self.metric,
            'optim_wrapper': OptimWrapper(OPTIMIZERS.build(self.cfg.optimizer.model_dump())),
            'param_scheduler': HOOKS.build(self.cfg.lr_scheduler.model_dump()),
        })
        if self.cfg.hooks:
            runner_cfg['custom_hooks'] = [CUSTOM_HOOKS.build(hook.model_dump()) for hook in self.cfg.hooks]
        if self.cfg.visualizer:
            self.visualizer = Visualizer.from_config(self.cfg.visualizer.model_dump())
            runner_cfg['visualizer'] = self.visualizer
        self.runner: Runner = RUNNERS.build(runner_cfg)

    async def run(self) -> None:
        if self.runner is None:
            raise ValueError("Runner is not built. Call build_runner() first.")
        await self.runner.train()

@CUSTOM_MODELS.register_module()
class GenericModel(MMBaseModel):
    def __init__(self, backbone: nn.Module, neck: Optional[nn.Module] = None, head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.feature_flag = False  # Example of a feature flag

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None, mode: str = 'loss') -> Dict[str, Any]:
        x = self.backbone(inputs)
        if self.neck:
            x = self.neck(x)
        if self.head:
            x = self.head(x)
        
        if self.feature_flag:
            x = self.additional_processing(x)  # New feature
        
        if mode == 'loss':
            if labels is None:
                raise ValueError("Labels must be provided for loss computation")
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return {'predictions': x, 'labels': labels}
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def additional_processing(self, x):
        # New feature implementation
        return x * 2

@CUSTOM_DATASETS.register_module()
class GenericDataset(Dataset):
    def __init__(self, data_root: str, ann_file: Optional[str] = None, transform: Optional[Callable] = None):
        self.data_root = data_root
        self.ann_file = ann_file
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Implement data loading logic here
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self.data)

@CUSTOM_METRICS.register_module()
class GenericMetric(BaseMetric):
    def process(self, data_batch: Any, data_samples: Any) -> None:
        # Process a batch of data
        predictions = data_samples['predictions']
        labels = data_samples['labels']
        self.results.append({
            'pred': predictions,
            'label': labels
        })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # Compute and return the evaluation metrics
        predictions = torch.cat([res['pred'] for res in results])
        labels = torch.cat([res['label'] for res in results])
        accuracy = (predictions.argmax(dim=1) == labels).float().mean().item()
        return {'accuracy': accuracy}

@CUSTOM_HOOKS.register_module()
class GenericHook(Hook):
    def before_train(self, runner: Runner) -> None:
        logger.info("Starting training")

    def after_train(self, runner: Runner) -> None:
        logger.info("Training completed")

    def before_train_epoch(self, runner: Runner) -> None:
        logger.info(f"Starting epoch {runner.epoch}")

    def after_train_epoch(self, runner: Runner) -> None:
        logger.info(f"Epoch {runner.epoch} completed")

    def before_train_iter(self, runner: Runner) -> None:
        logger.info(f"Starting iteration {runner.iter}")

    def after_train_iter(self, runner: Runner) -> None:
        logger.info(f"Iteration {runner.iter} completed")

@CUSTOM_FUNCTIONS.register_module()
def custom_build_function(cfg: dict, registry: Registry, *args, **kwargs) -> Any:
    """Example of a custom build function"""
    return registry.get(cfg['type'])(*args, **kwargs, **cfg.get('init_args', {}))

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting MMEngine pipeline")
    
    # Convert Hydra config to a regular dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize MMEngine config
    config = MMEngineConfig(**cfg_dict)
    
    init_default_scope(config.default_scope)
    set_random_seed(config.seed)
    
    engine = BaseMMEngine(config)
    
    try:
        engine.build_model()
        engine.build_dataset()
        engine.build_metric()
        engine.build_runner()
        
        asyncio.run(engine.run())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        logger.info("MMEngine pipeline completed")

if __name__ == "__main__":
    main()