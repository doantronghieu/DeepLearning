# MM

## MMengine

### Get Started

#### Introduction

#### ✅ Installation

#### ✅ 15 minutes to get started with MMEngine

### Examples

#### Train a GAN

#### Train a Segmentation Model

### Common Usage

#### ✅ Resume Training

#### ✅ Distributed Training

#### ✅ Speed up Training

#### ✅ Save Memory on GPU

#### ✅ Training Big Models

#### ✅ Better performance optimizers

#### ✅ Visualize Training Logs

#### ✅ Set Random Seed

#### ✅ Debug Tricks

#### ✅ Calculate the FLOPs and Parameters of Model

#### ✅ Setting the Frequency of Logging, Checkpoint Saving, and Validation

#### EpochBasedTraining to IterBasedTraining

### Tutorials

#### ✅ Runner

#### ✅ Dataset and DataLoader

#### ✅ Model

#### ✅ Evaluation

#### ✅temp OptimWrapper

##### OptimWrapper vs Optimizer

###### Model training

###### Get learning rate/momentum

###### Export/load state dicts

###### Use multiple optimizers

###### Configure the OptimWapper in Runner

##### Advanced usages

###### Set different hyperparamters for different types of parameters

###### Set different hyperparamters for different model modules

###### Customize optimizer construction policies

###### Adjust hyperparameters during training

#### Parameter Scheduler

##### Usage

###### Use a single LRScheduler

###### Combine multiple LRSchedulers (e.g. learning rate warm-up)

##### How to adjust other hyperparameters

###### Momentum

###### Generic parameter scheduler

#### Hook

##### Built-in Hooks

###### LoggerHook

###### CheckpointHook

###### ParamSchedulerHook

###### IterTimerHook

###### DistSamplerSeedHook

###### RuntimeInfoHook

###### EMAHook

###### EmptyCacheHook

###### SyncBuffersHook

###### ProfilerHook

##### Customize Your Hooks

## Advanced tutorials

### Registry

### Config

### BaseDataset

### Data transform

### Weight initialization

### Visualization

### Abstract Data Element

### Distribution Communication

### Logging

### File IO

### Global manager (ManagerMixin)

### Use modules from other libraries

### Test time augmentation

### Model Complexity Analysis

## Design

### Hook Design

### Runner Design

### Evaluation Design

### Visualization Design

### Logging Design

### Inference Design

## Migration guide

### Migrate Runner from MMCV to MMEngine

### Migrate Hook from MMCV to MMEngine

### Migrate Model from MMCV to MMEngine

### Migrate parameter scheduler from MMCV to MMEngine

### Migrate Data Transform to OpenMMLab 2.0

## API Reference

mmengine.analysis
ActivationAnalyzer
FlopAnalyzer
mmengine.analysis.activation_count
mmengine.analysis.flop_count
mmengine.analysis.parameter_count
mmengine.analysis.parameter_count_table
mmengine.analysis.get_model_complexity_info
mmengine.registry
Registry
DefaultScope
mmengine.registry.build_from_cfg
mmengine.registry.build_model_from_cfg
mmengine.registry.build_runner_from_cfg
mmengine.registry.build_scheduler_from_cfg
mmengine.registry.count_registered_modules
mmengine.registry.traverse_registry_tree
mmengine.registry.init_default_scope
mmengine.config
Config
ConfigDict
DictAction
read_base
mmengine.runner
Runner
Loop
Checkpoints
AMP
Miscellaneous
mmengine._strategy
BaseStrategy
SingleDeviceStrategy
DDPStrategy
DeepSpeedStrategy
FSDPStrategy
ColossalAIStrategy
MMDeepSpeedEngineWrapper
DeepSpeedOptimWrapper
ColossalAIOptimWrapper
mmengine.hooks
Hook
CheckpointHook
EMAHook
LoggerHook
NaiveVisualizationHook
ParamSchedulerHook
RuntimeInfoHook
DistSamplerSeedHook
IterTimerHook
SyncBuffersHook
EmptyCacheHook
ProfilerHook
NPUProfilerHook
PrepareTTAHook
EarlyStoppingHook
mmengine.model
Module
Model
EMA
Model Wrapper
Weight Initialization
Utils
mmengine.optim
Optimizer
Scheduler
mmengine.evaluator
Evaluator
Metric
Utils
mmengine.structures
BaseDataElement
InstanceData
LabelData
PixelData
mmengine.dataset
Dataset
Dataset Wrapper
Sampler
Utils
mmengine.infer
BaseInferencer
mmengine.device
mmengine.device.get_device
mmengine.device.get_max_cuda_memory
mmengine.device.is_cuda_available
mmengine.device.is_npu_available
mmengine.device.is_mlu_available
mmengine.device.is_mps_available
mmengine.hub
mmengine.hub.get_config
mmengine.hub.get_model
mmengine.logging
MMLogger
MessageHub
HistoryBuffer
mmengine.logging.print_log
mmengine.visualization
Visualizer
visualization Backend
mmengine.fileio
File Backend
File Handler
File IO
Parse File
mmengine.dist
dist
utils
mmengine.utils
Manager
Path
Package
Version
Progress Bar
Miscellaneous
mmengine.utils.dl_utils
TimeCounter
mmengine.utils.dl_utils.collect_env
mmengine.utils.dl_utils.load_url
mmengine.utils.dl_utils.has_batch_norm
mmengine.utils.dl_utils.is_norm
mmengine.utils.dl_utils.mmcv_full_available
mmengine.utils.dl_utils.tensor2imgs
mmengine.utils.dl_utils.TORCH_VERSION
mmengine.utils.dl_utils.set_multi_processing
mmengine.utils.dl_utils.torch_meshgrid
mmengine.utils.dl_utils.is_jit_tracing

## End
