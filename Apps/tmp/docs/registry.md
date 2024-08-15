# Registry

## mmengine.registry

### Registry

A registry to map strings to classes or functions.

#### Attributes

- **name** (str): Registry name.
- **build_func** (callable, optional): Function to construct instance from Registry. `build_from_cfg()` is used if neither parent nor build_func is specified. Defaults to None.
- **parent** (Registry, optional): Parent registry. The class registered in children registry could be built from parent. Defaults to None.
- **scope** (str, optional): The scope of the registry, used as the key to search for children registries. Defaults to None.
- **locations** (list): Locations to import the modules registered in this registry. Defaults to [].

#### Examples

```python
# Define a registry
MODELS = Registry('models')

# Register the `ResNet` class to `MODELS`
@MODELS.register_module()
class ResNet:
    pass

# Build model from `MODELS`
resnet = MODELS.build(dict(type='ResNet'))

# Register a function
@MODELS.register_module()
def resnet50():
    pass

resnet = MODELS.build(dict(type='resnet50'))

# Hierarchical registry
DETECTORS = Registry('detectors', parent=MODELS, scope='det')
@DETECTORS.register_module()
class FasterRCNN:
    pass

fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

# Add locations to enable auto-import
DETECTORS = Registry('detectors', parent=MODELS, scope='det', locations=['det.models.detectors'])
@DETECTORS.register_module()
class MaskRCNN:
    pass

# The registry will auto-import det.models.detectors.MaskRCNN
fasterrcnn = DETECTORS.build(dict(type='det.MaskRCNN'))
```

More advanced usages can be found at [MMEngine Registry Documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html).

#### Methods

- **build(cfg, *args, **kwargs)**: Build an instance by calling `build_func` with the provided config dictionary.

- **get(key)**: Retrieve the registry record by key, which may include scope information.

- **import_from_location()**: Import modules from pre-defined locations.

- **infer_scope()**: Infer the scope of the registry based on its package name.

- **register_module(name=None, force=False, module=None)**: Register a module or class. Can be used as a decorator or a normal function.

- **split_scope_key(key)**: Split the scope and key from the given key.

- **switch_scope_and_registry(scope)**: Temporarily switch the default scope to the target scope and retrieve the corresponding registry.

### DefaultScope

The scope of the current task used to reset the current registry, which can be accessed globally.

#### Attributes

- **name** (str): Name of the default scope for global access.
- **scope_name** (str): Scope of the current task.

#### Examples

```python
from mmengine.model import MODELS

# Define default scope in the runner
DefaultScope.get_instance('task', scope_name='mmdet')

# Get the default scope globally
scope_name = DefaultScope.get_instance('task').scope_name
```

#### Methods

- **get_current_instance()**: Get the latest created default scope instance. Returns None if no DefaultScope instance exists.

- **overwrite_default_scope(scope_name)**: Overwrite the current default scope with a new scope name.

- **scope_name**: Get the current scope name.

### mmengine.registry.build_from_cfg

Build a module from a configuration dictionary when it is a class configuration or call a function when it is a function configuration.

#### Parameters

- **cfg** (dict): Configuration dictionary, which should include the key “type”.
- **registry** (Registry): The registry to search for the type.
- **default_args** (dict, optional): Default initialization arguments.

#### Examples

```python
from mmengine import Registry, build_from_cfg

MODELS = Registry('models')

@MODELS.register_module()
class ResNet:
    def __init__(self, depth, stages=4):
        self.depth = depth
        self.stages = stages

cfg = dict(type='ResNet', depth=50)
model = build_from_cfg(cfg, MODELS)

@MODELS.register_module()
def resnet50():
    pass

resnet = build_from_cfg(dict(type='resnet50'), MODELS)
```

### mmengine.registry.build_model_from_cfg

Build a PyTorch model from a configuration dictionary or a list of dictionaries.

#### Parameters

- **cfg** (dict, list of dicts): Configuration of modules, either a single dict or a list of dicts.
- **registry** (Registry): Registry for the module.
- **default_args** (dict, optional): Default arguments for building the module.

#### Returns

- A built `nn.Module`.

### mmengine.registry.build_runner_from_cfg

Build a Runner object from a configuration dictionary.

#### Parameters

- **cfg** (dict): Configuration dictionary. If it includes "runner_type", it will build a custom runner; otherwise, it builds a default runner.
- **registry** (Registry): The registry to search for the type.

#### Examples

```python
from mmengine.registry import Registry, build_runner_from_cfg

RUNNERS = Registry('runners', build_func=build_runner_from_cfg)

@RUNNERS.register_module()
class CustomRunner(Runner):
    def setup_env(env_cfg):
        pass

cfg = dict(runner_type='CustomRunner', ...)
custom_runner = RUNNERS.build(cfg)
```

### mmengine.registry.build_scheduler_from_cfg

Build a `ParamScheduler` instance from a configuration dictionary.

#### Parameters

- **cfg** (dict): Configuration dictionary, which may contain the key `convert_to_iter_based`.
- **registry** (Registry): The PARAM_SCHEDULERS registry.
- **default_args** (dict, optional): Default initialization arguments, including key `optimizer`.

### mmengine.registry.count_registered_modules

Scan all modules in MMEngine’s root and child registries and dump to JSON.

#### Parameters

- **save_path** (str, optional): Path to save the JSON file.
- **verbose** (bool): Whether to print log messages.

#### Returns

- A dictionary of statistics about all registered modules.

### mmengine.registry.traverse_registry_tree

Traverse the entire registry tree from a given node and collect information about all registered modules.

#### Parameters

- **registry** (Registry): A registry node in the registry tree.
- **verbose** (bool): Whether to print log messages.

#### Returns

- A list of statistics about all modules in each node of the registry tree.

### mmengine.registry.init_default_scope

Initialize the given default scope.

#### Parameters

- **scope** (str): Name of the default scope.

---

## Registry

OpenMMLab supports a rich collection of algorithms and datasets, resulting in many modules with similar functionalities. For instance, ResNet and SE-ResNet are implemented based on the `ResNet` and `SEResNet` classes, respectively. These classes have similar functions and interfaces and are part of the model components in the algorithm library. To manage these functionally similar modules, MMEngine implements a registry. Most algorithm libraries in OpenMMLab, including MMDetection, MMDetection3D, MMPretrain, and MMagic, use this registry to manage their modules.

## What is a Registry

The registry in MMEngine functions as a combination of a mapping table and a module build function. The mapping table maintains a link from strings to classes or functions, allowing users to find the corresponding class or function by its name. For example, it maps the string "ResNet" to the `ResNet` class. The module build function defines how to locate the corresponding class or function based on a string and how to instantiate it or call it. For instance, it can find `nn.BatchNorm2d` and instantiate it using the string "bn," or locate the `build_batchnorm2d` function and return its result. By default, registries in MMEngine use the `build_from_cfg` function to find and instantiate the class or function corresponding to a string.

Classes or functions managed by a registry typically have similar interfaces and functionalities, making the registry an abstraction of those classes or functions. For example, the `MODELS` registry can be seen as an abstraction managing classes such as `ResNet`, `SEResNet`, and `RegNetX`, along with constructors like `build_ResNet`, `build_SEResNet`, and `build_RegNetX`.

## Getting Started

To use the registry for managing modules in your codebase, follow these steps:

1. **Create a Registry**: Define a registry for your modules.
2. **Create a Build Method** (optional): Implement a custom method for instantiating the class, though the default method is often sufficient.
3. **Add the Module to the Registry**: Register your module with the registry.

For example, to implement and switch between different activation modules without modifying the code, you would first create a registry:

```python
from mmengine import Registry

# `scope` represents the domain of the registry. If not set, the default value is the package name.
# e.g. in mmdetection, the scope is mmdet
# `locations` indicates the location where the modules in this registry are defined.
# The Registry will automatically import the modules when building them according to these predefined locations.
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])
```

The `mmengine.models.activations` module corresponds to the `mmengine/models/activations.py` file. Modules implemented in this file, such as `Sigmoid`, `ReLU`, and `Softmax`, will be automatically imported when building with the registry.

```python
import torch.nn as nn

# Register the modules
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x

@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x
```

With the modules registered, you can create mappings between strings and classes or functions via `ACTIVATION`.

```python
print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

### Note

To trigger the registry mechanism, ensure that the module is imported. Modules can be registered in three ways:

1. Implement the module in predefined locations.
2. Import the file manually.
3. Use the `custom_imports` field in the configuration.

Once registered, the activation module can be used in the configuration file:

```python
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)

# Switching to ReLU
act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call ReLU.forward
print(output)
```

For custom build processes, you can define a `build_activation` function and pass it to the registry:

```python
def build_activation(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    print(f'build activation: {act_type}')
    act_cls = registry.get(act_type)
    act = act_cls(*args, **kwargs, **cfg_)
    return act

ACTIVATION = Registry('activation', build_func=build_activation, scope='mmengine', locations=['mmengine.models.activations'])

@ACTIVATION.register_module()
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Tanh.forward')
        return x

act_cfg = dict(type='Tanh')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# build activation: Tanh
# call Tanh.forward
print(output)
```

### Note

The above example demonstrates customizing the instance-building method using `build_func`, which is similar to the default `build_from_cfg` method. In most cases, the default method suffices.

## Advanced Usage

MMEngine’s registry supports hierarchical registration, enabling cross-project calls. MMEngine provides twenty-two root registries for various components:

- `RUNNERS`: Registry for Runner.
- `RUNNER_CONSTRUCTORS`: Constructors for Runner.
- `LOOPS`: Manages training, validation, and testing processes.
- `HOOKS`: Hooks like `CheckpointHook` and `ParamSchedulerHook`.
- `DATASETS`: Datasets.
- `DATA_SAMPLERS`: Sampler of DataLoader.
- `TRANSFORMS`: Data preprocessing methods.
- `MODELS`: Model modules.
- `MODEL_WRAPPERS`: Model wrappers for distributed data parallelism.
- `WEIGHT_INITIALIZERS`: Tools for weight initialization.
- `OPTIMIZERS`: Registers all Optimizers and custom Optimizers.
- `OPTIM_WRAPPER`: Wrapper for Optimizer-related operations.
- `OPTIM_WRAPPER_CONSTRUCTORS`: Constructors for optimizer wrappers.
- `PARAM_SCHEDULERS`: Parameter schedulers like `MultiStepLR`.
- `METRICS`: Evaluation metrics for model accuracy.
- `EVALUATOR`: Evaluation metrics used for model accuracy.
- `TASK_UTILS`: Task-intensive components.
- `VISUALIZERS`: Module for drawing prediction boxes.
- `VISBACKENDS`: Backend for storing training logs.
- `LOG_PROCESSORS`: Controls log statistics methods.
- `FUNCTIONS`: Registers various functions.
- `INFERENCERS`: Registers inferencers for different tasks.

### Use the Module of the Parent Node

Define a `RReLU` module in MMEngine and register it to the `MODELS` root registry:

```python
import torch.nn as nn
from mmengine import Registry, MODELS

@MODELS.register_module()
class RReLU(nn.Module):
    def __init__(self, lower=0.125, upper=0.333, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call RReLU.forward')
        return x
```

In a project called MMAlpha, define a `MODELS` registry with `MMEngine` as its parent:

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models'])
```

To use modules from `MMEngine` in `MMAlpha`:

```python
@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x
```

Use `LogSoftmax` in `MMAlpha`:

```python
model = MODELS.build(cfg=dict(type='LogSoftmax'))
```

To use modules from the parent node:

```python
model = MODELS.build(cfg=dict(type='RReLU', lower=0.2))
# or specify scope explicitly
model = MODELS.build(cfg=dict(type='mmengine.RReLU'))
```

If no prefix is added, the build method will first search the current node and then the parent nodes or ancestors. To avoid ambiguity, specify the scope prefix when needed.

```python
import torch

input = torch.randn(2)
output = model(input)
# call RReLU.forward
print(output)
```

### How the Parent Node Knows About Child Registries

When using modules from a child registry, the parent node needs to switch to the appropriate scope as defined in `default_runtime.py` for class searching. You can also initialize the scope accordingly:

```python
from mmalpha.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import init_default_scope
import torch.nn as nn

@MODELS.register_module()
class LogSoft

max(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x

# Works because we are using mmalpha registry
MODELS.build(dict(type="LogSoftmax"))

# Fails because mmengine registry does not know about modules registered in mmalpha
MMENGINE_MODELS.build(dict(type="LogSoftmax"))

# Works because we are using mmalpha registry
init_default_scope('mmalpha')
MMENGINE_MODELS.build(dict(type="LogSoftmax"))
```

### Use the Module of a Sibling Node

To call a module from a sibling node, such as `MMBeta`, which also defines `MODELS` and sets its parent node to `MMEngine`:

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmbeta')
```

To use modules from `MMAlpha` in `MMBeta`:

```python
model = MODELS.build(cfg=dict(type='mmalpha.LogSoftmax'))
output = model(input)
# call LogSoftmax.forward
print(output)
```

When calling modules from sibling nodes, specify the scope prefix. For multiple modules, this can be cumbersome, so `DefaultScope` allows for temporary switching:

```python
model = MODELS.build(cfg=dict(type='LogSoftmax', _scope_='mmalpha'))
output = model(input)
# call LogSoftmax.forward
print(output)
```