# Documents

## Introduction

Introduction
MMEngine is a foundational library for training deep learning models based on PyTorch.

Module Introduction
MMEngine abstracts the components involved in the training process and their relationships. Components of the same type in different algorithm libraries share the same interface definition.

Core Modules and Related Components
The core module of the training engine is the Runner. The Runner is responsible for executing training, testing, and inference tasks and managing the various components required during these processes. In specific locations throughout the execution of training, testing, and inference tasks, the Runner sets up Hooks to allow users to extend, insert, and execute custom logic. The Runner primarily invokes the following components to complete the training and inference loops:

Dataset: Responsible for constructing datasets in training, testing, and inference tasks, and feeding the data to the model. In usage, it is wrapped by a PyTorch DataLoader, which launches multiple subprocesses to load the data.

Model: Accepts data and outputs the loss during the training process; accepts data and performs predictions during testing and inference tasks. In a distributed environment, the model is wrapped by a Model Wrapper (e.g., MMDistributedDataParallel).

Optimizer Wrapper: The optimizer wrapper performs backpropagation to optimize the model during the training process and supports mixed-precision training and gradient accumulation through a unified interface.

Parameter Scheduler: Dynamically adjusts optimizer hyperparameters such as learning rate and momentum during the training process.

During training intervals or testing phases, the Metrics & Evaluator are responsible for evaluating the performance of the model. The Evaluator evaluates the model’s predictions based on the dataset. Within the Evaluator, there is an abstraction called Metrics, which calculates various metrics such as recall, accuracy, and others.

To ensure a unified interface, the communication interfaces between the evaluators, models, and data in various algorithm libraries within OpenMMLab 2.0 are encapsulated using Data Elements.

During training and inference execution, the aforementioned components can utilize the logging management module and visualizer for structured and unstructured logging storage and visualization. Logging Modules: Responsible for managing various log information generated during the execution of the Runner. The Message Hub implements data sharing between components, runners, and log processors, while the Log Processor processes the log information. The processed logs are then sent to the Logger and Visualizer for management and display. The Visualizer is responsible for visualizing the model’s feature maps, prediction results, and structured logs generated during the training process. It supports multiple visualization backends such as TensorBoard and WanDB.

Common Base Modules
MMEngine also implements various common base modules required during the execution of algorithmic models, including:

Config: In the OpenMMLab algorithm library, users can configure the training, testing process, and related components by writing a configuration file (config).

Registry: Responsible for managing modules within the algorithm library that have similar functionality. Based on the abstraction of algorithm library modules, MMEngine defines a set of root registries. Registries within the algorithm library can inherit from these root registries, enabling cross-algorithm library module invocations and interactions. This allows for seamless integration and utilization of modules across different algorithms within the OpenMMLab framework.

File I/O: Provides a unified interface for file read/write operations in various modules, supporting multiple file backend systems and formats in a consistent manner, with extensibility.

Distributed Communication Primitives: Handles communication between different processes during distributed program execution. This interface abstracts the differences between distributed and non-distributed environments and automatically handles data devices and communication backends.

Other Utilities: There are also utility modules, such as ManagerMixin, which implements a way to create and access global variables. The base class for many globally accessible objects within the Runner is ManagerMixin.

## 15 minutes to get started with MMEngine

In this tutorial, we’ll take training a ResNet-50 model on CIFAR-10 dataset as an example. We will build a complete and configurable pipeline for both training and validation in only 80 lines of code with MMEngine. The whole process includes the following steps:

15 minutes to get started with MMEngine

Build a Model

Build a Dataset and DataLoader

Build a Evaluation Metrics

Build a Runner and Run the Task

Build a Model
First, we need to build a model. In MMEngine, the model should inherit from BaseModel. Aside from parameters representing inputs from the dataset, its forward method needs to accept an extra argument called mode:

for training, the value of mode is “loss,” and the forward method should return a dict containing the key “loss”.

for validation, the value of mode is “predict”, and the forward method should return results containing both predictions and labels.

```python

import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

Build a Dataset and DataLoader
Next, we need to create Dataset and DataLoader for training and validation. For basic training and validation, we can simply use built-in datasets supported in TorchVision.

```python

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

Build a Evaluation Metrics
To validate and test the model, we need to define a Metric called accuracy to evaluate the model. This metric needs inherit from BaseMetric and implements the process and compute_metrics methods where the process method accepts the output of the dataset and other outputs when mode="predict". The output data at this scenario is a batch of data. After processing this batch of data, we save the information to self.results property. compute_metrics accepts a results parameter. The input results of compute_metrics is all the information saved in process (In the case of a distributed environment, results are the information collected from all process in all the processes). Use these information to calculate and return a dict that holds the results of the evaluation metrics

```python

from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy=100 * total_correct / total_size)
```

Build a Runner and Run the Task
Now we can build a Runner with previously defined Model, DataLoader, and Metrics, and some other configs shown as follows:

```python

from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    # the model used for training and validation.
    # Needs to meet specific interface requirements
    model=MMResNet50(),
    # working directory which saves training logs and weight files
    work_dir='./work_dir',
    # train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # optimize wrapper for optimization with additional features like
    # AMP, gradtient accumulation, etc
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # trainging coinfs for specifying training epoches, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # validation dataloaer also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=Accuracy),
)

runner.train()
```

Finally, let’s put all the codes above together into a complete script that uses the MMEngine executor for training and validation:

```python

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

## Registry

OpenMMLab supports a rich collection of algorithms and datasets, therefore, many modules with similar functionality are implemented. For example, the implementations of ResNet and SE-ResNet are based on the classes ResNet and SEResNet, respectively, which have similar functions and interfaces and belong to the model components of the algorithm library. To manage these functionally similar modules, MMEngine implements the registry. Most of the algorithm libraries in OpenMMLab use registry to manage their modules, including MMDetection, MMDetection3D, MMPretrain and MMagic, etc.

What is a registry
The registry in MMEngine can be considered as a union of a mapping table and a build function of modules. The mapping table maintains a mapping from strings to classes or functions, allowing the user to find the corresponding class or function with its name/notation. For example, the mapping from the string "ResNet" to the ResNet class. The module build function defines how to find the corresponding class or function based on a string and how to instantiate the class or call the function. For example, finding nn.BatchNorm2d and instantiating the BatchNorm2d module by the string "bn", or finding the build_batchnorm2d function by the string "build_batchnorm2d" and then returning the result. The registries in MMEngine use the build_from_cfg function by default to find and instantiate the class or function corresponding to the string.

The classes or functions managed by a registry usually have similar interfaces and functionality, so the registry can be treated as an abstraction of those classes or functions. For example, the registry MODELS can be treated as an abstraction of all models, which manages classes such as ResNet, SEResNet and RegNetX and constructors such as build_ResNet, build_SEResNet and build_RegNetX.

Getting started
There are three steps required to use the registry to manage modules in the codebase.

Create a registry.

Create a build method for instantiating the class (optional because in most cases you can just use the default method).

Add the module to the registry

Suppose we want to implement a series of activation modules and want to be able to switch to different modules by just modifying the configuration without modifying the code.

Let’s create a registry first.

``` python
from mmengine import Registry
# `scope` represents the domain of the registry. If not set, the default value is the package name.
# e.g. in mmdetection, the scope is mmdet
# `locations` indicates the location where the modules in this registry are defined.
# The Registry will automatically import the modules when building them according to these predefined locations.
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])
```

The module mmengine.models.activations specified by locations corresponds to the mmengine/models/activations.py file. When building modules with registry, the ACTIVATION registry will automatically import implemented modules from this file. Therefore, we can implement different activation layers in the mmengine/models/activations.py file, such as Sigmoid, ReLU, and Softmax.

```python

import torch.nn as nn

# use the register_module
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

The key of using the registry module is to register the implemented modules into the ACTIVATION registry. With the @ACTIVATION.register_module() decorator added before the implemented module, the mapping between strings and classes or functions can be built and maintained by ACTIVATION. We can achieve the same functionality with ACTIVATION.register_module(module=ReLU) as well.

By registering, we can create a mapping between strings and classes or functions via ACTIVATION.

```python
print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

Note

The key to trigger the registry mechanism is to make the module imported. There are three ways to register a module into the registry

Implement the module in the locations. The registry will automatically import modules in the predefined locations. This is to ease the usage of algorithm libraries so that users can directly use REGISTRY.build(cfg).

Import the file manually. This is common when developers implement a new module in/out side the algorithm library.

Use custom_imports field in config. Please refer to Importing custom Python modules for more details.

Once the implemented module is successfully registered, we can use the activation module in the configuration file.

```python

import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)

# call Sigmoid.forward

print(output)
We can switch to ReLU by just changing this configuration.

act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)

# call ReLU.forward

print(output)
```

If we want to check the type of input parameters (or any other operations) before creating an instance, we can implement a build method and pass it to the registry to implement a custom build process.

Create a build_activation function.

```python

def build_activation(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    print(f'build activation: {act_type}')
    act_cls = registry.get(act_type)
    act = act_cls(*args,**kwargs, **cfg_)
    return act
Pass the buid_activation to build_func.

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

Note

In the above example, we demonstrate how to customize the method of building an instance of a class using the build_func. This is similar to the default build_from_cfg method. In most cases, using the default method will be fine.

MMEngine’s registry can register classes as well as functions.

```python
FUNCTION = Registry('function', scope='mmengine')

@FUNCTION.register_module()
def print_args(**kwargs):
    print(kwargs)

func_cfg = dict(type='print_args', a=1, b=2)
func_res = FUNCTION.build(func_cfg)
```

Advanced usage
The registry in MMEngine supports hierarchical registration, which enables cross-project calls, meaning that modules from one project can be used in another project. Though there are other ways to implement this, the registry provides a much easier solution.

To easily make cross-library calls, MMEngine provides twenty two root registries, including:

RUNNERS: the registry for Runner.

RUNNER_CONSTRUCTORS: the constructors for Runner.

LOOPS: manages training, validation and testing processes, such as EpochBasedTrainLoop.

HOOKS: the hooks, such as CheckpointHook, and ParamSchedulerHook.

DATASETS: the datasets.

DATA_SAMPLERS: Sampler of DataLoader, used to sample the data.

TRANSFORMS: various data preprocessing methods, such as Resize, and Reshape.

MODELS: various modules of the model.

MODEL_WRAPPERS: model wrappers for parallelizing distributed data, such as MMDistributedDataParallel.

WEIGHT_INITIALIZERS: the tools for weight initialization.

OPTIMIZERS: registers all Optimizers and custom Optimizers in PyTorch.

OPTIM_WRAPPER: the wrapper for Optimizer-related operations such as OptimWrapper, and AmpOptimWrapper.

OPTIM_WRAPPER_CONSTRUCTORS: the constructors for optimizer wrappers.

PARAM_SCHEDULERS: various parameter schedulers, such as MultiStepLR.

METRICS: the evaluation metrics for computing model accuracy, such as Accuracy.

EVALUATOR: one or more evaluation metrics used to calculate the model accuracy.

TASK_UTILS: the task-intensive components, such as AnchorGenerator, and BboxCoder.

VISUALIZERS: the management drawing module that draws prediction boxes on images, such as DetVisualizer.

VISBACKENDS: the backend for storing training logs, such as LocalVisBackend, and TensorboardVisBackend.

LOG_PROCESSORS: controls the log statistics window and statistics methods, by default we use LogProcessor. You may customize LogProcessor if you have special needs.

FUNCTIONS: registers various functions, such as collate_fn in DataLoader.

INFERENCERS: registers inferencers of different tasks, such as DetInferencer, which is used to perform inference on the detection task.

Use the module of the parent node
Let’s define a RReLU module in MMEngine and register it to the MODELS root registry.

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

Now suppose there is a project called MMAlpha, which also defines a MODELS and sets its parent node to the MODELS of MMEngine, which creates a hierarchical structure.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models'])
```

The following figure shows the hierarchy of MMEngine and MMAlpha.

The count_registered_modules function can be used to print the modules that have been registered to MMEngine and their hierarchy.

```python
from mmengine.registry import count_registered_modules

count_registered_modules()
We define a customized LogSoftmax module in MMAlpha and register it to the MODELS in MMAlpha.

@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x
```

Here we use the LogSoftmax in the configuration of MMAlpha.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax'))
We can also use the modules of the parent node MMEngine here in the MMAlpha.

model = MODELS.build(cfg=dict(type='RReLU', lower=0.2))

# scope is optional

model = MODELS.build(cfg=dict(type='mmengine.RReLU'))
```

If no prefix is added, the build method will first find out if the module exists in the current node and return it if there is one. Otherwise, it will continue to look up the parent nodes or even the ancestor node until it finds the module. If the same module exists in both the current node and the parent nodes, we need to specify the scope prefix to indicate that we want to use the module of the parent nodes.

```python
import torch

input = torch.randn(2)
output = model(input)

# call RReLU.forward

print(output)
```

How does the parent node know about child registry?
When working in our MMAlpha it might be necessary to use the Runner class defined in MMENGINE. This class is in charge of building most of the objects. If these objects are added to the child registry (MMAlpha), how is MMEngine able to find them? It cannot, MMEngine needs to switch to the Registry from MMEngine to MMAlpha according to the scope which is defined in default_runtime.py for searching the target class.

We can also init the scope accordingly, see example below:

```python
from mmalpha.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import init_default_scope
import torch.nn as nn

@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x

# Works because we are using mmalpha registry

MODELS.build(dict(type="LogSoftmax"))

# Fails because mmengine registry does not know about stuff registered in mmalpha

MMENGINE_MODELS.build(dict(type="LogSoftmax"))

# Works because we are using mmalpha registry

init_default_scope('mmalpha')
MMENGINE_MODELS.build(dict(type="LogSoftmax"))
```

Use the module of a sibling node
In addition to using the module of the parent nodes, users can also call the module of a sibling node.

Suppose there is another project called MMBeta, which, like MMAlpha, defines MODELS and set its parent node to MMEngine.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmbeta')
```

The following figure shows the registry structure of MMAlpha and MMBeta.

Now we call the modules of MMAlpha in MMBeta.

```python
model = MODELS.build(cfg=dict(type='mmalpha.LogSoftmax'))
output = model(input)

# call LogSoftmax.forward

print(output)
```

Calling a module of a sibling node requires the scope prefix to be specified in type, so the above configuration requires the prefix mmalpha.

However, if you need to call several modules of a sibling node, each with a prefix, this requires a lot of modification. Therefore, MMEngine introduces the DefaultScope, with which Registry can easily support temporary switching of the current node to the specified node.

If you need to switch the current node to the specified node temporarily, just set _scope_ to the scope of the specified node in cfg.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax', _scope_='mmalpha'))
output = model(input)

# call LogSoftmax.forward

print(output)
```

## Runner

Runner
Welcome to the tutorial of runner, the core of MMEngine’s user interface!

The runner, as an “integrator” in MMEngine, covers all aspects of the framework and shoulders the responsibility of organizing and scheduling nearly all modules. Therefore, the code logic in it has to take into account various situations, making it relatively hard to understand. But don’t worry! In this tutorial, we will leave out some messy details and have a quick overview of commonly used APIs, functionalities, and examples. Hopefully, this should provide you with a clear and easy-to-understand user interface. After reading through this tutorial, you will be able to:

Master the common usage and configuration of the runner

Learn the best practice - writing config files - of the runner

Know about the basic dataflow and execution order

Feel by yourself the advantages of using runner (perhaps)

Example codes of the runner
To build your training pipeline with a runner, there are typically two ways to get started:

Refer to runner’s API documentation for argument-by-argument configuration

Make your custom modifications based on some existing configurations, such as Getting started in 15 minutes and downstream repositories like MMDet

Pros and cons lie in both approaches. For the former one, beginners may be lost in a vast number of configurable arguments. For the latter one, beginners may find it hard to get a good reference, since neither an over-simplified nor an over-detailed reference is conducive to them.

We argue that the key to learning runner is using it as a memo. You should remember its most commonly used arguments and only focus on those less used when in need, since default values usually work fine. In the following, we will provide a beginner-friendly example to illustrate the most commonly used arguments of the runner, along with advanced guidelines for those less used.

A beginner-friendly example
Hint

In this tutorial, we hope you can focus more on overall architecture instead of implementation details. This “top-down” way of thinking is exactly what we advocate. Don’t worry, you will definitely have plenty of opportunities and guidance afterward to focus on modules you want to improve.

Before running the actual example below, you should first run this piece of code for the preparation of the model, dataset, and metric. However, these implementations are not important in this tutorial and you can simply look through

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS


@MODELS.register_module()
class MyAwesomeModel(BaseModel):
    def __init__(self, layers=4, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            act_type = nn.ReLU
        elif activation == 'silu':
            act_type = nn.SiLU
        elif activation == 'none':
            act_type = nn.Identity
        else:
            raise NotImplementedError
        sequence = [nn.Linear(2, 64), act_type()]
        for _ in range(layers-1):
            sequence.extend([nn.Linear(64, 64), act_type()])
        self.mlp = nn.Sequential(*sequence)
        self.classifier = nn.Linear(64, 2)

    def forward(self, data, labels, mode):
        x = self.mlp(data)
        x = self.classifier(x)
        if mode == 'tensor':
            return x
        elif mode == 'predict':
            return F.softmax(x, dim=1), labels
        elif mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}


@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, is_train, size):
        self.is_train = is_train
        if self.is_train:
            torch.manual_seed(0)
            self.labels = torch.randint(0, 2, (size,))
        else:
            torch.manual_seed(3407)
            self.labels = torch.randint(0, 2, (size,))
        r = 3 * (self.labels+1) + torch.randn(self.labels.shape)
        theta = torch.rand(self.labels.shape) * 2 * torch.pi
        self.data = torch.vstack([r*torch.cos(theta), r*torch.sin(theta)]).T

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)
```

Click to show a long example. Be well prepared

```python
from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner


runner = Runner(
    # your model
    model=MyAwesomeModel(
        layers=2,
        activation='relu'),
    # work directory for saving checkpoints and logs
    work_dir='exp/my_awesome_model',

    # training data
    train_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=True,
            size=10000),
        shuffle=True,
        collate_fn=default_collate,
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    # training configurations
    train_cfg=dict(
        by_epoch=True,   # display in epoch number instead of iterations
        max_epochs=10,
        val_begin=2,     # start validation from the 2nd epoch
        val_interval=1), # do validation every 1 epoch

    # OptimizerWrapper, new concept in MMEngine for richer optimization options
    # Default value works fine for most cases. You may check our documentations
    # for more details, e.g. 'AmpOptimWrapper' for enabling mixed precision
    # training.
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001)),
    # ParamScheduler to adjust learning rates or momentums during training
    param_scheduler=dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1),

    # validation data
    val_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=False,
            size=1000),
        shuffle=False,
        collate_fn=default_collate,
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    # validation configurations, usually leave it an empty dict
    val_cfg=dict(),
    # evaluation metrics and evaluator
    val_evaluator=dict(type=Accuracy),

    # following are advanced configurations, try to default when not in need
    # hooks are advanced usage, try to default when not in need
    default_hooks=dict(
        # the most commonly used hook for modifying checkpoint saving interval
        checkpoint=dict(type='CheckpointHook', interval=1)),

    # `luancher` and `env_cfg` responsible for distributed environment
    launcher='none',
    env_cfg=dict(
        cudnn_benchmark=False,   # whether enable cudnn_benchmark
        backend='nccl',   # distributed communication backend
        mp_cfg=dict(mp_start_method='fork')),  # multiprocessing configs
    log_level='INFO',

    # load model weights from given path. None for no loading.
    load_from=None,
    # resume training from the given path
    resume=False
)

# start training your model
runner.train()
```

Explanations on example codes
Really a long piece of code, isn’t it! However, if you read through the above example, you may have already understood the training process in general even without knowing any implementation details, thanks to the compactness and readability of runner codes (probably). This is what MMEngine expects: a structured, modular, and standardized training process that allows for more reliable reproductions and clearer comparisons.

The above example may lead you to the following confusion:

There are too many arguments!
Don’t worry. As we mentioned before, use runner as a memo. The runner covers all aspects just to ensure you won’t miss something important. You don’t actually need to configure everything. The simple example in 15 minutes still works fine, and it can be even more simplified by removing val_evaluator, val_dataloader, and val_cfg without breaking down. All configurable arguments are driven by your demands. Those not in your focus usually work fine by default.

Why are some arguments passed as dicts?
Well, this is related to MMEngine’s style. In MMEngine, we provide 2 different styles of runner construction: a) manual construction and b) construction via registry. If you are confused, the following example will give a good illustration:

```python
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.registry import MODELS # root registry for your custom model

@MODELS.register_module() # decorator for registration
class MyAwesomeModel(BaseModel): # your custom model
    def __init__(self, layers=18, activation='silu'):
        ...

# An example of construction via registry
runner = Runner(
    model=dict(
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    ...
)

# An example of manual construction
model = MyAwesomeModel(layers=18, activation='relu')
runner = Runner(
    model=model,
    ...
)
```

Similar to the above example, most arguments in the runner accept both 2 types of inputs. They are conceptually equivalent. The difference is, in the former style, the module (passed in as a dict) will be built in the runner when actually needed, while in the latter style, the module has been built before being passed to the runner. The following figure illustrates the core idea of registry: it maintains the mapping between a module’s build method and its registry name. If you want to learn more about the full usage of the registry, you are recommended to read Registry tutorial.

Runner Registry Illustration

You might still be confused after the explanation. Why should we let the Runner build modules from dicts? What are the benefits? If you have such questions, then we are proud to answer: “Absolutely - no benefits!” In fact, module construction via registry only works to its best advantage when combined with a configuration file. It is still far from the best practice to write as the above example. We provide it here just to make sure you can read and get used to this writing style, which may facilitate your understanding of the actual best practice we will soon talk about - the configuration file. Stay tuned!

If you as a beginner do not immediately understand, it doesn’t matter too much, because manual construction is still a good choice, especially for small-scale development and trial-and-error due to its being IDE friendly. However, you are still expected to read and get used to the writing style via registry, so that you can avoid being unnecessarily confused and puzzled in subsequent tutorials.

Where can I find the possible configuration options for the xxx argument?
You will find extensive instructions and examples in those tutorials of the corresponding modules. You can also find all possible arguments in Runner’s API documentation. If neither of the above resolves your query, you are always encouraged to start a topic in our discussion forum. It also helps us improve documentation.

I come from repositoried like MMDet/MMCls... Why does this example differ from what I've been exposed to?
Downstream repositories in OpenMMLab have widely adopted the writing style of config files. In the following chapter, we will show the usage of config files, the best practice of the runner in MMEngine, based on the above example with a slight variation.

Best practice of the Runner - config files
MMEngine provides a powerful config file system that supports Python syntax. You can almost seamlessly (which we will illustrate below) convert from the previous sample code to a config file. Here is an example:

```python
# Save the following codes in example_config.py
# Almost copied from the above example, with some commas removed
model = dict(type='MyAwesomeModel',
    layers=2,
    activation='relu')
work_dir = 'exp/my_awesome_model'

train_dataloader = dict(
    dataset=dict(type='MyDataset',
        is_train=True,
        size=10000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_begin=2,
    val_interval=1)
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[4, 8],
    gamma=0.1)

val_dataloader = dict(
    dataset=dict(type='MyDataset',
        is_train=False,
        size=1000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1000,
    pin_memory=True,
    num_workers=2)
val_cfg = dict()
val_evaluator = dict(type='Accuracy')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False
```

Given the above config file, we can simply load configurations and run the training pipeline in a few lines of codes as follows:

```python
from mmengine.config import Config
from mmengine.runner import Runner
config = Config.fromfile('example_config.py')
runner = Runner.from_cfg(config)
runner.train()
```

Note

Although it supports Python syntax, a valid config file needs to meet the condition that all variables must be Python built-in types such as str, dict and int. Therefore, the config system is highly dependent on the registry mechanism to enable construction from built-in types to other types such as nn.Module.

Note

When using config files, you typically don’t need to manually register every module. For instance, all optimizers in torch.optim including Adam and SGD have already been registered in mmengine.optim. The rule of thumb is, try to directly access modules provided by PyTorch, and only start to register them manually after error occurs.

Note

When using config files, the implementations of your custom modules may be stored in separate files and thus not registered properly, which will lead to errors in the build process. You may find solutions in Config tutorial.

Warning

Although sharing nearly the same codes, from_cfg and __init__ differs in some default values like env_cfg.

Writing config files of the runner has been widely adopted in downstream repositories in OpenMMLab projects. It has been a de facto convention and best practice. The config files are far more featured than illustrated above. You can refer to Config tutorial for more advanced features including keywords inheriting and overriding.

Basic dataflow
Hint

In this chapter, we’ll dive deeper into the runner to illustrate dataflow and data format convention between modules managed by the runner. It may be relatively abstract and dry if you haven’t built a training pipeline with MMEngine. Therefore, you are free to skip for now and read it in conjunction with practice in the future when in need.

Now let’s dive slightly deeper into the runner, and illustrate the dataflow and data format convention under the hood (or, under the engine)!

Basic Dataflow

The diagram above illustrates the basic dataflow of the runner, where the dashed border, gray filled shapes represent different data formats, while solid boxes represent modules/methods. Due to the great flexibility and extensibility of MMEngine, you can always inherit some key base classes and override their methods, so the above diagram doesn’t always hold. It only holds when you are not customizing your own Runner or TrainLoop, and you are not overriding train_step, val_step or test_step method in your custom model. Actually, this is common for most tasks like detection and segmentation, as referred to Model tutorial.

Can you state the exact type of each data item shown in the diagram?
Unfortunately, this is not possible. Although we did heavy type annotations in MMEngine, Python is still a highly dynamic programming language, and deep learning as a data-centric system needs to be flexible enough to deal with a wide range of complex data sources. You always have full freedom to decide when you need (and sometimes must) break type conventions. Therefore, when you are customizing your module (e.g. val_evaluator), you need to make sure its input is compatible with upstream (e.g. model) output and its output can be parsed by downstream. MMEngine puts the flexibility of handling data in the hands of the user, and thus also requires the user to ensure compatibility of dataflow, which, in fact, is not that difficult once you get started.

The uniformity of data formats has always been a problem in deep learning. We are trying to improve it in MMEngine in our own way. If you are interested, you can refer to BaseDataset and BaseDataElement - but please note that they are mainly geared towards advanced users.

What's the data format convention between dataloader, model and evaluator?
For the basic dataflow shown in the diagram above, the data transfer between the above three modules can be represented by the following pseudo-code:

```python
# training
for data_batch in train_dataloader:
    data_batch = data_preprocessor(data_batch)
    if isinstance(data_batch, dict):
        losses = model.forward(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model.forward(*data_batch, mode='loss')
    else:
        raise TypeError()

# validation
for data_batch in val_dataloader:
    data_batch = data_preprocessor(data_batch)
    if isinstance(data_batch, dict):
        outputs = model.forward(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model.forward(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

The key points of the above pseudo-code is:

Outputs of data_preprocessor are passed to model after unpacking

The data_samples argument of the evaluator receives the prediction results of the model, while the data_batch argument receives the raw data coming from dataloader

What is data_preprocessor? Can I do image pre-processing such as crop and resize in it?
Though drawn separately in the diagram, data_preprocessor is a part of the model and thus can be found in Model tutorial in DataPreprocessor chapter.

In most cases, data_preprocessor needs no special attention or manual configuration. The default data_preprocessor will only do data transfer between host and GPU devices. However, if your model has incompatible inputs format with dataloader’s output, you can also customize you own data_preprocessor for data formatting.

Image pre-processing such as crop and resize is more recommended in data transforms module, but for batch-related data transforms (e.g. batch-resize), it can be implemented here.

Why does module produce 3 different outputs? What is the meaning of "loss", "predict" and "tensor"?
As described in get started in 15 minutes, you need to implement 3 data paths in your custom model’s forward function to suit different pipelines for training, validation and testing. This is further discussed in Model tutorial.

I can see that the red line is for training process and the blue line for validation/testing, but what is the green line?
Currently model outputs in “tensor” mode has not been officially used in runner. The “tensor” mode can output some intermediate results and thus facilitating debugging process.

What if I override methods such as train_step? Will the diagram totally fail?
The behavior of default train_step, val_step and test_step covers the dataflow from data_preprocessor to model outputs and optim_wrapper. The rest of the diagram will not be spoiled.

Why use the runner? (Optional reading)
Hint

Contents in this chapter will not teach you how to use the runner and MMEngine. If you are being pushed by your employer/advisor/DDL to work out a result in a few hours, it may not help you and you can feel free to skip it. However, we highly recommend taking time to read through this chapter, since it will help you better understand the aim and style of MMEngine.

Relax, time for some philosophy
Congratulations for reading through the runner tutorial, a long, long but kind of interesting (hope so) tutorial! Please believe that all of these - this tutorial, the runner, MMEngine - are intended to make things easier for you.

The runner is the “manager” of all modules in MMEngine. In the runner, all the distinct modules - whether visible ones like model and dataset, or obscure ones like logging, distributed environment and random seed - are getting organized and scheduled. The runner deals with the complex relationship between different modules and provides you with a clear, easy-to-understand and configurable interface. The benefits of this design are:

You can modify or add your codes without spoiling your whole codebase. For example, you may start with single GPU training and you can always add a few lines of configuration codes to enable multi GPUs or even multi nodes training.

You can continuously benefit from new features without worrying about backward compatibility. Mixed precision training, visualization, state of the art distributed training methods, various device backends… We will continue to absorb the best suggestions and cutting-edge technologies from the community while ensuring backward compatibility, and provide them to you in a clear interface.

You can focus on your own awesome ideas without being bothered by other annoying and irrelevant details. The default values will handle most cases.

So, MMEngine and the runner will truly make things easier for you. With only a little effort on migration, your code and experiments will evolve with MMEngine. With a little more effort, the config file system allows you to manage your data, model, and experiments more efficiently. Convenience and reliability are the aims we strive for.

The blue one, or the red one - are you prepared to use MMEngine?

## Dataset and DataLoader

Hint

If you have never been exposed to PyTorch’s Dataset and DataLoader classes, you are recommended to read through PyTorch official tutorial to get familiar with some basic concepts.

Datasets and DataLoaders are necessary components in MMEngine’s training pipeline. They are conceptually derived from and consistent with PyTorch. Typically, a dataset defines the quantity, parsing, and pre-processing of the data, while a dataloader iteratively loads data according to settings such as batch_size, shuffle, num_workers, etc. Datasets are encapsulated with dataloaders and they together constitute the data source.

In this tutorial, we will step through their usage in MMEngine runner from the outside (dataloader) to the inside (dataset) and give some practical examples. After reading through this tutorial, you will be able to:

Master the configuration of dataloaders in MMEngine

Learn to use existing datasets (e.g. those from torchvision) from config files

Know about building and using your own dataset

Details on dataloader
Dataloaders can be configured in MMEngine’s Runner with 3 arguments:

train_dataloader: Used in Runner.train() to provide training data for models

val_dataloader: Used in Runner.val() or in Runner.train() at regular intervals for model evaluation

test_dataloader: Used in Runner.test() for the final test

MMEngine has full support for PyTorch native DataLoader objects. Therefore, you can simply pass your valid, already built dataloaders to the runner, as shown in getting started in 15 minutes. Meanwhile, thanks to the Registry Mechanism of MMEngine, those arguments also accept dicts as inputs, as illustrated in the following example (referred to as example 1). The keys in the dictionary correspond to arguments in DataLoader’s init function.

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

When passed to the runner in the form of a dict, the dataloader will be lazily built in the runner when actually needed.

Note

For more configurable arguments of the DataLoader, please refer to PyTorch API documentation

Note

If you are interested in the details of the building procedure, you may refer to build_dataloader

You may find example 1 differs from that in getting started in 15 minutes in some arguments. Indeed, due to some obscure conventions in MMEngine, you can’t seamlessly switch it to a dict by simply replacing DataLoader with dict. We will discuss the differences between our convention and PyTorch’s in the following sections, in case you run into trouble when using config files.

sampler and shuffle
One obvious difference is that we add a sampler argument to the dict. This is because we require sampler to be explicitly specified when using a dict as a dataloader. Meanwhile, shuffle is also removed from DataLoader arguments, because it conflicts with sampler in PyTorch, as referred to in PyTorch DataLoader API documentation.

Note

In fact, shuffle is just a notation for convenience in PyTorch implementation. If shuffle is set to True, the dataloader will automatically switch to RandomSampler

With a sampler argument, codes in example 1 is nearly equivalent to code block below

```python
from mmengine.dataset import DefaultSampler

dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```

Warning

The equivalence of the above codes holds only if: 1) you are training with a single process, and 2) no randomness argument is passed to the runner. This is due to the fact that sampler should be built after distributed environment setup to be correct. The runner will guarantee the correct order and proper random seed by applying lazy initialization techniques, which is only possible for dict inputs. Instead, when building a sampler manually, it requires extra work and is highly error-prone. Therefore, the code block above is just for illustration and definitely not recommended. We strongly suggest passing sampler as a dict to avoid potential problems.

DefaultSampler
The above example may make you wonder what a DefaultSampler is, why use it and whether there are other options. In fact, DefaultSampler is a built-in sampler in MMEngine which eliminates the gap between distributed and non-distributed training and thus enabling a seamless conversion between them. If you have the experience of using DistributedDataParallel in PyTorch, you may be impressed by having to change the sampler argument to make it correct. However, in MMEngine, you don’t need to bother with this DefaultSampler.

DefaultSampler accepts the following arguments:

shuffle: Set to True to load data in the dataset in random order

seed: Random seed used to shuffle the dataset. Typically it doesn’t require manual configuration here because the runner will handle it with randomness configuration

round_up: When set this to True, this is the same behavior as setting drop_last=False in PyTorch DataLoader. You should take care of it when doing migration from PyTorch.

Note

For more details about DefaultSampler, please refer to its API docs

DefaultSampler handles most of the cases. We ensure that error-prone details such as random seeds are handled properly when you are using it in a runner. This prevents you from getting into troubles with distributed training. Apart from DefaultSampler, you may also be interested in InfiniteSampler for iteration-based training pipelines. If you have more advanced demands, you may want to refer to the codes of these two built-in samplers to implement your own one and register it to DATA_SAMPLERS registry.

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

The obscure collate_fn
Among the arguments of PyTorch DataLoader, collate_fn is often ignored by users, but in MMEngine you must pay special attention to it. When you pass the dataloader argument as a dict, MMEngine will use the built-in pseudo_collate by default, which is significantly different from that, default_collate, in PyTorch. Therefore, when doing a migration from PyTorch, you have to explicitly specify the collate_fn in config files to be consistent in behavior.

Note

MMEngine uses pseudo_collate as default value is mainly due to historical compatibility reasons. You don’t have to look deeply into it. You can just know about it and avoid potential errors.

MMEngine provides 2 built-in collate_fn:

pseudo_collate: Default value in MMEngine. It won’t concatenate data through batch index. Detailed explanations can be found in pseudo_collate API doc

default_collate: It behaves almost identically to PyTorch’s default_collate. It will transfer data into Tensor and concatenate them through batch index. More details and slight differences from PyTorch can be found in default_collate API doc

If you want to use a custom collate_fn, you can register it to FUNCTIONS registry.

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

Details on dataset
Typically, datasets define the quantity, parsing, and pre-processing of the data. It is encapsulated in dataloader, allowing the latter to load data in batches. Since we fully support PyTorch DataLoader, the dataset is also compatible. Meanwhile, thanks to the registry mechanism, when a dataloader is given as a dict, its dataset argument can also be given as a dict, which enables lazy initialization in the runner. This mechanism allows for writing config files.

Use torchvision datasets
torchvision provides various open datasets. They can be directly used in MMEngine as shown in getting started in 15 minutes, where a CIFAR10 dataset is used together with torchvision’s built-in data transforms.

However, if you want to use the dataset in config files, registration is needed. What’s more, if you also require data transforms in torchvision, some more registrations are required. The following example illustrates how to do it.

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose

# register CIFAR10 dataset in torchvision
# data transforms should also be built here
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

# register data transforms in torchvision
DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

# specify in runner
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```

Note

The above example makes extensive use of the registry mechanism and borrows the Compose module from MMEngine. If you urge to use torchvision dataset in your config files, you can refer to it and make some slight modifications. However, we recommend you borrow datasets from downstream repos such as MMDet, MMPretrain, etc. This may give you a better experience.

Customize your dataset
You are free to customize your own datasets, as you would with PyTorch. You can also copy existing datasets from your previous PyTorch projects. If you want to learn how to customize your dataset, please refer to PyTorch official tutorials

Use MMEngine BaseDataset
Apart from directly using PyTorch native Dataset class, you can also use MMEngine’s built-in class BaseDataset to customize your own one, as referred to BaseDataset tutorial. It makes some conventions on the format of annotation files, which makes the data interface more unified and multi-task training more convenient. Meanwhile, BaseDataset can easily cooperate with built-in data transforms in MMEngine, which releases you from writing one from scratch.

Currently, BaseDataset has been widely used in downstream repos of OpenMMLab 2.0 projects.

## Model

Model
Runner and model
As mentioned in basic dataflow, the dataflow between DataLoader, model and evaluator follows some rules. Don’t remember clearly? Let’s review it:

```python
# Training process
for data_batch in train_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=True)
    if isinstance(data_batch, dict):
        losses = model(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model(*data_batch, mode='loss')
    else:
        raise TypeError()
# Validation process
for data_batch in val_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=False)
    if isinstance(data_batch, dict):
        outputs = model(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

In runner tutorial, we simply mentioned the relationship between DataLoader, model and evaluator, and introduced the concept of data_preprocessor. You may have a certain understanding of the model. However, during the running of Runner, the situation is far more complex than the above pseudo-code.

In order to focus your attention on the algorithm itself, and ignore the complex relationship between the model, DataLoader and evaluator, we designed BaseModel. In most cases, the only thing you need to do is to make your model inherit from BaseModel, and implement the forward as required to perform the training, testing, and validation process.

Before continuing reading the model tutorial, let’s throw out two questions that we hope you will find the answers after reading the model tutorial:

When do we update the parameters of model? and how to update the parameters by a custom optimization process?

Why is the concept of data_preprocessor necessary? What functions can it perform?

Interface introduction
Usually, we should define a model to implement the body of the algorithm. In MMEngine, model will be managed by Runner, and need to implement some interfaces, such as train_step, val_step, and test_step. For high-level tasks like detection, classification, and segmentation, the interfaces mentioned above commonly implement a standard workflow. For example, train_step will calculate the loss and update the parameters of the model, and val_step/test_step will calculate the metrics and return the predictions. Therefore, MMEnine abstracts the BaseModel to implement the common workflow.

Benefits from the BaseModel, we only need to make the model inherit from BaseModel, and implement the forward function to perform the training, testing, and validation process.

Note

BaseModel inherits from BaseModule, which can be used to initialize the model parameters dynamically.

forward: The arguments of forward need to match with the data given by DataLoader. If the DataLoader samples a tuple data, forward needs to accept the value of unpacked *data. If DataLoader returns a dict data, forward needs to accept the key-value of unpacked **data. forward also accepts mode parameter, which is used to control the running branch:

mode='loss': loss mode is enabled in training process, and forward returns a differentiable loss dict. Each key-value pair in loss dict will be used to log the training status and optimize the parameters of model. This branch will be called by train_step

mode='predict': predict mode is enabled in validation/testing process, and forward will return predictions, which matches with arguments of process. Repositories of OpenMMLab have a more strict rules. The predictions must be a list and each element of it must be a BaseDataElement. This branch will be called by val_step

mode='tensor': In tensor and predict modes, forward will return the predictions. The difference is that forward will return a tensor or a container or tensor which has not been processed by a series of post-process methods, such as non-maximum suppression (NMS). You can customize your post-process method after getting the result of tensor mode.

train_step: Get the loss dict by calling forward with loss mode. BaseModel implements a standard optimization process as follows:

```python
def train_step(self, data, optim_wrapper):
    # See details in the next section
    data = self.data_preprocessor(data, training=True)
    # `loss` mode, return a loss dict. Actually train_step accepts
    #  both tuple  dict input, and unpack it with ** or *
    loss = self(**data, mode='loss')
    # Parse the loss dict and return the parsed losses for optimization
    # and log_vars for logging
    parsed_losses, log_vars = self.parse_losses()
    optim_wrapper.update_params(parsed_losses)
    return log_vars
val_step: Get the predictions by calling forward with predict mode.

def val_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=False)
    outputs = self(**data, mode='predict')
    return outputs
```

test_step: There is no difference between val_step and test_step in BaseModel. But we can customize it in the subclasses, for example, you can get validation loss in val_step.

Understand the interfaces of BaseModel, now we are able to come up with a more complete pseudo-code:

```python
# training
for data_batch in train_dataloader:
    loss_dict = model.train_step(data_batch)
# validation
for data_batch in val_dataloader:
    preds = model.test_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

Great!, ignoring Hook, the pseudo-code above almost implements the main logic in loop! Let’s go back to 15 minutes to get started with MMEngine, we may truly understand what MMResNet has done:

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

    # train_step, val_step and test_step have been implemented in BaseModel.
    # We list the equivalent code here for better understanding
    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
```

Now, you may have a deeper understanding of dataflow, and can answer the first question in Runner and model.

BaseModel.train_step implements the standard optimization, and if we want to customize a new optimization process, we can override it in the subclass. However, it is important to note that we need to make sure that train_step returns a loss dict.

DataPreprocessor
If your computer is equipped with a GPU (or other hardware that can accelerate training, such as MPS, IPU, etc.), when you run the 15 minutes tutorial, you will see that the program is running on the GPU, but, when does MMEngine move the data and model from the CPU to the GPU?

In fact, the Runner will move the model to the specified device during the construction, while the data will be moved to the specified device at the self.data_preprocessor(data) mentioned in the code snippet of the previous section. The moved data will be further passed to the model.

Makes sense but it’s weird, isn’t it? At this point you may be wondering:

MMResNet50 does not define data_preprocessor, but why it can still access data_preprocessor and move data to GPU?

Why BaseModel does not move data by data = data.to(device), but needs the DataPreprocessor to move data?

The answer to the first question is that: MMResNet50 inherit from BaseModel, and super().__init__ will build a default data_preprocessor for it. The equivalent implementation of the default one is like this:

```python
class BaseDataPreprocessor(nn.Module):
    def forward(self, data, training=True):  # ignore the training parameter here
        # suppose data given by CIFAR10 is a tuple. Actually
        # BaseDataPreprocessor could move various type of data
        # to target device.
        return tuple(_data.cuda() for _data in data)
```

BaseDataPreprocessor will move the data to the specified device.

Before answering the second question, let’s think about a few more questions

Where should we perform normalization? transform or Model?

It sounds reasonable to put it in transform to take advantage of Dataloader’s multi-process acceleration, and in the model to move it to GPU to use GPU resources to accelerate normalization. However, while we are debating whether CPU normalization is faster than GPU normalization, the time of data moving from CPU to GPU is much longer than the former.

In fact, for less computationally intensive operations like normalization, it takes much less time than data transferring, which has a higher priority for being optimized. If I could move the data to the specified device while it is still in uint8 and before it is normalized (the size of normalized float data is 4 times larger than that of unit8), it would reduce the bandwidth and greatly improve the efficiency of data transferring. This “lagged” normalization behavior is one of the main reasons why we designed the DataPreprocessor. The data preprocessor moves the data first and then normalizes it.

How we implement the data augmentation like MixUp and Mosaic?

Although it seems that MixUp and Mosaic are just special data transformations that should be implemented in transform. However, considering that these two transformations involve fusing multiple images into one, it would be very difficult to implement them in transform since the current paradigm of transform is to do various enhancements on one image. It would be hard to read additional images from dataset because the dataset is not accessible in the transform. However, if we implement Mosaic or Mixup based on the batch_data sampled from Dataloader, everything becomes easy. We can access multiple images at the same time, and we can easily perform the image fusion operation.

```python
class MixUpDataPreprocessor(nn.Module):
    def __init__(self, num_class, alpha):
        self.alpha = alpha

    def forward(self, data, training=True):
        data = tuple(_data.cuda() for _data in data)
        # Only perform MixUp in training mode
        if not training:
            return data

        label = F.one_hot(label)  # label to OneHot
        batch_size = len(label)
        index = torch.randperm(batch_size)  # Get the index of fused image
        img, label = data
        lam = np.random.beta(self.alpha, self.alpha)  # Fusion factor

        # MixUp
        img = lam * img + (1 - lam) * img[index, :]
        label = lam * batch_scores + (1 - lam) * batch_scores[index, :]
        # Since the returned label is onehot encoded, the `forward` of the
        # model should also be adjusted.
        return tuple(img, label)
```

Therefore, besides data transferring and normalization, another major function of data_preprocessor is BatchAugmentation. The modularity of the data preprocessor also helps us to achieve a free combination between algorithms and data augmentation.

What should we do if the data sampled from the DataLoader does not match the model input, should I modify the DataLoader or the model interface?

The answer is: neither is appropriate. The ideal solution is to do the adaptation without breaking the existing interface between the model and the DataLoader. DataPreprocessor could also handle this, you can customize your DataPreprocessor to convert the incoming to the target type.

By now, You must understand the rationale of the data preprocessor and can confidently answer the two questions posed at the beginning of the tutorial! But you may still wonder what is the optim_wrapper passed to train_step, and how do the predictions returned by test_step and val_step relate to the evaluator. You will find more introduction in the evaluation tutorial and the optimizer wrapper tutorial.

## Evaluation

In model validation and testing, it is often necessary to make a quantitative evaluation of model accuracy. We can achieve this by specifying the metrics in the configuration file.

Evaluation in model training or testing
Using a single evaluation metric
When training or testing a model based on MMEngine, users only need to specify the evaluation metrics for the validation and testing stages through the val_evaluator and test_evaluator fields in the configuration file. For example, when using MMPretrain to train a classification model, if the user wants to evaluate the top-1 and top-5 classification accuracy during the model validation stage, they can configure it as follows:

```python
# using classification accuracy evaluation metric
val_evaluator = dict(type='Accuracy', top_k=(1, 5))
```

For specific parameter settings of evaluation metrics, users can refer to the documentation of the relevant algorithm libraries, such as the Accuracy documentation in the above example.

Using multiple evaluation metrics
If multiple evaluation metrics need to be evaluated simultaneously, val_evaluator or test_evaluator can be set as a list, with each item being the configuration information for an evaluation metric. For example, when using MMDetection to train a panoptic segmentation model, if the user wants to evaluate both the object detection (COCO AP/AR) and panoptic segmentation accuracy during the model testing stage, they can configure it as follows:

```python
test_evaluator = [
    # object detection metric
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        ann_file='annotations/instances_val2017.json',
    ),
    # panoramic segmentation metric
    dict(
        type='CocoPanopticMetric',
        ann_file='annotations/panoptic_val2017.json',
        seg_prefix='annotations/panoptic_val2017',
    )
]
```

Customizing evaluation metrics
If the common evaluation metrics provided in the algorithm library cannot meet the needs, users can also add custom evaluation metrics. As an example, we present the implementation of custom metrics with the simplified classification accuracy:

When defining a new evaluation metric class, you need to inherit the base class BaseMetric (for an introduction to this base class, you can refer to the design document). In addition, the evaluation metric class needs to be registered with the registrar METRICS (for a description of the registrar, please refer to the Registry documentation).

Implement the process() method. This method has two input parameters, which are a batch of test data samples, data_batch, and model prediction results, data_samples. We extract the sample category labels and the classification prediction results from them and store them in self.results respectively.

Implement the compute_metrics() method. This method has one input parameter results, which holds the results of all batches of test data processed by the process() method. The sample category labels and classification predictions are extracted from the results to calculate the classification accuracy (acc). Finally, the calculated evaluation metrics are returned in the form of a dictionary.

(Optional) You can assign a value to the class attribute default_prefix. This attribute is automatically prefixed to the output metric name (e.g. defaut_prefix='my_metric', then the actual output metric name is 'my_metric/acc') to further distinguish the different metrics. This prefix can also be rewritten in the configuration file via the prefix parameter. We recommend describing the default_prefix value for the metric class and the names of all returned metrics in the docstring.

The specific implementation is as follows:

```python
from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # set default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # fetch classification prediction results and category labels
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # store the results of the current batch into self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # aggregate the classification prediction results and category labels for all samples
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # calculate the classification accuracy
        acc = (preds == gts).sum() / preds.size

        # return evaluation metric results
        return {'accuracy': acc}
```

Using offline results for evaluation
Another common way of model evaluation is to perform offline evaluation using model prediction results saved in files in advance. In this case, the user needs to manually build Evaluator and call the corresponding interface of the evaluator to complete the evaluation. For more details about offline evaluation and the relationship between the evaluator and the metric, please refer to the design document. We only give an example of offline evaluation here:

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# Build the evaluator. The parameter `metrics` is the configuration of the evaluation metric
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# Reads the test data from a file. The data format needs to refer to the metric used.
data = load('test_data.pkl')

# The model prediction result is read from the file. The result is inferred by the algorithm to be evaluated on the test dataset.
# The data format needs to refer to the metric used.
data_samples = load('prediction.pkl')

# Call the evaluator offline evaluation interface and get the evaluation results
# chunk_size indicates the number of samples processed at a time, which can be adjusted according to the memory size
results = evaluator.offline_evaluate(data, data_samples, chunk_size=128)
```

## OptimWrapper

OptimWrapper
In previous tutorials of runner and model, we have more or less mentioned the concept of OptimWrapper, but we have not introduced why we need it and what are the advantages of OptimWrapper compared to Pytorch’s native optimizer. In this tutorial, we will help you understand the advantages and demonstrate how to use the wrapper.

As its name suggests, OptimWrapper is a high-level abstraction of PyTorch’s native optimizer, which provides a unified set of interfaces while adding more functionality. OptimWrapper supports different training strategies, including mixed precision training, gradient accumulation, and gradient clipping. We can choose the appropriate training strategy according to our needs. OptimWrapper also defines a standard process for parameter updating based on which users can switch between different training strategies for the same set of code.

OptimWrapper vs Optimizer
Now we use both the native optimizer of PyTorch and the OptimWrapper in MMEngine to perform single-precision training, mixed-precision training, and gradient accumulation to show the difference in implementations.

Model training
1.1 Single-precision training with SGD in PyTorch

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F

inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

1.2 Single-precision training with OptimWrapper in MMEngine

```python
from mmengine.optim import OptimWrapper

optim_wrapper = OptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    optim_wrapper.update_params(loss)
```

The OptimWrapper.update_params achieves the standard process for gradient computation, parameter updating, and gradient zeroing, which can be used to update the model parameters directly.

2.1 Mixed-precision training with SGD in PyTorch

```python
from torch.cuda.amp import autocast

model = model.cuda()
inputs = [torch.zeros(10, 1, 1, 1)] * 10
targets = [torch.ones(10, 1, 1, 1)] * 10

for input, target in zip(inputs, targets):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

2.2 Mixed-precision training with OptimWrapper in MMEngine

```python
from mmengine.optim import AmpOptimWrapper

optim_wrapper = AmpOptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

To enable mixed precision training, users need to use AmpOptimWrapper.optim_context which is similar to the autocast for enabling the context for mixed precision training. In addition, AmpOptimWrapper.optim_context can accelerate the gradient accumulation during the distributed training, which will be introduced in the next example.

3.1 Mixed-precision training and gradient accumulation with SGD in PyTorch

```python
for idx, (input, target) in enumerate(zip(inputs, targets)):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    if idx % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3.2 Mixed-precision training and gradient accumulation with OptimWrapper in MMEngine

```python
optim_wrapper = AmpOptimWrapper(optimizer=optimizer, accumulative_counts=2)

for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

We only need to configure the accumulative_counts parameter and call the update_params interface to achieve the gradient accumulation function. Besides, in the distributed training scenario, if we configure the gradient accumulation with optim_context context enabled, we can avoid unnecessary gradient synchronization during the gradient accumulation step.

The OptimWrapper also provides a more fine-grained interface for users to customize with their own parameter update logics.

backward: Accept a loss dictionary, and compute the gradient of parameters.

step: Same as optimizer.step, and update the parameters.

zero_grad: Same as optimizer.zero_grad, and zero the gradient of parameters

We can use the above interface to implement the same logic of parameters updating as the Pytorch optimizer.

```python
for idx, (input, target) in enumerate(zip(inputs, targets)):
    optimizer.zero_grad()
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.backward(loss)
    if idx % 2 == 0:
        optim_wrapper.step()
        optim_wrapper.zero_grad()
```

We can also configure a gradient clipping strategy for the OptimWrapper.

```python
# based on torch.nn.utils.clip_grad_norm_ method
optim_wrapper = AmpOptimWrapper(
    optimizer=optimizer, clip_grad=dict(max_norm=1))

# based on torch.nn.utils.clip_grad_value_ method
optim_wrapper = AmpOptimWrapper(
    optimizer=optimizer, clip_grad=dict(clip_value=0.2))
```

Get learning rate/momentum
The OptimWrapper provides the get_lr and get_momentum for the convenience of getting the learning rate and momentum of the first parameter group in the optimizer.

```python
import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import OptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optim_wrapper = OptimWrapper(optimizer)

print(optimizer.param_groups[0]['lr'])  # 0.01
print(optimizer.param_groups[0]['momentum'])  # 0
print(optim_wrapper.get_lr())  # {'lr': [0.01]}
print(optim_wrapper.get_momentum())  # {'momentum': [0]}
```

Export/load state dicts
Similar to the optimizer, the OptimWrapper provides the state_dict and load_state_dict interfaces for exporting and loading the optimizer states. For the AmpOptimWrapper, it can export mixed-precision training parameters as well.

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, AmpOptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

optim_wrapper = OptimWrapper(optimizer=optimizer)
amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)

# export state dicts
optim_state_dict = optim_wrapper.state_dict()
amp_optim_state_dict = amp_optim_wrapper.state_dict()

print(optim_state_dict)
print(amp_optim_state_dict)
optim_wrapper_new = OptimWrapper(optimizer=optimizer)
amp_optim_wrapper_new = AmpOptimWrapper(optimizer=optimizer)

# load state dicts
amp_optim_wrapper_new.load_state_dict(amp_optim_state_dict)
optim_wrapper_new.load_state_dict(optim_state_dict)
```

Use multiple optimizers
Considering that algorithms like GANs usually need to use multiple optimizers to train the generator and the discriminator, MMEngine provides a container class called OptimWrapperDict to manage them. OptimWrapperDict stores the sub-OptimWrapper in the form of dict, and can be accessed and traversed just like a dict.

Unlike regular OptimWrapper, OptimWrapperDict does not provide methods such as update_prarms, optim_context, backward, step, etc. Therefore, it cannot be used directly to train models. We suggest implementing the logic of parameter updating by accessing the sub-OptimWarpper in OptimWrapperDict directly.

Users may wonder why not just use dict to manage multiple optimizers since OptimWrapperDict does not have training capabilities. Actually, the core function of OptimWrapperDict is to support exporting or loading the state dictionary of all sub-OptimWrapper and to support getting learning rates and momentums as well. Without OptimWrapperDict, MMEngine needs to do a lot of if-else in OptimWrapper to get the states of the OptimWrappers.

```python
from torch.optim import SGD
import torch.nn as nn

from mmengine.optim import OptimWrapper, OptimWrapperDict

gen = nn.Linear(1, 1)
disc = nn.Linear(1, 1)
optimizer_gen = SGD(gen.parameters(), lr=0.01)
optimizer_disc = SGD(disc.parameters(), lr=0.01)

optim_wapper_gen = OptimWrapper(optimizer=optimizer_gen)
optim_wapper_disc = OptimWrapper(optimizer=optimizer_disc)
optim_dict = OptimWrapperDict(gen=optim_wapper_gen, disc=optim_wapper_disc)

print(optim_dict.get_lr())  # {'gen.lr': [0.01], 'disc.lr': [0.01]}
print(optim_dict.get_momentum())  # {'gen.momentum': [0], 'disc.momentum': [0]}
```

As shown in the above example, OptimWrapperDict exports learning rates and momentums for all OptimWrappers easily, and OptimWrapperDict can export and load all the state dicts in a similar way.

Configure the OptimWapper in Runner
We first need to configure the optimizer for the OptimWrapper. MMEngine automatically adds all optimizers in PyTorch to the OPTIMIZERS registry, and users can specify the optimizers they need in the form of a dict. All supported optimizers in PyTorch are listed here.

Now we take setting up a SGD OptimWrapper as an example.

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
Here we have set up an OptimWrapper with a SGD optimizer with the learning rate and momentum parameters as specified. Since OptimWrapper is designed for standard single precision training, we can also omit the type field in the configuration:

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(optimizer=optimizer)
```

To enable mixed-precision training and gradient accumulation, we change type to AmpOptimWrapper and specify the accumulative_counts parameter.

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, accumulative_counts=2)
```

Note

If you are new to reading the MMEngine tutorial and are not familiar with concepts such as configs and registries, it is recommended to skip the following advanced tutorials for now and read other documents first. Of course, if you already have a good understanding of this prerequisite knowledge, we highly recommend reading the advanced part which covers:

How to customize the learning rate, decay coefficient, and other parameters of the model parameters in the configuration of OptimWrapper.

how to customize the construction policy of the optimizer.

Apart from the pre-requisite knowledge of the configs and the registries, it is recommended to have a thorough understanding of the native construction of PyTorch optimizer before starting the advanced tutorials.

Advanced usages
PyTorch’s optimizer allows different hyperparameters to be set for each parameter in the model, such as using different learning rates for the backbone and head for a classification model.

```python
from torch.optim import SGD
import torch.nn as nn

model = nn.ModuleDict(dict(backbone=nn.Linear(1, 1), head=nn.Linear(1, 1)))
optimizer = SGD([{'params': model.backbone.parameters()},
     {'params': model.head.parameters(), 'lr': 1e-3}],
    lr=0.01,
    momentum=0.9)
```

In the above example, we set a learning rate of 0.01 for the backbone, while another learning rate of 1e-3 for the head. Users can pass a list of dictionaries containing the different parts of the model’s parameters and their corresponding hyperparameters to the optimizer, allowing for fine-grained adjustment of the model optimization.

In MMEngine, the optimizer wrapper constructor allows users to set hyperparameters in different parts of the model directly by setting the paramwise_cfg in the configuration file rather than by modifying the code of building the optimizer.

Set different hyperparamters for different types of parameters
The default optimizer wrapper constructor in MMEngine supports setting different hyperparameters for different types of parameters in the model. For example, we can set norm_decay_mult=0 for paramwise_cfg to set the weight decay factor to 0 for the weight and bias of the normalization layer to implement the trick of not decaying the weight of the normalization layer as mentioned in the Bag of Tricks.

Here, we set the weight decay coefficient in all normalization layers (head.bn) in ToyModel to 0 as follows.

```python
from mmengine.optim import build_optim_wrapper
from collections import OrderedDict

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(
            dict(layer0=nn.Linear(1, 1), layer1=nn.Linear(1, 1)))
        self.head = nn.Sequential(
            OrderedDict(
                linear=nn.Linear(1, 1),
                bn=nn.BatchNorm1d(1)))


optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer = build_optim_wrapper(ToyModel(), optim_wrapper)
```

In addition to configuring the weight decay, paramwise_cfg of MMEngine’s default optimizer wrapper constructor supports the following hyperparameters as well.

lr_mult: Learning rate for all parameters.

decay_mult: Decay coefficient for all parameters.

bias_lr_mult: Learning rate coefficient of the bias (excluding bias of normalization layer and offset of the deformable convolution).

bias_decay_mult: Weight decay coefficient of the bias (excluding bias of normalization layer and offset of the deformable convolution).

norm_decay_mult: Weight decay coefficient for weights and bias of the normalization layer.

flat_decay_mult: Weight decay coefficient of the one-dimension parameters.

dwconv_decay_mult: Decay coefficient of the depth-wise convolution.

bypass_duplicate: Whether to skip duplicate parameters, default to False.

dcn_offset_lr_mult: Learning rate of the deformable convolution.

Set different hyperparamters for different model modules
In addition, as shown in the PyTorch code above, in MMEngine we can also set different hyperparameters for any module in the model by setting custom_keys in paramwise_cfg.

If we want to set the learning rate and the decay coefficient to 0 for backbone.layer0, and set the learning rate to 0.001 for the rest of the modules in the backbone. At the same time, we want to keep all the learning rate to 0.001 for the head module. We can do it in this way:

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
            'backbone': dict(lr_mult=1),
            'head': dict(lr_mult=0.1)
        }))
optimizer = build_optim_wrapper(ToyModel(), optim_wrapper)
```

The state dictionary of the above model can be printed as the following:

```python
for name, val in ToyModel().named_parameters():
    print(name)
```

Each field in custom_keys is defined as follows.

'backbone': dict(lr_mult=1): Set the learning rate of the parameter whose name is prefixed with backbone to 1.

'backbone.layer0': dict(lr_mult=0, decay_mult=0): Set the learning rate of the parameter with the prefix backbone.layer0 to 0 and the decay coefficient to 0. This configuration has a higher priority than the first one.

'head': dict(lr_mult=0.1): Set the learning rate of the parameter whose name is prefixed with head to 0.1.

Customize optimizer construction policies
Like other modules in MMEngine, the optimizer wrapper constructor is also managed by the registry. We can customize the hyperparameter policies by implementing custom optimizer wrapper constructors.

For example, we can implement an optimizer wrapper constructor called LayerDecayOptimWrapperConstructor that automatically set decreasing learning rates for layers of different depths of the model.

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.logging import print_log


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=None)
        self.decay_factor = paramwise_cfg.get('decay_factor', 0.5)

        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(self, params, module, prefix='' ,lr=None):
        if lr is None:
            lr = self.base_lr

        for name, param in module.named_parameters(recurse=False):
            param_group = dict()
            param_group['params'] = [param]
            param_group['lr'] = lr
            params.append(param_group)
            full_name = f'{prefix}.{name}' if prefix else name
            print_log(f'{full_name} : lr={lr}', logger='current')

        for name, module in module.named_children():
            chiled_prefix = f'{prefix}.{name}' if prefix else name
            self.add_params(
                params, module, chiled_prefix, lr=lr * self.decay_factor)


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleDict(dict(linear=nn.Linear(1, 1)))
        self.linear = nn.Linear(1, 1)


model = ToyModel()

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(decay_factor=0.5),
    constructor='LayerDecayOptimWrapperConstructor')

optimizer = build_optim_wrapper(model, optim_wrapper)
```

When add_params is called for the first time, the params argument is an empty list and the module is the ToyModel instance. Please refer to the Optimizer Wrapper Constructor Documentation for detailed explanations on overloading.

Similarly, if we want to construct multiple optimizers, we also need to implement a custom constructor.

```python
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultipleOptimiWrapperConstructor:
    ...
```

Adjust hyperparameters during training
The hyperparameters in the optimizer can only be set to a fixed value at the time it is constructed, and you cannot adjust parameters such as the learning rate during training by just using the optimizer wrapper. In MMEngine, we have implemented a parameter scheduler that allows the tuning of parameters during training. For the usage of the parameter scheduler, please refer to the Parameter Scheduler

## Parameter Scheduler

Parameter Scheduler
During neural network training, optimization hyperparameters (e.g. learning rate) are usually adjusted along with the training process. One of the simplest and most common learning rate adjustment strategies is multi-step learning rate decay, which reduces the learning rate to a fraction at regular intervals. PyTorch provides LRScheduler to implement various learning rate adjustment strategies. In MMEngine, we have extended it and implemented a more general ParamScheduler. It can adjust optimization hyperparameters such as learning rate and momentum. It also supports the combination of multiple schedulers to create more complex scheduling strategies.

Usage
We first introduce how to use PyTorch’s torch.optim.lr_scheduler to adjust learning rate.

How to use PyTorch's builtin learning rate scheduler?
mmengine.optim.scheduler supports most of PyTorch’s learning rate schedulers such as ExponentialLR, LinearLR, StepLR, MultiStepLR, etc. Please refer to parameter scheduler API documentation for all of the supported schedulers.

MMEngine also supports adjusting momentum with parameter schedulers. To use momentum schedulers, replace LR in the class name to Momentum, such as ExponentialMomentum, LinearMomentum. Further, we implement the general parameter scheduler ParamScheduler, which is used to adjust the specified hyperparameters in the optimizer, such as weight_decay, etc. This feature makes it easier to apply some complex hyperparameter tuning strategies.

Different from the above example, MMEngine usually does not need to manually implement the training loop and call optimizer.step(). The runner will automatically manage the training progress and control the execution of the parameter scheduler through ParamSchedulerHook.

Use a single LRScheduler
If only one scheduler needs to be used for the entire training process, there is no difference with PyTorch’s learning rate scheduler.

```python
# build the scheduler manually
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
param_scheduler = MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

runner = Runner(
    model=model,
    optim_wrapper=dict(
        optimizer=optimizer),
    param_scheduler=param_scheduler,
    ...
    )
```

If using the runner with the registry and config file, we can specify the scheduler by setting the param_scheduler field in the config. The runner will automatically build a parameter scheduler based on this field:

```python
# build the scheduler with config file
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
```

Note that the parameter by_epoch is added here, which controls the frequency of learning rate adjustment. When set to True, it means adjusting by epoch. When set to False, it means adjusting by iteration. The default value is True.

In the above example, it means to adjust according to epochs. At this time, the unit of the parameters is epoch. For example, [8, 11] in milestones means that the learning rate will be multiplied by 0.1 at the end of the 8 and 11 epoch.

When the frequency is modified, the meaning of the count-related settings of the scheduler will be changed accordingly. When by_epoch=True, the numbers in milestones indicate at which epoch the learning rate decay is performed, and when by_epoch=False it indicates at which iteration the learning rate decay is performed.

Here is an example of adjusting by iterations: At the end of the 600th and 800th iterations, the learning rate will be multiplied by 0.1 times.

```python
param_scheduler = dict(type='MultiStepLR', by_epoch=False, milestones=[600, 800], gamma=0.1)
```

If users want to use the iteration-based frequency while filling the scheduler config settings by epoch, MMEngine’s scheduler also provides an automatic conversion method. Users can call the build_iter_from_epoch method and provide the number of iterations for each training epoch to construct a scheduler object updated by iterations:

```python
epoch_length = len(train_dataloader)
param_scheduler = MultiStepLR.build_iter_from_epoch(optimizer, milestones=[8, 11], gamma=0.1, epoch_length=epoch_length)
```

If using config to build a scheduler, just add convert_to_iter_based=True to the field. The runner will automatically call build_iter_from_epoch to convert the epoch-based config to an iteration-based scheduler object:

```python
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1, convert_to_iter_based=True)
```

Below is a Cosine Annealing learning rate scheduler that is updated by epoch, where the learning rate is only modified after each epoch:

```python
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=12)
```

After automatically conversion, the learning rate is updated by iteration. As you can see from the graph below, the learning rate changes more smoothly.

```python
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=12, convert_to_iter_based=True)
```

Combine multiple LRSchedulers (e.g. learning rate warm-up)
In the training process of some algorithms, the learning rate is not adjusted according to a certain scheduling strategy from beginning to end. The most common example is learning rate warm-up.

For example, in the first few iterations, a linear strategy is used to increase the learning rate from a small value to normal, and then another strategy is applied.

MMEngine supports combining multiple schedulers together. Just modify the param_scheduler field in the config file to a list of scheduler config, and the ParamSchedulerHook can automatically process the scheduler list. The following example implements learning rate warm-up.

```python
param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # Updated by iterations
         begin=0,
         end=50),  # Warm up for the first 50 iterations
    # The main LRScheduler
    dict(type='MultiStepLR',
         by_epoch=True,  # Updated by epochs
         milestones=[8, 11],
         gamma=0.1)
]
```

Note that the begin and end parameters are added here. These two parameters specify the valid interval of the scheduler. The valid interval usually only needs to be set when multiple schedulers are combined, and can be ignored when using a single scheduler. When the begin and end parameters are specified, it means that the scheduler only takes effect in the [begin, end) interval, and the unit is determined by the by_epoch parameter.

In the above example, the by_epoch of LinearLR in the warm-up phase is False, which means that the scheduler only takes effect in the first 50 iterations. After more than 50 iterations, the scheduler will no longer take effect, and the second scheduler, which is MultiStepLR, will control the learning rate. When combining different schedulers, the by_epoch parameter does not have to be the same for each scheduler.

Here is another example:

```python
param_scheduler = [
    # Use a linear warm-up at [0, 100) iterations
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=100),
    # Use a cosine learning rate at [100, 900) iterations
    dict(type='CosineAnnealingLR',
         T_max=800,
         by_epoch=False,
         begin=100,
         end=900)
]
```

The above example uses a linear learning rate warm-up for the first 100 iterations, and then uses a cosine annealing learning rate scheduler with a period of 800 from the 100th to the 900th iteration.

Users can combine any number of schedulers. If the valid intervals of two schedulers are not connected to each other which leads to an interval that is not covered, the learning rate of this interval remains unchanged. If the valid intervals of the two schedulers overlap, the adjustment of the learning rate will be triggered in the order of the scheduler config (similar with ChainedScheduler).

We recommend using different learning rate scheduling strategies in different stages of training to avoid overlapping of the valid intervals. Be careful If you really need to stack two schedulers overlapped. We recommend using learning rate visualization tool to visualize the learning rate after stacking, to avoid the adjustment not as expected.

How to adjust other hyperparameters
Momentum
Like learning rate, momentum is a schedulable hyperparameter in the optimizer’s parameter group. The momentum scheduler is used in exactly the same way as the learning rate scheduler. Just add the momentum scheduler config to the list in the param_scheduler field.

Example:

```python
param_scheduler = [
    # the lr scheduler
    dict(type='LinearLR', ...),
    # the momentum scheduler
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

Generic parameter scheduler
MMEngine also provides a set of generic parameter schedulers for scheduling other hyperparameters in the param_groups of the optimizer. Change LR in the class name of the learning rate scheduler to Param, such as LinearParamScheduler. Users can schedule the specific hyperparameters by setting the param_name variable of the scheduler.

Here is an example:

```python
param_scheduler = [
    dict(type='LinearParamScheduler',
         param_name='lr',  # adjust the 'lr' in `optimizer.param_groups`
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

By setting the param_name to 'lr', this parameter scheduler is equivalent to LinearLRScheduler.

In addition to learning rate and momentum, users can also schedule other parameters in optimizer.param_groups. The schedulable parameters depend on the optimizer used. For example, when using the SGD optimizer with weight_decay, the weight_decay can be adjusted as follows:

```python
param_scheduler = [
    dict(type='LinearParamScheduler',
         param_name='weight_decay',  # adjust 'weight_decay' in `optimizer.param_groups`
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```
