# Quantization

## Introduction to Quantization

## Dynamic Quantization

CODED

## FX Graph Mode Quantization User Guide

CODED

## FX Graph Mode Post Training Dynamic Quantization

CODED

## Static Quantization with Eager Mode in PyTorch

CODED

## FX Graph Mode Post Training Static Quantization

CODED

## PyTorch Numeric Suite Tutorial

CODED

## PyTorch 2 Export Post Training Quantization

This tutorial introduces the steps to do post training static quantization in graph mode based on `torch._export.export`. Compared to FX Graph Mode Quantization, this flow is expected to have significantly higher model coverage (88% on 14K models), better programmability, and a simplified UX.

Exportable by `torch.export.export` is a prerequisite to use the flow, you can find what are the constructs that’s supported in Export DB.

The high level architecture of quantization 2 with quantizer could look like this:

```text
float_model(Python)                          Example Input
    \                                              /
     \                                            /
—-------------------------------------------------------
|                        export                        |
—-------------------------------------------------------
                            |
                    FX Graph in ATen     Backend Specific Quantizer
                            |                       /
—--------------------------------------------------------
|                     prepare_pt2e                      |
—--------------------------------------------------------
                            |
                     Calibrate/Train
                            |
—--------------------------------------------------------
|                    convert_pt2e                       |
—--------------------------------------------------------
                            |
                    Quantized Model
                            |
—--------------------------------------------------------
|                       Lowering                        |
—--------------------------------------------------------
                            |
        Executorch, Inductor or <Other Backends>
```

The PyTorch 2 export quantization API looks like this:

```python
import torch
from torch._export import capture_pre_autograd_graph

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)

example_inputs = (torch.randn(1, 5),)
m = M().eval()

# Step 1. program capture
# NOTE: this API will be updated to torch.export API in the future, but the captured
# result shoud mostly stay the same
m = capture_pre_autograd_graph(m, *example_inputs)
# we get a model with aten ops

# Step 2. quantization
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)

# calibration omitted

m = convert_pt2e(m)
# we have a model with aten ops doing integer computations when possible
```

### Motivation of PyTorch 2 Export Quantization

In PyTorch versions prior to 2, we have FX Graph Mode Quantization that uses `QConfigMapping` and `BackendConfig` for customizations. `QConfigMapping` allows modeling users to specify how they want their model to be quantized, `BackendConfig` allows backend developers to specify the supported ways of quantization in their backend. While that API covers most use cases relatively well, it is not fully extensible. There are two main limitations for the current API:

1. Limitation around expressing quantization intentions for complicated operator patterns (how an operator pattern should be observed/quantized) using existing objects: `QConfig` and `QConfigMapping`.
2. Limited support on how user can express their intention of how they want their model to be quantized. For example, if users want to quantize the every other linear in the model, or the quantization behavior has some dependency on the actual shape of the Tensor (for example, only observe/quantize inputs and outputs when the linear has a 3D input), backend developer or modeling users need to change the core quantization API/flow.

A few improvements could make the existing flow better:

1. We use `QConfigMapping` and `BackendConfig` as separate objects, `QConfigMapping` describes user’s intention of how they want their model to be quantized, `BackendConfig` describes what kind of quantization a backend supports. `BackendConfig` is backend-specific, but `QConfigMapping` is not, and the user can provide a `QConfigMapping` that is incompatible with a specific `BackendConfig`, this is not a great UX. Ideally, we can structure this better by making both configuration (`QConfigMapping`) and quantization capability (`BackendConfig`) backend-specific, so there will be less confusion about incompatibilities.
2. In `QConfig` we are exposing observer/ fake_quant observer classes as an object for the user to configure quantization, this increases the things that the user may need to care about. For example, not only the dtype but also how the observation should happen, these could potentially be hidden from the user so that the user flow is simpler.

Here is a summary of the benefits of the new API:

- Programmability (addressing 1. and 2.): When a user’s quantization needs are not covered by available quantizers, users can build their own quantizer and compose it with other quantizers as mentioned above.
- Simplified UX (addressing 3.): Provides a single instance with which both backend and users interact. Thus you no longer have the user facing quantization config mapping to map users intent and a separate quantization config that backends interact with to configure what backend support. We will still have a method for users to query what is supported in a quantizer. With a single instance, composing different quantization capabilities also becomes more natural than previously.

For example XNNPACK does not support `embedding_byte` and we have natively support for this in ExecuTorch. Thus, if we had `ExecuTorchQuantizer` that only quantized `embedding_byte`, then it can be composed with `XNNPACKQuantizer`. (Previously, this used to be concatenating the two `BackendConfig` together and since options in `QConfigMapping` are not backend specific, user also need to figure out how to specify the configurations by themselves that matches the quantization capabilities of the combined backend. With a single quantizer instance, we can compose two quantizers and query the composed quantizer for capabilities, which makes it less error prone and cleaner, for example, `composed_quantizer.quantization_capabilities()`).

- Separation of concerns (addressing 4.): As we design the quantizer API, we also decouple specification of quantization, as expressed in terms of dtype, min/max (# of bits), symmetric, and so on, from the observer concept. Currently, the observer captures both quantization specification and how to observe (Histogram vs MinMax observer). Modeling users are freed from interacting with observer and fake quant objects with this change.

### Define Helper Functions and Prepare Dataset

We’ll start by doing the necessary imports, defining some helper functions and prepare the data. These steps are identical to Static Quantization with Eager Mode in PyTorch.

To run the code in this tutorial using the entire ImageNet dataset, first download Imagenet by following the instructions at [here](ImageNet Data). Unzip the downloaded file into the `data_path` folder.

Download the torchvision resnet18 model and rename it to `data/resnet18_pretrained_float.pth`.

```python
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(self.__dict__)

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
example_inputs = (next(iter(data_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cpu")
float_model.eval()

# create another instance of the model since
# we need to keep the original model around
model_to_quantize = load_model(saved_model_dir + float_model_file).to("cpu")
```

### Set the model to eval mode

For post training quantization, we’ll need to set the model to the eval mode.

```python
model_to_quantize.eval()
```

### Export the model with `torch.export`

Here is how you can use `torch.export` to export the model:

```python
from torch._export import capture_pre_autograd_graph

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)
# or capture with dynamic dimensions
# from torch._export import dynamic_dim
# exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)])
```

`capture_pre_autograd_graph` is a short term API, it will be updated to use the official `torch.export` API when that is ready.

### Import the Backend Specific Quantizer and Configure how to Quantize the Model

The following code snippets describes how to quantize the model:

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
```

Quantizer is backend specific, and each Quantizer will provide their own way to allow users to configure their model. Just as an example, here is the different configuration APIs supported by `XNNPackQuantizer`:

```python
quantizer.set_global(qconfig_opt)  # qconfig_opt is an optional quantization config
    .set_object_type(torch.nn.Conv2d, qconfig_opt) # can be a module type
    .set_object_type(torch.nn.functional.linear, qconfig_opt) # or torch functional op
    .set_module_name("foo.bar", qconfig_opt)
```

NOTE

Check out our tutorial that describes how to write a new Quantizer.

### Prepare the Model for Post Training Quantization

`prepare_pt2e` folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.

```python
prepared_model = prepare_pt2e(exported_model, quantizer)
print(prepared_model.graph)
```

### Calibration

The calibration function is run after the observers are inserted in the model. The purpose for calibration is to run through some sample examples that is representative of the workload (for example a sample of the training data set) so that the observers in the model are able to observe the statistics of the Tensors and we can later use this information to calculate quantization parameters.

```python
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

calibrate(prepared_model, data_loader_test)  # run calibration on sample data
```

### Convert the Calibrated Model to a Quantized Model

`convert_pt2e` takes a calibrated model and produces a quantized model.

```python
quantized_model = convert_pt2e(prepared_model)
print(quantized_model)
```

At this step, we currently have two representations that you can choose from, but exact representation we offer in the long term might change based on feedback from PyTorch users.

- Q/DQ Representation (default)

Previous documentation for representations all quantized operators are represented as dequantize -> fp32_op -> qauntize.

```python
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
             x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
             weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    weight_permuted = torch.ops.aten.permute_copy.default(weight_fp32, [1, 0]);
    out_fp32 = torch.ops.aten.addmm.default(bias_fp32, x_fp32, weight_permuted)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
    out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8
```

- Reference Quantized Model Representation (available in the nightly build)

We will have a special representation for selected ops, for example, quantized linear. Other ops are represented as dq -> float32_op -> q and q/dq are decomposed into more primitive operators. You can get this representation by using `convert_pt2e(..., use_reference_representation=True)`.

```python
# Reference Quantized Pattern for quantized linear
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_int16 = x_int8.to(torch.int16)
    weight_int16 = weight_int8.to(torch.int16)
    acc_int32 = torch.ops.out_dtype(torch.mm, torch.int32, (x_int16 - x_zero_point), (weight_int16 - weight_zero_point))
    bias_scale = x_scale * weight_scale
    bias_int32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_int32 = acc_int32 + bias_int32
    acc_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, torch.int32, acc_int32, x_scale * weight_scale / output_scale) + output_zero_point
    out_int8 = torch.ops.aten.clamp(acc_int32, qmin, qmax).to(torch.int8)
    return out_int8
```

### Checking Model Size and Accuracy Evaluation

Now we can compare the size and model accuracy with baseline model.

```python
# Baseline model size and accuracy
scripted_float_model_file = "resnet18_scripted.pth"

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Quantized model size and accuracy
print("Size of model after quantization")
print_size_of_model(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
```

NOTE

We can’t do performance evaluation now since the model is not lowered to target device, it’s just a representation of quantized computation in ATen operators.

NOTE

The weights are still in fp32 right now, we may do constant propagation for quantize op to get integer weights in the future.

If you want to get better accuracy or performance, try configuring quantizer in different ways, and each quantizer will have its own way of configuration, so please consult the documentation for the quantizer you are using to learn more about how you can have more control over how to quantize a model.

### Save and Load Quantized Model

We’ll show how to save and load the quantized model.

```python
# 0. Store reference output, for example, inputs, and check evaluation accuracy:
example_inputs = (next(iter(data_loader))[0],)
ref = quantized_model(*example_inputs)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# 1. Export the model and Save ExportedProgram
pt2e_quantized_model_file_path = saved_model_dir + "resnet18_pt2e_quantized.pth"
# capture the model to get an ExportedProgram
quantized_ep = torch.export.export(quantized_model, example_inputs)
# use torch.export.save to save an ExportedProgram
torch.export.save(quantized_ep, pt2e_quantized_model_file_path)

# 2. Load the saved ExportedProgram
loaded_quantized_ep = torch.export.load(pt2e_quantized_model_file_path)
loaded_quantized_model = loaded_quantized_ep.module()

# 3. Check results for example inputs and check evaluation accuracy again:
res = loaded_quantized_model(*example_inputs)
print("diff:", ref - res)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
```

### Debugging the Quantized Model

You can use Numeric Suite that can help with debugging in eager mode and FX graph mode. The new version of Numeric Suite working with PyTorch 2 Export models is still in development.

### Lowering and Performance Evaluation

The model produced at this point is not the final model that runs on the device, it is a reference quantized model that captures the intended quantized computation from the user, expressed as ATen operators and some additional quantize/dequantize operators, to get a model that runs on real devices, we’ll need to lower the model. For example, for the models that run on edge devices, we can lower with delegation and ExecuTorch runtime operators.

### (PyTorch 2 Export Post Training Quantization) Conclusion

In this tutorial, we went through the overall quantization flow in PyTorch 2 Export Quantization using `XNNPACKQuantizer` and got a quantized model that could be further lowered to a backend that supports inference with XNNPACK backend. To use this for your own backend, please first follow the tutorial and implement a Quantizer for your backend, and then quantize the model with that Quantizer.

## PyTorch 2 Export Quantization-Aware Training (QAT)

This tutorial shows how to perform quantization-aware training (QAT) in graph mode based on `torch.export.export`. For more details about PyTorch 2 Export Quantization in general, refer to the post training quantization tutorial.

The PyTorch 2 Export QAT flow looks like the following—it is similar to the post training quantization (PTQ) flow for the most part:

```python
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)

example_inputs = (torch.randn(1, 5),)
m = M()

# Step 1. program capture
# NOTE: this API will be updated to torch.export API in the future, but the captured
# result shoud mostly stay the same
m = capture_pre_autograd_graph(m, *example_inputs)
# we get a model with aten ops

# Step 2. quantization-aware training
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_qat_pt2e(m, quantizer)

# train omitted

m = convert_pt2e(m)
# we have a model with aten ops doing integer computations when possible

# move the quantized model to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(m)
```

Note that calling `model.eval()` or `model.train()` after program capture is not allowed, because these methods no longer correctly change the behavior of certain ops like dropout and batch normalization. Instead, please use `torch.ao.quantization.move_exported_model_to_eval()` and `torch.ao.quantization.move_exported_model_to_train()` (coming soon) respectively.

### Define Helper Functions and Prepare the Dataset

To run the code in this tutorial using the entire ImageNet dataset, first download ImageNet by following the instructions in ImageNet Data. Unzip the downloaded file into the `data_path` folder.

Next, download the torchvision resnet18 model and rename it to `data/resnet18_pretrained_float.pth`.

We’ll start by doing the necessary imports, defining some helper functions and prepare the data. These steps are very similar to the ones defined in the static eager mode post training quantization tutorial:

```python
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(self.__dict__)

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, device):
    torch.ao.quantization.move_exported_model_to_eval(model)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    # Note: do not call model.train() here, since this doesn't work on an exported model.
    # Instead, call `torch.ao.quantization.move_exported_model_to_train(model)`, which will
    # be added in the near future
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 32
eval_batch_size = 32

data_loader, data_loader_test = prepare_data_loaders(data_path)
example_inputs = (next(iter(data_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cuda")
```

### Export the model with `torch.export` (PyTorch 2 Export Quantization-Aware Training (QAT))

Here is how you can use `torch.export` to export the model:

```python
from torch._export import capture_pre_autograd_graph

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(float_model, example_inputs)
# or, to capture with dynamic dimensions:
from torch._export import dynamic_dim

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(
    float_model,
    example_inputs,
    constraints=[dynamic_dim(example_inputs[0], 0)],
)
```

NOTE: `capture_pre_autograd_graph` is a short term API, it will be updated to use the official `torch.export` API when that is ready.

### Import the Backend Specific Quantizer and Configure how to Quantize the Model (PyTorch 2 Export Quantization-Aware Training (QAT))

The following code snippets describe how to quantize the model:

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
```

Quantizer is backend specific, and each Quantizer will provide their own way to allow users to configure their model.

NOTE: Check out our tutorial that describes how to write a new Quantizer.

### Prepare the Model for Quantization-Aware Training

`prepare_qat_pt2e` inserts fake quantizes in appropriate places in the model and performs the appropriate QAT “fusions”, such as Conv2d + BatchNorm2d, for better training accuracies. The fused operations are represented as a subgraph of ATen ops in the prepared graph.

```python
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
print(prepared_model)
```

NOTE: If your model contains batch normalization, the actual ATen ops you get in the graph depend on the model’s device when you export the model. If the model is on CPU, then you’ll get `torch.ops.aten._native_batch_norm_legit`. If the model is on CUDA, then you’ll get `torch.ops.aten.cudnn_batch_norm`. However, this is not fundamental and may be subject to change in the future.

Between these two ops, it has been shown that `torch.ops.aten.cudnn_batch_norm` provides better numerics on models like MobileNetV2. To get this op, either call `model.cuda()` before export, or run the following after prepare to manually swap the ops:

```python
for n in prepared_model.graph.nodes:
    if n.target == torch.ops.aten._native_batch_norm_legit.default:
        n.target = torch.ops.aten.cudnn_batch_norm.default
prepared_model.recompile()
```

In the future, we plan to consolidate the batch normalization ops such that the above will no longer be necessary.

### Training Loop

The training loop is similar to the ones in previous versions of QAT. To achieve better accuracies, you may optionally disable observers and updating batch normalization statistics after a certain number of epochs, or evaluate the QAT or the quantized model trained so far every N epochs.

```python
num_epochs = 10
num_train_batches = 20
num_eval_batches = 20
num_observer_update_epochs = 4
num_batch_norm_update_epochs = 3
num_epochs_between_evals = 2

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(num_epochs):
    train_one_epoch(prepared_model, criterion, optimizer, data_loader, "cuda", num_train_batches)

    # Optionally disable observer/batchnorm stats after certain number of epochs
    if epoch >= num_observer_update_epochs:
        print("Disabling observer for subseq epochs, epoch = ", epoch)
        prepared_model.apply(torch.ao.quantization.disable_observer)
    if epoch >= num_batch_norm_update_epochs:
        print("Freezing BN for subseq epochs, epoch = ", epoch)
        for n in prepared_model.graph.nodes:
            # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
            # We set the `training` flag to False here to freeze BN stats
            if n.target in [
                torch.ops.aten._native_batch_norm_legit.default,
                torch.ops.aten.cudnn_batch_norm.default,
            ]:
                new_args = list(n.args)
                new_args[5] = False
                n.args = new_args
        prepared_model.recompile()

    # Check the quantized accuracy every N epochs
    # Note: If you wish to just evaluate the QAT model (not the quantized model),
    # then you can just call `torch.ao.quantization.move_exported_model_to_eval/train`.
    # However, the latter API is not ready yet and will be available in the near future.
    if (nepoch + 1) % num_epochs_between_evals == 0:
        prepared_model_copy = copy.deepcopy(prepared_model)
        quantized_model = convert_pt2e(prepared_model_copy)
        top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d: Evaluation accuracy on %d images, %2.2f' % (nepoch, num_eval_batches * eval_batch_size, top1.avg))
```

### Saving and Loading Model Checkpoints

Model checkpoints for the PyTorch 2 Export QAT flow are the same as in any other training flow. They are useful for pausing training and resuming it later, recovering from failed training runs, and performing inference on different machines at a later time. You can save model checkpoints during or after training as follows:

```python
checkpoint_path = "/path/to/my/checkpoint_%s.pth" % nepoch
torch.save(prepared_model.state_dict(), "checkpoint_path")
```

To load the checkpoints, you must export and prepare the model the exact same way it was initially exported and prepared. For example:

```python
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchvision.models.resnet import resnet18

example_inputs = (torch.rand(2, 3, 224, 224),)
float_model = resnet18(pretrained=False)
exported_model = capture_pre_autograd_graph(float_model, example_inputs)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
prepared_model.load_state_dict(torch.load(checkpoint_path))

# resume training or perform inference
```

### Convert the Trained Model to a Quantized Model

`convert_pt2e` takes a calibrated model and produces a quantized model. Note that, before inference, you must first call `torch.ao.quantization.move_exported_model_to_eval()` to ensure certain ops like dropout behave correctly in the eval graph. Otherwise, we would continue to incorrectly apply dropout in the forward pass during inference, for example.

```python
quantized_model = convert_pt2e(prepared_model)

# move certain ops like dropout to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(m)

print(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Final evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
```

### Conclusion (PyTorch 2 Export Quantization-Aware Training (QAT))

In this tutorial, we demonstrated how to run Quantization-Aware Training (QAT) flow in PyTorch 2 Export Quantization. After convert, the rest of the flow is the same as Post-Training Quantization (PTQ); the user can serialize/deserialize the model and further lower it to a backend that supports inference with XNNPACK backend. For more detail, follow the PTQ tutorial.

## PyTorch BackendConfig

The BackendConfig API enables developers to integrate their backends with PyTorch quantization. It is currently only supported in FX graph mode quantization, but support may be extended to other modes of quantization in the future. In this tutorial, we will demonstrate how to use this API to customize quantization support for specific backends.

Suppose we are a backend developer and we wish to integrate our backend with PyTorch’s quantization APIs. Our backend consists of two ops only: quantized linear and quantized conv-relu. In this section, we will walk through how to achieve this by quantizing an example model using a custom BackendConfig through `prepare_fx` and `convert_fx`.

```python
import torch
from torch.ao.quantization import (
    default_weight_observer,
    get_default_qconfig_mapping,
    MinMaxObserver,
    QConfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
```

### 1. Derive reference pattern for each quantized operator

For quantized linear, suppose our backend expects the reference pattern `[dequant - fp32_linear - quant]` and lowers it into a single quantized linear op. The way to achieve this is to first insert quant-dequant ops before and after the float linear op, such that we produce the following reference model:

```text
quant1 - [dequant1 - fp32_linear - quant2] - dequant2
```

Similarly, for quantized conv-relu, we wish to produce the following reference model, where the reference pattern in the square brackets will be lowered into a single quantized conv-relu op:

```text
quant1 - [dequant1 - fp32_conv_relu - quant2] - dequant2
```

### 2. Set DTypeConfigs with backend constraints

In the reference patterns above, the input dtype specified in the `DTypeConfig` will be passed as the dtype argument to `quant1`, while the output dtype will be passed as the dtype argument to `quant2`. If the output dtype is `fp32`, as in the case of dynamic quantization, then the output quant-dequant pair will not be inserted. This example also shows how to specify restrictions on quantization and scale ranges on a particular dtype.

```python
quint8_with_constraints = DTypeWithConstraints(
    dtype=torch.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_min_lower_bound=2  -12,
)

# Specify the dtypes passed to the quantized ops in the reference model spec
weighted_int8_dtype_config = DTypeConfig(
    input_dtype=quint8_with_constraints,
    output_dtype=quint8_with_constraints,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)
```

### 3. Set up fusion for conv-relu

Note that the original user model contains separate conv and relu ops, so we need to first fuse the conv and relu ops into a single conv-relu op (`fp32_conv_relu`), and then quantize this op similar to how the linear op is quantized. We can set up fusion by defining a function that accepts 3 arguments, where the first is whether or not this is for QAT, and the remaining arguments refer to the individual items of the fused pattern.

```python
def fuse_conv2d_relu(is_qat, conv, relu):
    """Return a fused ConvReLU2d from individual conv and relu modules."""
    return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)
```

### 4. Define the BackendConfig

Now we have all the necessary pieces, so we go ahead and define our `BackendConfig`. Here we use different observers (will be renamed) for the input and output for the linear op, so the quantization params passed to the two quantize ops (`quant1` and `quant2`) will be different. This is commonly the case for weighted ops like linear and conv.

For the conv-relu op, the observation type is the same. However, we need two `BackendPatternConfigs` to support this op, one for fusion and one for quantization. For both conv-relu and linear, we use the `DTypeConfig` defined above.

```python
linear_config = BackendPatternConfig() \
    .set_pattern(torch.nn.Linear) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Linear) \
    .set_qat_module(torch.nn.qat.Linear) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

# For fusing Conv2d + ReLU into ConvReLU2d
# No need to set observation type and dtype config here, since we are not
# inserting quant-dequant ops in this step yet
conv_relu_config = BackendPatternConfig() \
    .set_pattern((torch.nn.Conv2d, torch.nn.ReLU)) \
    .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_fuser_method(fuse_conv2d_relu)

# For quantizing ConvReLU2d
fused_conv_relu_config = BackendPatternConfig() \
    .set_pattern(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Conv2d) \
    .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

backend_config = BackendConfig("my_backend") \
    .set_backend_pattern_config(linear_config) \
    .set_backend_pattern_config(conv_relu_config) \
    .set_backend_pattern_config(fused_conv_relu_config)
```

### 5. Set up QConfigMapping that satisfies the backend constraints

In order to use the ops defined above, the user must define a `QConfig` that satisfies the constraints specified in the `DTypeConfig`. For more detail, see the documentation for `DTypeConfig`. We will then use this `QConfig` for all the modules used in the patterns we wish to quantize.

```python
# Note: Here we use a quant_max of 127, but this could be up to 255 (see `quint8_with_constraints`)
activation_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127, eps=2  -12)
qconfig = QConfig(activation=activation_observer, weight=default_weight_observer)

# Note: All individual items of a fused pattern, e.g. Conv2d and ReLU in
# (Conv2d, ReLU), must have the same QConfig
qconfig_mapping = QConfigMapping() \
    .set_object_type(torch.nn.Linear, qconfig) \
    .set_object_type(torch.nn.Conv2d, qconfig) \
    .set_object_type(torch.nn.BatchNorm2d, qconfig) \
    .set_object_type(torch.nn.ReLU, qconfig)
```

### 6. Quantize the model through prepare and convert

Finally, we quantize the model by passing the `BackendConfig` we defined into `prepare` and `convert`. This produces a quantized linear module and a fused quantized conv-relu module.

```python
class MyModel(torch.nn.Module):
    def __init__(self, use_bn: bool):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.use_bn = use_bn

    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
model = MyModel(use_bn=False)
prepared = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
prepared(*example_inputs)  # calibrate
converted = convert_fx(prepared, backend_config=backend_config)
>>> print(converted)

GraphModule(
  (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.012136868201196194, zero_point=67, qscheme=torch.per_tensor_affine)
  (conv): QuantizedConvReLU2d(3, 3, kernel_size=(3, 3), stride=(1, 1), scale=0.0029353597201406956, zero_point=0)
  (sigmoid): Sigmoid()
)

def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
    conv = self.conv(linear);  linear = None
    dequantize_2 = conv.dequantize();  conv = None
    sigmoid = self.sigmoid(dequantize_2);  dequantize_2 = None
    return sigmoid
```

### 7. Experiment with faulty BackendConfig setups

As an experiment, here we modify the model to use conv-bn-relu instead of conv-relu, but use the same `BackendConfig`, which doesn’t know how to quantize conv-bn-relu. As a result, only linear is quantized, but conv-bn-relu is neither fused nor quantized.

```python
>>> print(converted)

GraphModule(
  (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.015307803638279438, zero_point=95, qscheme=torch.per_tensor_affine)
  (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
)

def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
    dequantize_1 = linear.dequantize();  linear = None
    conv = self.conv(dequantize_1);  dequantize_1 = None
    bn = self.bn(conv);  conv = None
    relu = self.relu(bn);  bn = None
    sigmoid = self.sigmoid(relu);  relu = None
    return sigmoid
```

As another experiment, here we use the default `QConfigMapping` that doesn’t satisfy the dtype constraints specified in the backend. As a result, nothing is quantized since the `QConfigs` are simply ignored.

```python
>>> print(converted)

GraphModule(
  (linear): Linear(in_features=10, out_features=3, bias=True)
  (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
)

def forward(self, x):
    linear = self.linear(x);  x = None
    conv = self.conv(linear);  linear = None
    bn = self.bn(conv);  conv = None
    relu = self.relu(bn);  bn = None
    sigmoid = self.sigmoid(relu);  relu = None
    return sigmoid
```

### Built-in BackendConfigs

PyTorch quantization supports a few built-in native `BackendConfigs` under the `torch.ao.quantization.backend_config` namespace:

- `get_fbgemm_backend_config`: for server target settings
- `get_qnnpack_backend_config`: for mobile and edge device target settings, also supports XNNPACK quantized ops
- `get_native_backend_config` (default): a `BackendConfig` that supports a union of the operator patterns supported in the FBGEMM and QNNPACK `BackendConfigs`

There are also other `BackendConfigs` under development (e.g. for TensorRT and x86), but these are still mostly experimental at the moment. If the user wishes to integrate a new, custom backend with PyTorch’s quantization API, they may define their own `BackendConfigs` using the same set of APIs used to define the natively supported ones as in the example above.

### BackendConfig

CLASS `torch.ao.quantization.backend_config.BackendConfig(name='')` [SOURCE]

Config that defines the set of patterns that can be quantized on a given backend, and how reference quantized models can be produced from these patterns.

A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph of the above. Each pattern supported on the target backend can be individually configured through `BackendPatternConfig` in terms of:

- The supported input/output activation, weight, and bias data types
- How observers and quant/dequant ops are inserted in order to construct the reference pattern, and
- (Optionally) Fusion, QAT, and reference module mappings.

The format of the patterns is described in: [https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md)

#### Example usage

```python
import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

weighted_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)

def fuse_conv2d_relu(is_qat, conv, relu):
    return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

# For quantizing Linear
linear_config = BackendPatternConfig(torch.nn.Linear) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Linear) \
    .set_qat_module(torch.ao.nn.qat.Linear) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

# For fusing Conv2d + ReLU into ConvReLU2d
conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU)) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_fuser_method(fuse_conv2d_relu)

# For quantizing ConvReLU2d
fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Conv2d) \
    .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

backend_config = BackendConfig("my_backend") \
    .set_backend_pattern_config(linear_config) \
    .set_backend_pattern_config(conv_relu_config) \
    .set_backend_pattern_config(fused_conv_relu_config)
```

#### Properties

- configs: `LIST[BACKENDPATTERNCONFIG]`
  - Return a copy of the list of configs set in this `BackendConfig`.

#### Class Methods

- from_dict(backend_config_dict) [SOURCE]
  - Create a `BackendConfig` from a dictionary with the following items:
    - “name”: the name of the target backend
    - “configs”: a list of dictionaries that each represents a `BackendPatternConfig`
  - Return type: `BackendConfig`

#### Methods

- set_backend_pattern_config(config) [SOURCE]
  - Set the config for a pattern that can be run on the target backend. This overrides any existing config for the given pattern.
  - Return type: `BackendConfig`

- set_backend_pattern_configs(configs) [SOURCE]
  - Set the configs for patterns that can be run on the target backend. This overrides any existing config for a given pattern if it was previously registered already.
  - Return type: `BackendConfig`

- set_name(name) [SOURCE]
  - Set the name of the target backend.
  - Return type: `BackendConfig`

- to_dict() [SOURCE]
  - Convert this `BackendConfig` to a dictionary with the items described in `from_dict()`.
  - Return type: `Dict[str, Any]`

### QConfigMapping

CLASS `torch.ao.quantization.qconfig_mapping.QConfigMapping` [SOURCE]

Mapping from model ops to `torch.ao.quantization.QConfig`s.

The user can specify QConfigs using the following methods (in increasing match priority):

- `set_global`: sets the global (default) QConfig
- `set_object_type`: sets the QConfig for a given module type, function, or method name
- `set_module_name_regex`: sets the QConfig for modules matching the given regex string
- `set_module_name`: sets the QConfig for modules matching the given module name
- `set_module_name_object_type_order`: sets the QConfig for modules matching a combination of the given module name, object type, and the index at which the module appears

#### Example usage (QConfigMapping)

```python
qconfig_mapping = QConfigMapping()
    .set_global(global_qconfig)
    .set_object_type(torch.nn.Linear, qconfig1)
    .set_object_type(torch.nn.ReLU, qconfig1)
    .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
    .set_module_name_regex("foo.*", qconfig2)
    .set_module_name("module1", qconfig1)
    .set_module_name("module2", qconfig2)
    .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)
```

CLASSMETHOD `from_dict(qconfig_dict)` [SOURCE]

Create a `QConfigMapping` from a dictionary with the following keys (all optional):

- `""` (for global QConfig)
- `"object_type"`
- `"module_name_regex"`
- `"module_name"`
- `"module_name_object_type_order"`

The values of this dictionary are expected to be lists of tuples.

Return type: `QConfigMapping`

---

METHOD `set_global(global_qconfig)` [SOURCE]

Set the global (default) QConfig.

Return type: `QConfigMapping`

---

METHOD `set_module_name(module_name, qconfig)` [SOURCE]

Set the QConfig for modules matching the given module name. If the QConfig for an existing module name was already set, the new QConfig will override the old one.

Return type: `QConfigMapping`

---

METHOD `set_module_name_object_type_order(module_name, object_type, index, qconfig)` [SOURCE]

Set the QConfig for modules matching a combination of the given module name, object type, and the index at which the module appears.

If the QConfig for an existing (module name, object type, index) was already set, the new QConfig will override the old one.

Return type: `QConfigMapping`

---

METHOD `set_module_name_regex(module_name_regex, qconfig)` [SOURCE]

Set the QConfig for modules matching the given regex string.

Regexes will be matched in the order in which they are registered through this method. Thus, the caller should register more specific patterns first, e.g.:

```python
qconfig_mapping = QConfigMapping()
    .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
    .set_module_name_regex("foo.*bar.*", qconfig2)
    .set_module_name_regex("foo.*", qconfig3)
```

In this example, `"foo.bar.conv0"` would match `qconfig1`, `"foo.bar.linear"` would match `qconfig2`, and `"foo.baz.relu"` would match `qconfig3`.

If the QConfig for an existing module name regex was already set, the new QConfig will override the old one while preserving the order in which the regexes were originally registered.

Return type: `QConfigMapping`

---

METHOD `set_object_type(object_type, qconfig)` [SOURCE]

Set the QConfig for a given module type, function, or method name. If the QConfig for an existing object type was already set, the new QConfig will override the old one.

Return type: `QConfigMapping`

---

METHOD `to_dict()` [SOURCE]

Convert this `QConfigMapping` to a dictionary with the following keys:

- `""` (for global QConfig)
- `"object_type"`
- `"module_name_regex"`
- `"module_name"`
- `"module_name_object_type_order"`

The values of this dictionary are lists of tuples.

Return type: `Dict[str, Any]`

## How to Write a Quantizer for PyTorch 2 Export Quantization

### Introduction - How to Write a Quantizer for PyTorch 2 Export Quantization

(prototype) PyTorch 2 Export Post Training Quantization introduced the overall API for pytorch 2 export quantization. The main difference from fx graph mode quantization in terms of API is that we made it explicit that quantiation is targeting a specific backend. So to use the new flow, backend need to implement a Quantizer class that encodes:

1. What is supported quantized operator or patterns in the backend
2. How can users express the way they want their floating point model to be quantized, for example, quantized the whole model to be int8 symmetric quantization, or quantize only linear layers etc.

An existing quantizer object defined for XNNPACK is in QNNPackQuantizer.

### Annotation API

Quantizer uses annotation API to convey quantization intent for different operators/patterns. Annotation API mainly consists of `QuantizationSpec` and `QuantizationAnnotation`.

`QuantizationSpec` is used to convey intent of how a tensor will be quantized, e.g. dtype, bitwidth, min, max values, symmetric vs. asymmetric etc. Furthermore, `QuantizationSpec` also allows quantizer to specify how a tensor value should be observed, e.g. `MinMaxObserver`, or `HistogramObserver`, or some customized observer.

`QuantizationAnnotation` composed of `QuantizationSpec` objects is used to annotate input tensors and output tensor of a pattern. Annotating input tensors is equivalent of annotating input edges, while annotating output tensor is equivalent of annotating node. `QuantizationAnnotation` is a dataclass with several fields:

- `input_qspec_map` field is of class `Dict` to map each input tensor (as input edge) to a `QuantizationSpec`.
- `output_qspec` field expresses the `QuantizationSpec` used to annotate the output tensor.
- `_annotated` field indicates if this node has already been annotated by quantizer.

To conclude, annotation API requires quantizer to annotate edges (input tensors) or nodes (output tensor) of the graph. Now, we will have a step-by-step tutorial for how to use the annotation API with different types of `QuantizationSpec`.

### 1. Annotate Common Operator Patterns

In order to use the quantized pattern/operators, e.g. quantized add, backend developers will have intent to quantize (as expressed by `QuantizationSpec`) inputs, output of the pattern. Following is an example flow (take add operator as example) of how this intent is conveyed in the quantization workflow with annotation API.

- Step 1: Identify the original floating point pattern in the FX graph.

There are several ways to identify this pattern: Quantizer may use a pattern matcher to match the operator pattern; Quantizer may go through the nodes from start to the end and compare the node’s target type to match the operator pattern. In this example, we can use the `get_source_partitions` to match this pattern. The original floating point add pattern only contain a single add node.

```python
add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add])
add_partitions = list(itertools.chain(*add_partitions.values()))
for add_partition in add_partitions:
    add_node = add_partition.output_nodes[0]
```

- Step 2: Define the `QuantizationSpec` for inputs and output of the pattern.

`QuantizationSpec` defines the data type, qscheme, and other quantization parameters about users’ intent of how to observe or fake quantize a tensor.

```python
act_quantization_spec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2-12),
)

input_act_qspec = act_quantization_spec
output_act_qspec = act_quantization_spec
```

- Step 3: Annotate the inputs and output of the pattern with `QuantizationAnnotation`.

In this example, we will create the `QuantizationAnnotation` object with the `QuantizationSpec` created in above step 2 for two inputs and one output of the add node.

```python
input_qspec_map = {}
input_act0 = add_node.args[0]
input_qspec_map[input_act0] = input_act_qspec

input_act1 = add_node.args[1]
input_qspec_map[input_act1] = input_act_qspec

add_node.meta["quantization_annotation"] = QuantizationAnnotation(
    input_qspec_map=input_qspec_map,
    output_qspec=output_act_qspec,
    _annotated=True,
)
```

After we annotate the add node like this, in the following up quantization flow, `HistogramObserver` will be inserted at its two input nodes and one output node in prepare phase. And `HistogramObserver` will be substituted with quantize node and dequantize node in the convert phase.

### 2. Annotate Operators that Shares Quantization Params

It is natural that users want to annotate a quantized model where quantization parameters can be shared among some tensors explicitly. Two typical use cases are:

- Example 1: One example is for add where having both inputs sharing quantization parameters makes operator implementation much easier. Without using of `SharedQuantizationSpec`, we must annotate add as example in above section 1, in which two inputs of add has different quantization parameters.
- Example 2: Another example is that of sharing quantization parameters between inputs and output. This typically results from operators such as maxpool, average_pool, concat etc.

`SharedQuantizationSpec` is designed for this use case to annotate tensors whose quantization parameters are shared with other tensors. Input of `SharedQuantizationSpec` is an `EdgeOrNode` object which can be an input edge or an output value.

NOTE:

- Sharing is transitive
- Some tensors might be effectively using shared quantization spec due to:
  - Two nodes/edges are configured to use `SharedQuantizationSpec`.
  - There is existing sharing of some nodes.

For example, let’s say we have two conv nodes conv1 and conv2, and both of them are fed into a cat node: `cat([conv1_out, conv2_out], ...)`. Let’s say the output of conv1, conv2, and the first input of cat are configured with the same configurations of `QuantizationSpec`. The second input of cat is configured to use `SharedQuantizationSpec` with the first input.

```python
conv1_out: qspec1(dtype=torch.int8, ...)
conv2_out: qspec1(dtype=torch.int8, ...)
cat_input0: qspec1(dtype=torch.int8, ...)
cat_input1: SharedQuantizationSpec((conv1, cat))  # conv1 node is the first input of cat
```

First of all, the output of conv1 is implicitly sharing quantization parameters (and observer object) with the first input of cat, and the same is true for the output of conv2 and the second input of cat. Therefore, since the user configures the two inputs of cat to share quantization parameters, by transitivity, conv2_out and conv1_out will also be sharing quantization parameters. In the observed graph, you will see the following:

```python
conv1 -> obs -> cat
conv2 -> obs   /
```

and both obs will be the same observer instance.

Input edge is the connection between input node and the node consuming the input, so it’s a `Tuple[Node, Node]`.

Output value is an FX Node.

Now, if we want to rewrite add annotation example with `SharedQuantizationSpec` to indicate two input tensors as sharing quantization parameters. We can define its `QuantizationAnnotation` as this:

- Step 1: Identify the original floating point pattern in the FX graph.

We can use the same methods introduced in `QuantizationSpec` example to identify the add pattern.

- Step 2: Annotate `input_act0` of add with `QuantizationSpec`.

- Step 3: Create a `SharedQuantizationSpec` object with input edge defined as `(input_act0, add_node)` which means to share the observer used for this edge. Then, user can annotate `input_act1` with this `SharedQuantizationSpec` object.

```python
input_qspec_map = {}
share_qparams_with_input_act0_qspec = SharedQuantizationSpec((input_act0, add_node))
input_qspec_map = {input_act0: act_quantization_spec, input_act1: share_qparams_with_input_act0_qspec}

add_node.meta["quantization_annotation"] = QuantizationAnnotation(
    input_qspec_map=input_qspec_map,
    output_qspec=act_quantization_spec,
    _annotated=True,
)
```

### 3. Annotate Operators with Fixed Quantization Parameters

Another typical use case to annotate a quantized model is for tensors whose quantization parameters are known beforehand. For example, operator like sigmoid, which has predefined and fixed scale/zero_point at input and output tensors. `FixedQParamsQuantizationSpec` is designed for this use case. To use `FixedQParamsQuantizationSpec`, users need to pass in parameters of scale and zero_point explicitly.

- Step 1: Identify the original floating point pattern in the FX graph.

We can use the same methods introduced in `QuantizationSpec` example to identify the sigmoid pattern.

- Step 2: Create `FixedQParamsQuantizationSpec` object with inputs of fixed scale, zero_point value.

These values will be used to create the quantize node and dequantize node in the convert phase.

- Step 3: Annotate inputs and output to use this `FixedQParamsQuantizationSpec` object.

```python
act_qspec = FixedQParamsQuantizationSpec(
    dtype=torch.uint8,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
    scale=1.0 / 256.0,
    zero_point=0,
)
sigmoid_node.meta["quantization_annotation"] = QuantizationAnnotation(
    input_qspec_map={input_act: act_qspec},
    output_qspec=act_qspec,
    _annotated=True,
)
```

### 4. Annotate Tensors with Derived Quantization Parameters

Another use case is to define the constraint for tensors whose quantization parameters are derived from other tensors. For example, if we want to annotate a convolution node, and define the scale of its bias input tensor as product of the activation tensor’s scale and weight tensor’s scale. We can use `DerivedQuantizationSpec` to annotate this conv node.

- Step 1: Identify the original floating point pattern in the FX graph.

We can use the same methods introduced in `QuantizationSpec` example to identify the convolution pattern.

- Step 2: Define `derive_qparams_fn` function, it accepts list of `ObserverOrFakeQuantize` (`ObserverBase` or `FakeQuantizeBase`) as input.

From each `ObserverOrFakeQuantize` object, user can get the scale, zero point value. User can define its heuristic about how to derive new scale, zero point value based on the quantization parameters calculated from the observer or fake quant instances.

- Step 3: Define `DerivedQuantizationSpec` obejct, it accepts inputs of:

- list of `EdgeOrNode` objects. The observer corresponding to each `EdgeOrNode` object will be passed into the `derive_qparams_fn` function.
- `derive_qparams_fn` function.
- several other quantization parameters such as dtype, qscheme.

- Step 4: Annotate the inputs and output of this conv node with `QuantizationAnnotation`.

```python
def derive_qparams_fn(obs_or_fqs: List[ObserverOrFakeQuantize]) -> Tuple[Tensor, Tensor]:
    assert len(obs_or_fqs) == 2, \
        "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(len(obs_or_fq))
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale, act_zp = act_obs_or_fq.calculate_qparams()
    weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
    return torch.tensor([act_scale * weight_scale]).to(torch.float32), torch.tensor([0]).to(torch.int32)

bias_qspec = DerivedQuantizationSpec(
    derived_from=[(input_act, node), (weight, node)],
    derive_qparams_fn=derive_qparams_fn,
    dtype=torch.int32,
    quant_min=-231,
    quant_max=231 - 1,
    qscheme=torch.per_tensor_symmetric,
)
input_qspec_map = {input_act: act_quantization_spec, weight: weight_quantization_spec, bias: bias_qspec}
node.meta["quantization_annotation"] = QuantizationAnnotation(
    input_qspec_map=input_qspec_map,
    output_qspec=act_quantization_spec,
    _annotated=True,
)
```

### 5. A Toy Example with Resnet18

After above annotation methods defined with `QuantizationAnnotation` API, we can now put them together to construct a `BackendQuantizer` and run a toy example with Torchvision Resnet18. To better understand the final example, here are the classes and utility functions that are used in the example:

- `QuantizationConfig` consists of `QuantizationSpec` for activation, weight, and bias separately.
- When annotating the model, `get_input_act_qspec`, `get_output_act_qspec`, `get_weight_qspec`, and `get_bias_qspec` can be used to get the `QuantizationSpec` from `QuantizationConfig` for a specific pattern.

### A Note on IR for PT2E Quantization Flow

IR means the intermediate representation of the model, for example, torch IR (`torch.nn` modules, `torch.nn.functional` ops) or aten IR (`torch.ops.aten.linear`, …). PT2E Quantization Flow is using pre autograd aten IR (the output of `torch.export` API) so that we support training. As is shown before, we need to match the operator or operator patterns before we can attach annotations on them, So the question is how do we match the pattern?

#### Motivation: Problem of Matching aten IR directly

The most straightforward way might be matching aten IR directly.

Example:

```python
for n in gm.graph.nodes:
      if n.op != "call_function" or n.target not in [
          torch.ops.aten.relu.default,
          torch.ops.aten.relu_.default,
      ]:
          continue
      relu_node = n
      maybe_conv_node = n.args[0]
      if (
          not isinstance(maybe_conv_node, Node)
          or maybe_conv_node.op != "call_function"
          or maybe_conv_node.target
          not in [
              torch.ops.aten.conv1d.default,
              torch.ops.aten.conv2d.default,
          ]
      ):
          continue

      # annotate conv and relu nodes
      ...
```

However one problem for using this IR is that the representation might change if the PyTorch implementation for modules or functional ops changed. But this could be unexpected since modeling users typically assume that when the eager mode model code doesn’t change, they should get the same model representation after program capture as well. One concrete effect for this problem is that if a Quantizer do annotations based on recognizing aten IR patterns, then it may fail to recognizing the pattern after PyTorch version update, and the same eager mode floating point may be left unquantized.

#### Recommendation: Use `SubgraphMatcherWithNameNodeMap` for pattern matching

Because of this, we recommend people to recognize the pattern through `SubgraphMatcherWithNameNodeMap` (an improved version of `SubgraphMatcher` that makes it easier to query the nodes that people want to annotate), through capturing a torch IR pattern (with the same program capture used for capturing the floating point model), instead of using the aten IR pattern directly.

Example:

```python
def conv_relu_pattern(input, weight, bias):
    conv = torch.nn.functional.conv2d(input, weight, bias)
    output = torch.nn.functional.relu(conv)
    # returns an additional dict that includes a map from name to node that we want to annotate
    return relu, {"input": input, "weight": weight, "bias": bias, "output": output}

matcher = SubgraphMatcherWithNameNodeMap(conv_relu_pattern)
matches = matcher.match(model)
for match in matches:
    # find input and output of the pattern
    # annotate the nodes
    name_node_map = match.name_node_map
    input_node = name_node_map["input"]
    weight_node = name_node_map["weight"]
    bias_node = name_node_map["bias"]
    output_node = name_node_map["relu"]
    input_node.users[0].meta["quantization_annotation"] = ...
    weight_node.users[0].meta["quantization_annotation"] = ...
    bias_node.users[0].meta["quantization_annotation"] = ...
    output_node.meta["quantization_annotation"] = ...
```

With this, the Quantizer will still be valid even when the implementation for nn modules and functionals changes, the aten IR for floating point model will change, but since we capture the pattern again instead of hardcoding the aten IR for the pattern, we’ll get the updated aten IR as well and will still be able to match the pattern.

One caveat is that if inputs of the pattern has multiple users, we don’t have a good way to identify which user node we want to annotate except for checking the aten op target.

Another caveat is that we need to make sure we have an exhaustive list of examples (e.g. 2D, 3D, 4D inputs, real v.s. symbolic inputs, training=True v.s. training=False etc.) for the pattern to make sure cover different possible aten IR outcomes captured from the torch IR pattern.

Note: We may provide some (pattern, list of example_inputs) or some pre-generated matcher object so people can just use them directly in the future.

### Conclusion - How to Write a Quantizer for PyTorch 2 Export Quantization

With this tutorial, we introduce the new quantization path in PyTorch 2. Users can learn about how to define a `BackendQuantizer` with the `QuantizationAnnotation` API and integrate it into the PyTorch 2 Export Quantization flow. Examples of `QuantizationSpec`, `SharedQuantizationSpec`, `FixedQParamsQuantizationSpec`, and `DerivedQuantizationSpec` are given for specific annotation use case. You can use `XNNPACKQuantizer` as an example to start implementing your own Quantizer. After that please follow this tutorial to actually quantize your model.

## PyTorch 2 Export Quantization with X86 Backend through Inductor

### Introduction (Export Quantization with X86 Backend through Inductor)

This tutorial introduces the steps for utilizing the PyTorch 2 Export Quantization flow to generate a quantized model customized for the x86 inductor backend and explains how to lower the quantized model into the inductor.

The PyTorch 2 export quantization flow uses `torch.export` to capture the model into a graph and perform quantization transformations on top of the ATen graph. This approach is expected to have significantly higher model coverage, better programmability, and a simplified UX. TorchInductor is the new compiler backend that compiles the FX Graphs generated by TorchDynamo into optimized C++/Triton kernels.

This flow of quantization 2 with Inductor supports both static and dynamic quantization. Static quantization works best for CNN models, like ResNet-50. And dynamic quantization is more suitable for NLP models, like RNN and BERT. For the difference between the two quantization types, please refer to the following page.

The quantization flow mainly includes three steps:

1. Capture the FX Graph from the eager Model based on the torch export mechanism.
2. Apply the Quantization flow based on the captured FX Graph, including defining the backend-specific quantizer, generating the prepared model with observers, performing the prepared model’s calibration or quantization-aware training, and converting the prepared model into the quantized model.
3. Lower the quantized model into inductor with the API `torch.compile`.

The high-level architecture of this flow could look like this:

```text
float_model(Python)                          Example Input
    \                                              /
     \                                            /
—--------------------------------------------------------
|                         export                       |
—--------------------------------------------------------
                            |
                    FX Graph in ATen
                            |            X86InductorQuantizer
                            |                 /
—--------------------------------------------------------
|                      prepare_pt2e                     |
|                           |                           |
|                     Calibrate/Train                   |
|                           |                           |
|                      convert_pt2e                     |
—--------------------------------------------------------
                            |
                     Quantized Model
                            |
—--------------------------------------------------------
|                    Lower into Inductor                |
—--------------------------------------------------------
                            |
                         Inductor
```

Combining Quantization in PyTorch 2 Export and TorchInductor, we have flexibility and productivity with the new Quantization frontend and outstanding out-of-box performance with the compiler backend. Especially on Intel fourth generation (SPR) Xeon processors which can further boost the models’ performance by leveraging the advanced-matrix-extensions feature.

### Post Training Quantization

Now, we will walk you through a step-by-step tutorial for how to use it with torchvision resnet18 model for post training quantization.

#### 1. Capture FX Graph

We will start by performing the necessary imports, capturing the FX Graph from the eager module.

```python
import torch
import torchvision.models as models
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._export import capture_pre_autograd_graph

# Create the Eager Model
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)

# Set the model to eval mode
model = model.eval()

# Create the data, using the dummy data here as an example
traced_bs = 50
x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
example_inputs = (x,)

# Capture the FX Graph to be quantized
with torch.no_grad():
    # if you are using the PyTorch nightlies or building from source with the pytorch master,
    # use the API of `capture_pre_autograd_graph`
    # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
    exported_model = capture_pre_autograd_graph(
        model,
        example_inputs
    )
    # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
    # please use the API of `torch._dynamo.export` to capture the FX Graph.
    # exported_model, guards = torch._dynamo.export(
    #     model,
    #     *copy.deepcopy(example_inputs),
    #     aten_graph=True,
    # )
```

Next, we will have the FX Module to be quantized.

#### 2. Apply Quantization

After we capture the FX Module to be quantized, we will import the Backend Quantizer for X86 CPU and configure how to quantize the model.

```python
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
```

NOTE:

The default quantization configuration in `X86InductorQuantizer` uses 8-bits for both activations and weights.

When Vector Neural Network Instruction is not available, the oneDNN backend silently chooses kernels that assume multiplications are 7-bit x 8-bit. In other words, potential numeric saturation and accuracy issue may happen when running on CPU without Vector Neural Network Instruction.

The quantization config is for static quantization by default. To apply dynamic quantization, add an argument `is_dynamic=True` when getting the config.

```python
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_dynamic=True))
```

After we import the backend-specific Quantizer, we will prepare the model for post-training quantization. `prepare_pt2e` folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.

```python
prepared_model = prepare_pt2e(exported_model, quantizer)
```

Now, we will calibrate the `prepared_model` after the observers are inserted in the model. This step is needed for static quantization only.

```python
# We use the dummy data as an example here
prepared_model(*example_inputs)

# Alternatively: user can define the dataset to calibrate
# def calibrate(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for image, target in data_loader:
#             model(image)
# calibrate(prepared_model, data_loader_test)  # run calibration on sample data
```

Finally, we will convert the calibrated Model to a quantized Model. `convert_pt2e` takes a calibrated model and produces a quantized model.

```python
converted_model = convert_pt2e(prepared_model)
```

After these steps, we finished running the quantization flow and we will get the quantized model.

#### 3. Lower into Inductor

After we get the quantized model, we will further lower it to the inductor backend. The default Inductor wrapper generates Python code to invoke both generated kernels and external kernels. Additionally, Inductor supports C++ wrapper that generates pure C++ code. This allows seamless integration of the generated and external kernels, effectively reducing Python overhead. In the future, leveraging the C++ wrapper, we can extend the capability to achieve pure C++ deployment. For more comprehensive details about C++ Wrapper in general, please refer to the dedicated tutorial on Inductor C++ Wrapper Tutorial.

```python
# Optional: using the C++ wrapper instead of default Python wrapper
import torch._inductor.config as config
config.cpp_wrapper = True
with torch.no_grad():
    optimized_model = torch.compile(converted_model)

    # Running some benchmark
    optimized_model(*example_inputs)
```

In a more advanced scenario, int8-mixed-bf16 quantization comes into play. In this instance, a Convolution or GEMM operator produces BFloat16 output data type instead of Float32 in the absence of a subsequent quantization node. Subsequently, the BFloat16 tensor seamlessly propagates through subsequent pointwise operators, effectively minimizing memory usage and potentially enhancing performance. The utilization of this feature mirrors that of regular BFloat16 Autocast, as simple as wrapping the script within the BFloat16 Autocast context.

```python
with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True), torch.no_grad():
    # Turn on Autocast to use int8-mixed-bf16 quantization. After lowering into Inductor CPP Backend,
    # For operators such as QConvolution and QLinear:
    # * The input data type is consistently defined as int8, attributable to the presence of a pair
        of quantization and dequantization nodes inserted at the input.
    # * The computation precision remains at int8.
    # * The output data type may vary, being either int8 or BFloat16, contingent on the presence
    #   of a pair of quantization and dequantization nodes at the output.
    # For non-quantizable pointwise operators, the data type will be inherited from the previous node,
    # potentially resulting in a data type of BFloat16 in this scenario.
    # For quantizable pointwise operators such as QMaxpool2D, it continues to operate with the int8
    # data type for both input and output.
    optimized_model = torch.compile(converted_model)

    # Running some benchmark
    optimized_model(*example_inputs)
```

Put all these codes together, we will have the toy example code. Please note that since the Inductor freeze feature does not turn on by default yet, run your example code with `TORCHINDUCTOR_FREEZING=1`.

For example:

```bash
TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_pytorch_2_1.py
```

With PyTorch 2.1 release, all CNN models from TorchBench test suite have been measured and proven effective comparing with Inductor FP32 inference path. Please refer to this document for detail benchmark number.

### Quantization Aware Training

The PyTorch 2 Export Quantization-Aware Training (QAT) is now supported on X86 CPU using `X86InductorQuantizer`, followed by the subsequent lowering of the quantized model into Inductor. For a more in-depth understanding of PT2 Export Quantization-Aware Training, we recommend referring to the dedicated PyTorch 2 Export Quantization-Aware Training.

The PyTorch 2 Export QAT flow is largely similar to the PTQ flow:

```python
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(1024, 1000)

   def forward(self, x):
      return self.linear(x)

example_inputs = (torch.randn(1, 1024),)
m = M()

# Step 1. program capture
# NOTE: this API will be updated to torch.export API in the future, but the captured
# result shoud mostly stay the same
exported_model = capture_pre_autograd_graph(m, example_inputs)
# we get a model with aten ops

# Step 2. quantization-aware training
# Use Backend Quantizer for X86 CPU
# To apply dynamic quantization, add an argument ``is_dynamic=True`` when getting the config.
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

# train omitted

converted_model = convert_pt2e(prepared_model)
# we have a model with aten ops doing integer computations when possible

# move the quantized model to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(converted_model)

# Lower the model into Inductor
with torch.no_grad():
  optimized_model = torch.compile(converted_model)
  _ = optimized_model(*example_inputs)
```

Please note that the Inductor freeze feature is not enabled by default. To use this feature, you need to run example code with `TORCHINDUCTOR_FREEZING=1`.

For example:

```bash
TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_qat.py
```

### Conclusion (Export Quantization with X86 Backend through Inductor)

With this tutorial, we introduce how to use Inductor with X86 CPU in PyTorch 2 Quantization. Users can learn about how to use `X86InductorQuantizer` to quantize a model and lower it into the inductor with X86 CPU devices.
