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

CODED

## PyTorch 2 Export Quantization-Aware Training (QAT)

## PyTorch BackendConfig

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
