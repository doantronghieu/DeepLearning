# TorchScript

## Introduction to TorchScript

This tutorial is an introduction to TorchScript, an intermediate representation of a PyTorch model (subclass of `nn.Module`) that can then be run in a high-performance environment such as C++.

```python
import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility
```

```plaintext
2.4.0+cu124
<torch._C.Generator object at 0x7fbad7d77db0>
```

### Basics of PyTorch Model Authoring

Let’s start out by defining a simple Module. A Module is the basic unit of composition in PyTorch. It contains:

- A constructor, which prepares the module for invocation
- A set of Parameters and sub-Modules. These are initialized by the constructor and can be used by the module during invocation.
- A forward function. This is the code that is run when the module is invoked.

Let’s examine a small example:

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))
```

So we’ve:

- Created a class that subclasses `torch.nn.Module`.
- Defined a constructor. The constructor doesn’t do much, just calls the constructor for super.
- Defined a forward function, which takes two inputs and returns two outputs. The actual contents of the forward function are not really important, but it’s sort of a fake RNN cell–that is–it’s a function that is applied on a loop.
- We instantiated the module, and made `x` and `h`, which are just 3x4 matrices of random values. Then we invoked the cell with `my_cell(x, h)`. This in turn calls our forward function.

Let’s do something a little more interesting:

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

We’ve redefined our module `MyCell`, but this time we’ve added a `self.linear` attribute, and we invoke `self.linear` in the forward function.

What exactly is happening here? `torch.nn.Linear` is a Module from the PyTorch standard library. Just like `MyCell`, it can be invoked using the call syntax. We are building a hierarchy of Modules.

Printing on a Module will give a visual representation of the Module’s subclass hierarchy. In our example, we can see our `Linear` subclass and its parameters.

By composing Modules in this way, we can succinctly and readably author models with reusable components.

You may have noticed `grad_fn` on the outputs. This is a detail of PyTorch’s method of automatic differentiation, called autograd. In short, this system allows us to compute derivatives through potentially complex programs. The design allows for a massive amount of flexibility in model authoring.

Now let’s examine said flexibility:

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

We’ve once again redefined our `MyCell` class, but here we’ve defined `MyDecisionGate`. This module utilizes control flow. Control flow consists of things like loops and if-statements.

Many frameworks take the approach of computing symbolic derivatives given a full program representation. However, in PyTorch, we use a gradient tape. We record operations as they occur, and replay them backwards in computing derivatives. In this way, the framework does not have to explicitly define derivatives for all constructs in the language.

How autograd works

### Basics of TorchScript

Now let’s take our running example and see how we can apply TorchScript.

In short, TorchScript provides tools to capture the definition of your model, even in light of the flexible and dynamic nature of PyTorch. Let’s begin by examining what we call tracing.

#### Tracing Modules

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)
```

We’ve rewinded a bit and taken the second version of our `MyCell` class. As before, we’ve instantiated it, but this time, we’ve called `torch.jit.trace`, passed in the Module, and passed in example inputs the network might see.

What exactly has this done? It has invoked the Module, recorded the operations that occurred when the Module was run, and created an instance of `torch.jit.ScriptModule` (of which `TracedModule` is an instance).

TorchScript records its definitions in an Intermediate Representation (or IR), commonly referred to in Deep learning as a graph. We can examine the graph with the `.graph` property:

```python
print(traced_cell.graph)
```

However, this is a very low-level representation and most of the information contained in the graph is not useful for end users. Instead, we can use the `.code` property to give a Python-syntax interpretation of the code:

```python
print(traced_cell.code)
```

So why did we do all this? There are several reasons:

- TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the same instance simultaneously.
- This format allows us to save the whole model to disk and load it into another environment, such as in a server written in a language other than Python.
- TorchScript gives us a representation in which we can do compiler optimizations on the code to provide more efficient execution.
- TorchScript allows us to interface with many backend/device runtimes that require a broader view of the program than individual operators.

We can see that invoking `traced_cell` produces the same results as the Python module:

```python
print(my_cell(x, h))
print(traced_cell(x, h))
```

### Using Scripting to Convert Modules

There’s a reason we used version two of our module, and not the one with the control-flow-laden submodule. Let’s examine that now:

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)
```

Looking at the `.code` output, we can see that the if-else branch is nowhere to be found! Why? Tracing does exactly what we said it would: run the code, record the operations that happen and construct a `ScriptModule` that does exactly that. Unfortunately, things like control flow are erased.

How can we faithfully represent this module in TorchScript? We provide a script compiler, which does direct analysis of your Python source code to transform it into TorchScript. Let’s convert `MyDecisionGate` using the script compiler:

```python
scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)
```

Hooray! We’ve now faithfully captured the behavior of our program in TorchScript. Let’s now try running the program:

```python
# New inputs
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(scripted_cell(x, h))
```

```plaintext
(tensor([[ 0.5679,  0.5762,  0.2506, -0.0734],
        [ 0.5228,  0.7122,  0.6985, -0.0656],
        [ 0.6187,  0.4487,  0.7456, -0.0238]], grad_fn=<TanhBackward0>), tensor([[ 0.5679,  0.5762,  0.2506, -0.0734],
        [ 0.5228,  0.7122,  0.6985, -0.0656],
        [ 0.6187,  0.4487,  0.7456, -0.0238]], grad_fn=<TanhBackward0>))
```

### Mixing Scripting and Tracing

Some situations call for using tracing rather than scripting (e.g. a module has many architectural decisions that are made based on constant Python values that we would like to not appear in TorchScript). In this case, scripting can be composed with tracing: `torch.jit.script` will inline the code for a traced module, and tracing will inline the code for a scripted module.

An example of the first case:

```python
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
```

And an example of the second case:

```python
class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)
```

This way, scripting and tracing can be used when the situation calls for each of them and used together.

### Saving and Loading models

We provide APIs to save and load TorchScript modules to/from disk in an archive format. This format includes code, parameters, attributes, and debug information, meaning that the archive is a freestanding representation of the model that can be loaded in an entirely separate process. Let’s save and load our wrapped RNN module:

```python
traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)
```

As you can see, serialization preserves the module hierarchy and the code we’ve been examining throughout. The model can also be loaded, for example, into C++ for python-free execution.

## Loading a TorchScript Model in C++

As its name suggests, the primary interface to PyTorch is the Python programming language. While Python is a suitable and preferred language for many scenarios requiring dynamism and ease of iteration, there are equally many situations where precisely these properties of Python are unfavorable. One environment in which the latter often applies is production – the land of low latencies and strict deployment requirements. For production scenarios, C++ is very often the language of choice, even if only to bind it into another language like Java, Rust, or Go. The following paragraphs will outline the path PyTorch provides to go from an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.

### Step 1: Converting Your PyTorch Model to Torch Script

A PyTorch model’s journey from Python to C++ is enabled by Torch Script, a representation of a PyTorch model that can be understood, compiled, and serialized by the Torch Script compiler. If you are starting out from an existing PyTorch model written in the vanilla “eager” API, you must first convert your model to Torch Script. In the most common cases, discussed below, this requires only little effort. If you already have a Torch Script module, you can skip to the next section of this tutorial.

There exist two ways of converting a PyTorch model to Torch Script. The first is known as tracing, a mechanism in which the structure of the model is captured by evaluating it once using example inputs, and recording the flow of those inputs through the model. This is suitable for models that make limited use of control flow. The second approach is to add explicit annotations to your model that inform the Torch Script compiler that it may directly parse and compile your model code, subject to the constraints imposed by the Torch Script language.

TIP

You can find the complete documentation for both of these methods, as well as further guidance on which to use, in the official Torch Script reference.

#### Converting to Torch Script via Tracing

To convert a PyTorch model to Torch Script via tracing, you must pass an instance of your model along with an example input to the `torch.jit.trace` function. This will produce a `torch.jit.ScriptModule` object with the trace of your model evaluation embedded in the module’s forward method:

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
```

The traced `ScriptModule` can now be evaluated identically to a regular PyTorch module:

```python
In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
In[2]: output[0, :5]
Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

#### Converting to Torch Script via Annotation

Under certain circumstances, such as if your model employs particular forms of control flow, you may want to write your model in Torch Script directly and annotate your model accordingly. For example, say you have the following vanilla PyTorch model:

```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output
```

Because the forward method of this module uses control flow that is dependent on the input, it is not suitable for tracing. Instead, we can convert it to a `ScriptModule`. In order to convert the module to the `ScriptModule`, one needs to compile the module with `torch.jit.script` as follows:

```python
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10, 20)
sm = torch.jit.script(my_module)
```

If you need to exclude some methods in your `nn.Module` because they use Python features that TorchScript doesn’t support yet, you could annotate those with `@torch.jit.ignore`.

`sm` is an instance of `ScriptModule` that is ready for serialization.

### Step 2: Serializing Your Script Module to a File

Once you have a `ScriptModule` in your hands, either from tracing or annotating a PyTorch model, you are ready to serialize it to a file. Later on, you’ll be able to load the module from this file in C++ and execute it without any dependency on Python. Say we want to serialize the ResNet18 model shown earlier in the tracing example. To perform this serialization, simply call `save` on the module and pass it a filename:

```python
traced_script_module.save("traced_resnet_model.pt")
```

This will produce a `traced_resnet_model.pt` file in your working directory. If you also would like to serialize `sm`, call `sm.save("my_module_model.pt")`. We have now officially left the realm of Python and are ready to cross over to the sphere of C++.

### Step 3: Loading Your Script Module in C++

To load your serialized PyTorch model in C++, your application must depend on the PyTorch C++ API – also known as LibTorch. The LibTorch distribution encompasses a collection of shared libraries, header files, and CMake build configuration files. While CMake is not a requirement for depending on LibTorch, it is the recommended approach and will be well supported into the future. For this tutorial, we will be building a minimal C++ application using CMake and LibTorch that simply loads and executes a serialized PyTorch model.

#### A Minimal C++ Application

Let’s begin by discussing the code to load a module. The following will already do:

```cpp
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}
```

The `<torch/script.h>` header encompasses all relevant includes from the LibTorch library necessary to run the example. Our application accepts the file path to a serialized PyTorch `ScriptModule` as its only command line argument and then proceeds to deserialize the module using the `torch::jit::load()` function, which takes this file path as input. In return, we receive a `torch::jit::script::Module` object. We will examine how to execute it in a moment.

#### Depending on LibTorch and Building the Application

Assume we stored the above code into a file called `example-app.cpp`. A minimal `CMakeLists.txt` to build it could look as simple as:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
```

The last thing we need to build the example application is the LibTorch distribution. You can always grab the latest stable release from the download page on the PyTorch website. If you download and unzip the latest archive, you should receive a folder with the following directory structure:

```text
libtorch/
  bin/
  include/
  lib/
  share/
```

- The `lib/` folder contains the shared libraries you must link against.
- The `include/` folder contains header files your program will need to include.
- The `share/` folder contains the necessary CMake configuration to enable the simple `find_package(Torch)` command above.

TIP

On Windows, debug and release builds are not ABI-compatible. If you plan to build your project in debug mode, please try the debug version of LibTorch. Also, make sure you specify the correct configuration in the `cmake --build .` line below.

The last step is building the application. For this, assume our example directory is laid out like this:

```text
example-app/
  CMakeLists.txt
  example-app.cpp
```

We can now run the following commands to build the application from within the `example-app/` folder:

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

where `/path/to/libtorch` should be the full path to the unzipped LibTorch distribution. If all goes well, it will look something like this:

```sh
root@4b5a67132e81:/example-app# mkdir build
root@4b5a67132e81:/example-app# cd build
root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring done
-- Generating done
-- Build files have been written to: /example-app/build
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
```

If we supply the path to the traced ResNet18 model `traced_resnet_model.pt` we created earlier to the resulting `example-app` binary, we should be rewarded with a friendly “ok”. Please note, if you try to run this example with `my_module_model.pt`, you will get an error saying that your input is of an incompatible shape. `my_module_model.pt` expects 1D instead of 4D.

```sh
root@4b5a67132e81:/example-app/build# ./example-app <path_to_model>/traced_resnet_model.pt
ok
```

### Step 4: Executing the Script Module in C++

Having successfully loaded our serialized ResNet18 in C++, we are now just a couple lines of code away from executing it! Let’s add those lines to our C++ application’s `main()` function:

```cpp
// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
```

The first two lines set up the inputs to our model. We create a vector of `torch::jit::IValue` (a type-erased value type `script::Module` methods accept and return) and add a single input. To create the input tensor, we use `torch::ones()`, the equivalent to `torch.ones` in the C++ API. We then run the `script::Module`’s forward method, passing it the input vector we created. In return, we get a new `IValue`, which we convert to a tensor by calling `toTensor()`.

TIP

To learn more about functions like `torch::ones` and the PyTorch C++ API in general, refer to its documentation at [https://pytorch.org/cppdocs](https://pytorch.org/cppdocs). The PyTorch C++ API provides near feature parity with the Python API, allowing you to further manipulate and process tensors just like in Python.

In the last line, we print the first five entries of the output. Since we supplied the same input to our model in Python earlier in this tutorial, we should ideally see the same output. Let’s try it out by re-compiling our application and running it with the same serialized model:

```sh
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
root@4b5a67132e81:/example-app/build# ./example-app traced_resnet_model.pt
-0.2698 -0.0381  0.4023 -0.3010 -0.0448
[ Variable[CPUFloatType]{1,5} ]
```

For reference, the output in Python previously was:

```python
tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

Looks like a good match!

TIP

To move your model to GPU memory, you can write `model.to(at::kCUDA);`. Make sure the inputs to a model are also living in CUDA memory by calling `tensor.to(at::kCUDA)`, which will return a new tensor in CUDA memory.

### Step 5: Getting Help and Exploring the API

This tutorial has hopefully equipped you with a general understanding of a PyTorch model’s path from Python to C++. With the concepts described in this tutorial, you should be able to go from a vanilla, “eager” PyTorch model, to a compiled `ScriptModule` in Python, to a serialized file on disk and – to close the loop – to an executable `script::Module` in C++.

Of course, there are many concepts we did not cover. For example, you may find yourself wanting to extend your `ScriptModule` with a custom operator implemented in C++ or CUDA, and executing this custom operator inside your `ScriptModule` loaded in your pure C++ production environment. The good news is: this is possible, and well supported! For now, you can explore this folder for examples, and we will follow up with a tutorial shortly. In the time being, the following links may be generally helpful:

- The Torch Script reference: [https://pytorch.org/docs/master/jit.html](https://pytorch.org/docs/master/jit.html)
- The PyTorch C++ API documentation: [https://pytorch.org/cppdocs/](https://pytorch.org/cppdocs/)
- The PyTorch Python API documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)

As always, if you run into any problems or have questions, you can use our forum or GitHub issues to get in touch.

## TorchScript for Deployment

### Requirements

- PyTorch 1.5
- TorchVision 0.6.0
- libtorch 1.5
- C++ compiler

### What is TorchScript?

TorchScript is an intermediate representation of a PyTorch model (subclass of `nn.Module`) that can then be run in a high-performance environment like C++. It’s a high-performance subset of Python that is meant to be consumed by the PyTorch JIT Compiler, which performs run-time optimization on your model’s computation. TorchScript is the recommended model format for doing scaled inference with PyTorch models. For more information, see the [PyTorch Introduction to TorchScript tutorial](https://pytorch.org/tutorials/advanced/torchscript_tutorial.html), the [Loading A TorchScript Model in C++ tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html), and the full [TorchScript documentation](https://pytorch.org/docs/stable/jit.html), all of which are available on [pytorch.org](https://pytorch.org).

### How to Export Your Model

As an example, let’s take a pretrained vision model. All of the pretrained models in TorchVision are compatible with TorchScript.

Run the following Python 3 code, either in a script or from the REPL:

```python
import torch
import torch.nn.functional as F
import torchvision.models as models

r18 = models.resnet18(pretrained=True)       # We now have an instance of the pretrained model
r18_scripted = torch.jit.script(r18)         # * This is the TorchScript export
dummy_input = torch.rand(1, 3, 224, 224)     # We should run a quick test
```

Let’s do a sanity check on the equivalence of the two models:

```python
unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Python model top 5 results:\n  {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))
```

You should see that both versions of the model give the same results:

```text
Python model top 5 results:
  tensor([[463, 600, 731, 899, 898]])
TorchScript model top 5 results:
  tensor([[463, 600, 731, 899, 898]])
```

With that check confirmed, go ahead and save the model:

```python
r18_scripted.save('r18_scripted.pt')
```

### Loading TorchScript Models in C++

Create the following C++ file and name it `ts-infer.cpp`:

```cpp
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ts-infer <path-to-exported-model>\n";
        return -1;
    }

    std::cout << "Loading model...\n";

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        std::cerr << e.msg_without_backtrace();
        return -1;
    }

    std::cout << "Model loaded successfully\n";

    torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval(); // turn off dropout and other training-time layers/functions

    // create an input "image"
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}));

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
    at::Tensor top5 = std::get<1>(top5_tensor);

    std::cout << top5[0] << "\n";

    std::cout << "\nDONE\n";
    return 0;
}
```

This program:

- Loads the model you specify on the command line
- Creates a dummy “image” input tensor
- Performs inference on the input

Also, notice that there is no dependency on TorchVision in this code. The saved version of your TorchScript model has your learning weights and your computation graph - nothing else is needed.

### Building and Running Your C++ Inference Engine

Create the following `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(ts-infer ts-infer.cpp)
target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)
```

Make the program:

```sh
cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
make
```

Now, we can run inference in C++, and verify that we get a result:

```sh
$ ./ts-infer r18_scripted.pt
Loading model...
Model loaded successfully
 418
 845
 111
 892
 644
[ CPULongType{5} ]

DONE
```

## [Model Freezing in TorchScript](https://pytorch.org/tutorials/prototype/torchscript_freezing.html)

In this tutorial, we introduce the syntax for model freezing in TorchScript. Freezing is the process of inlining PyTorch module parameters and attributes values into the TorchScript internal representation. Parameter and attribute values are treated as final values and they cannot be modified in the resulting Frozen module.

### Basic Syntax

Model freezing can be invoked using the API below:

```python
torch.jit.freeze(mod: ScriptModule, names: str[]) -> ScriptModule
```

Note that the input module can either be the result of scripting or tracing. See [Intro to TorchScript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) for more details.

### Example

Next, we demonstrate how freezing works using an example:

```python
import torch, time

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

    @torch.jit.export
    def version(self):
        return 1.0

net = torch.jit.script(Net())
fnet = torch.jit.freeze(net)

print(net.conv1.weight.size())
print(net.conv1.bias)

try:
    print(fnet.conv1.bias)
    # without exception handling, prints:
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
    # with name 'conv1'
except RuntimeError:
    print("field 'conv1' is inlined. It does not exist in 'fnet'")

try:
    fnet.version()
    # without exception handling, prints:
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
    # with name 'version'
except RuntimeError:
    print("method 'version' is not deleted in fnet. Only 'forward' is preserved")

fnet2 = torch.jit.freeze(net, ["version"])

print(fnet2.version())

B = 1
warmup = 1
iter = 1000
input = torch.rand(B, 1, 28, 28)

start = time.time()
for i in range(warmup):
    net(input)
end = time.time()
print("Scripted - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(warmup):
    fnet(input)
end = time.time()
print("Frozen   - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1, 28, 28)
    net(input)
end = time.time()
print("Scripted - Inference: {0:5.2f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1, 28, 28)
    fnet2(input)
end = time.time()
print("Frozen    - Inference time: {0:5.2f}".format(end-start), flush=True)
```

In our example, warm up time measures the first two runs. The frozen model is 50% faster than the scripted model. On some more complex models, we observed even higher speed up of warm up time. Freezing achieves this speed up because it is doing some of the work TorchScript has to do when the first couple runs are initiated.

Inference time measures inference execution time after the model is warmed up. Although we observed significant variation in execution time, the frozen model is often about 15% faster than the scripted model. When input is larger, we observe a smaller speed up because the execution is dominated by tensor operations.

### Conclusion

In this tutorial, we learned about model freezing. Freezing is a useful technique to optimize models for inference and it also can significantly reduce TorchScript warmup time.

## [torch.jit](https://pytorch.org/docs/master/jit.html)

## [Deploying with Flask](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)

### What is Flask?

Flask is a lightweight web server written in Python. It provides a convenient way for you to quickly set up a web API for predictions from your trained PyTorch model, either for direct use, or as a web service within a larger system.

### Setup and Supporting Files

We’re going to create a web service that takes in images, and maps them to one of the 1000 classes of the ImageNet dataset. To do this, you’ll need an image file for testing. Optionally, you can also get a file that will map the class index output by the model to a human-readable class name.

### Option 1: To Get Both Files Quickly

You can pull both of the supporting files quickly by checking out the TorchServe repository and copying them to your working folder. (NB: There is no dependency on TorchServe for this tutorial - it’s just a quick way to get the files.) Issue the following commands from your shell prompt:

```sh
git clone https://github.com/pytorch/serve
cp serve/examples/image_classifier/kitten.jpg .
cp serve/examples/image_classifier/index_to_name.json .
```

### Option 2: Bring Your Own Image

The `index_to_name.json` file is optional in the Flask service below. You can test your service with your own image - just make sure it’s a 3-color JPEG.

### Building Your Flask Service

The full Python script for the Flask service is shown at the end of this recipe; you can copy and paste that into your own `app.py` file. Below we’ll look at individual sections to make their functions clear.

#### Imports

```python
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
```

In order:

- We’ll be using a pre-trained DenseNet model from `torchvision.models`
- `torchvision.transforms` contains tools for manipulating your image data
- Pillow (PIL) is what we’ll use to load the image file initially
- And of course we’ll need classes from `flask`

#### Pre-Processing

```python
def transform_image(infile):
    input_transforms = [transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg
```

The web request gave us an image file, but our model expects a PyTorch tensor of shape `(N, 3, 224, 224)` where `N` is the number of items in the input batch. (We will just have a batch size of 1.) The first thing we do is compose a set of TorchVision transforms that resize and crop the image, convert it to a tensor, then normalize the values in the tensor. (For more information on this normalization, see the documentation for `torchvision.models`.)

After that, we open the file and apply the transforms. The transforms return a tensor of shape `(3, 224, 224)` - the 3 color channels of a 224x224 image. Because we need to make this single image a batch, we use the `unsqueeze_(0)` call to modify the tensor in place by adding a new first dimension. The tensor contains the same data, but now has shape `(1, 3, 224, 224)`.

In general, even if you’re not working with image data, you will need to transform the input from your HTTP request into a tensor that PyTorch can consume.

#### Inference

```python
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction
```

The inference itself is the simplest part: When we pass the input tensor to the model, we get back a tensor of values that represent the model’s estimated likelihood that the image belongs to a particular class. The `max()` call finds the class with the maximum likelihood value, and returns that value with the ImageNet class index. Finally, we extract that class index from the tensor containing it with the `item()` call, and return it.

#### Post-Processing

```python
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name
```

The `render_prediction()` method maps the predicted class index to a human-readable class label. It’s typical, after getting the prediction from your model, to perform post-processing to make the prediction ready for either human consumption, or for another piece of software.

### Running The Full Flask App

Paste the following into a file called `app.py`:

```python
import io
import json
import os

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
model.eval()                                              # Turns off autograd

img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg

# Get a prediction
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()
```

To start the server from your shell prompt, issue the following command:

```sh
FLASK_APP=app.py flask run
```

By default, your Flask server is listening on port 5000. Once the server is running, open another terminal window, and test your new inference server:

```sh
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@kitten.jpg"
```

If everything is set up correctly, you should receive a response similar to the following:

```json
{"class_id":285,"class_name":"Egyptian_cat"}
```
