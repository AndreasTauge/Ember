# Ember

Ember is a small neural-network and automatic-differentiation project written
in Rust. I made it for fun to learn about machine learning (or i guess the math behind it): the goal is to get more familiar with
how tensors, computation graphs, forward passes, and backward
passes work and maybe i will use it for some future machine learning project if it is not insanely slow. 

## Project structure

```text
src/
├── tensor/
│   └── mod.rs                 Tensor storage and numerical operations
├── autograd/
│   ├── mod.rs                 Exposes the tensor autograd module
│   ├── scalar_autodiff.rs     Earlier scalar-based autograd implementation that i quickly abandoned
│   └── tensor_autodiff.rs     Tensor computation graph and backpropagation
├── nn/
│   ├── mod.rs                 Exposes neural-network layers
│   └── linear.rs              Fully connected linear layer
├── lib.rs                     Exposes Ember's public modules
└── main.rs                    Small executable/example
```

The three main parts build on one another:

```text
tensor  ->  autograd  ->  nn
numbers     gradients     layers
```

## `tensor`: values, shapes, and math operations

`Tensor` is Ember's basic data container. It stores all values in one flat
`Vec<f32>` and stores a separate shape describing how those values should be
viewed.

For example:

```rust
let tensor = Tensor::new(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    vec![2, 3],
);
```

The shape `[2, 3]` means two rows with three values in each row:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

The tensor module currently implements:

- Element-wise `add`, `sub`, `mul`, and `div`
- Scalar arithmetic
- `exp`, `pow`, and `tanh`
- `sum` and `mean`
- Two-dimensional matrix multiplication
- Two-dimensional transpose
- Helpers such as `zeros_like` and `ones_like`

### Broadcasting

Addition supports broadcasting. Broadcasting lets a smaller tensor be reused
across a larger tensor when their latter dimensions are compatible. This makes forward passes a bit easier because you can batch examples and the bias addition will still work with a smaller bias tensor. 

For example, a matrix with shape `[2, 3]` can be added to a vector with shape
`[3]`:

```text
matrix = [
  [10, 20, 30],
  [40, 50, 60]
]

bias = [1, 2, 3]
```

The vector is treated as though it appears once for every matrix row:

```text
result = [
  [10 + 1, 20 + 2, 30 + 3],
  [40 + 1, 50 + 2, 60 + 3]
]
```

It is not permanently copied. `broadcast_index` maps each output position to
the correct position in the smaller tensor.

Two dimensions are broadcast-compatible when they are equal or one of them is
`1`. Missing dimensions on the left are also treated as `1`:

```text
[2, 3] +    [3]  -> [2, 3]
[2, 3] + [1, 3]  -> [2, 3]
[2, 3] + [2, 1]  -> [2, 3]
```

`sum_to_shape` performs the reverse operation for autograd. If one bias value
was reused for several batch rows during the forward pass, its gradient must
be the sum of the gradient contributions from all those rows.

## `autograd`: recording calculations and computing gradients

Autograd automatically calculates how a final result changes with respect to
the tensors used to produce it. These derivatives are the gradients used to
train neural networks.

The tensor autograd implementation uses a `Graph`. Rather than passing tensors
directly between graph operations, Ember stores each tensor as a `Node` and
returns its numerical `TensorId`.

Each node contains:

```text
data       the value calculated during the forward pass
grad       the gradient calculated during the backward pass
operation  the operation and input nodes that produced the value
```

Creating a tensor adds a leaf node:

```rust
let x = graph.tensor(tensor);
```

Performing an operation calculates its value immediately and records how it was
created:

```rust
let product = graph.matmul(x, weight);
let output = graph.add(product, bias);
```

Conceptually, this produces a graph like:

```text
x ------\
         matmul -> product --\
weight -/                   add -> output
bias ----------------------/
```

### The forward pass

Calls such as `graph.add` and `graph.matmul` perform the forward calculation.
They store both the result and the operation required to calculate gradients
later.

Use `graph.data(id)` to inspect a node's forward value.

### The backward pass

`graph.backward()` walks through the nodes in reverse order. It starts the last
node with a gradient of one because a value's derivative with respect to itself
is one. Each operation then passes its gradient to its inputs using that
operation's derivative.

For multiplication:

```text
z = x * y
dz/dx = y
dz/dy = x
```

For matrix multiplication:

```text
left gradient  = output gradient * transpose(right)
right gradient = transpose(left) * output gradient
```

For broadcast addition, the output gradient is reduced with `sum_to_shape` so
that it matches each original input. This is what turns one gradient per batch
row back into one accumulated gradient per neuron bias.

Use `graph.grad(id)` to inspect a node's gradient after `backward()`.

The graph currently treats the most recently created node as the final output.
For a typical scalar loss, call `graph.sum(output)` or construct another scalar
loss node before calling `backward()`.

### Scalar autograd

`scalar_autodiff.rs` contains the earlier version of the same idea using single
`f32` values instead of tensors. It is useful as a simpler reference for
understanding the graph and chain rule, but it is not currently used in
`autograd/mod.rs`.

## `nn`: reusable neural-network layers

The `nn` module builds layers from tensor operations recorded by autograd.

The current `Linear` layer owns two `TensorId` parameters:

- `weight`: controls how each input feature contributes to each output neuron
- `bias`: one additional trainable value for each output neuron

Its forward calculation is:

```text
output = input * weight + bias
```

For a batch of inputs, the expected shapes are:

```text
input:  [batch_size, input_features]
weight: [input_features, output_features]
bias:                   [output_features]
output: [batch_size,    output_features]
```

Rows represent different examples in the batch. Columns in the output represent
different neurons. The bias therefore matches the output-neuron dimension, not
the entire output shape. Broadcasting applies the same neuron biases to every
example in the batch.

Example:

```rust
use ember::{
    autograd::tensor_autodiff::Graph,
    nn::linear::Linear,
    tensor::Tensor,
};

let mut graph = Graph::new();

// Two input features and two output neurons.
let layer = Linear::new(
    &mut graph,
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    Tensor::new(vec![0.5, -0.5], vec![2]),
);

// A batch containing two examples.
let input = graph.tensor(Tensor::new(
    vec![1.0, 2.0, 3.0, 4.0],
    vec![2, 2],
));

let output = layer.forward(&mut graph, input);
let loss = graph.sum(output);
graph.backward();

println!("output: {:?}", graph.data(output).values());
println!("weight gradient: {:?}", graph.grad(layer.weight).values());
println!("bias gradient: {:?}", graph.grad(layer.bias).values());
println!("loss: {}", graph.data(loss).item());
```

## Current limitations

Ember is still small and intentionally incomplete. In particular:

- Matrix multiplication currently supports only two-dimensional tensors.
- Shapes are checked at runtime with assertions rather than a structured error
  type.
- There is no optimizer or parameter-update step yet.
- There are no activation, loss, or container layers in `nn` yet.
- Tensors run only on the CPU and use `f32` values.

## Running the project

Run the example:

```bash
cargo run
```

Run the tests:

```bash
cargo test
```

Run the linter:

```bash
cargo clippy --all-targets -- -D warnings
```
