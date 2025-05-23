This project is a Python implementation of a scalar-valued automatic differentiation (autograd) engine and a basic neural network library. **It was created by closely following Andrej Karpathy's renowned 'Neural Networks: Zero to Hero' video side by side.** The primary motivation behind this endeavor was to gain a deep, hands-on understanding of the fundamental mechanisms of backpropagation and gradient descent, which are at the heart of how neural networks learn.

This repository serves as a personal exploration and a testament to the "learn by doing" philosophy.

## What This Project Implements

* **Scalar Autograd Engine (`Value` object):**
    * Tracks a computational graph for mathematical operations.
    * Automatically computes gradients via backpropagation (`backward()` method).
    * Supports essential arithmetic operations (`+`, `*`, `-`, `/`, `**`) and activation functions like `tanh` and `exp`.
* **Basic Neural Network Components:**
    * `Neuron`: A single neuron with weights, a bias, and `tanh` activation.
    * `Layer`: A collection of neurons forming a layer.
    * `MLP`: A Multi-Layer Perceptron built from these layers.
* **Computational Graph Visualization:**
    * Utilities to render the graph of operations using Graphviz, offering visual insight into the forward and backward passes.

## Project Structure

MicroGrad-Repo/
├── micrograd/            # The core library code
│   ├── engine.py       # Contains the Value class for autograd
│   ├── nn.py           # Neuron, Layer, and MLP class definitions
│   ├── viz.py          # Visualization functions using Graphviz
│   └── init.py     # Makes 'micrograd' a Python package
├── examples/             # Python scripts demonstrating usage
|   |── Graph_outputs/        # Directory where graph images are saved (auto-created)
│   ├── basic_usage.py
│   ├── single_neuron_example.py
│   └── train_mlp.py
├── README.md             # This file
