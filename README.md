This project is a Python implementation of a scalar-valued automatic differentiation (autograd) engine and a basic neural network library.

Created by closely following Andrej Karpathy's renowned "Neural Networks: Zero to Hero" series.

The primary motivation behind this project was to gain a deep, hands-on understanding of backpropagation and gradient descent, which are the core learning mechanisms in neural networks. This repository is a personal exploration and a testament to the "learn by doing" philosophy.

## ✨ What This Project Implements

### Scalar Autograd Engine (Value class)
* Tracks a computational graph for mathematical operations.
* Performs automatic differentiation using a `.backward()` method.
* Supports core arithmetic operations: `+`, `-`, `*`, `/`, `**`.
* Includes activation functions like `tanh`, `exp`.

### 🎃 Neural Network Components
* **Neuron:** Represents a single neuron with weights, bias, and `tanh` activation.
* **Layer:** A collection of neurons.
* **MLP:** A Multi-Layer Perceptron, built from layers.

### 📈 Graph Visualization
* Uses Graphviz to visualize the computational graph, providing insights into both the forward and backward passes.

## Project Structure

![image](https://github.com/user-attachments/assets/743947fc-3489-4f46-9e04-5aef3bfcbf6c)
