# examples/single_neuron_example.py

import sys
sys.path.insert(0, '..')

from micrograd import Value, draw_dot
import os

output_dir = "graph_outputs"
os.makedirs(output_dir, exist_ok=True)

# Inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias of the neuron
b = Value(6.8813735870195432, label='b')

# Neuron computation
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

print(f"Output 'o': {o}")

# Backward pass from 'o'
o.backward()

# Visualize
dot = draw_dot(o)
dot.render(os.path.join(output_dir, 'single_neuron_computation'), view=False, format='png')
print(f"Graph 'single_neuron_computation.png' saved in '{output_dir}/'")

print("\nGradients after o.backward():")
print(f"x1.grad: {x1.grad}")
print(f"w1.grad: {w1.grad}")
print(f"x2.grad: {x2.grad}")
print(f"w2.grad: {w2.grad}")
print(f"b.grad: {b.grad}")