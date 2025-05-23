# examples/basic_usage.py

import sys
sys.path.insert(0, '..')

from micrograd import Value, draw_dot
import os

# Create a directory for graph outputs if it doesn't exist
output_dir = "graph_outputs"
os.makedirs(output_dir, exist_ok=True)


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

print(f"L: {L}") # Value(data=-8.0)

# Visualize the graph before backward pass
dot_before = draw_dot(L)
dot_before.render(os.path.join(output_dir, 'basic_usage_L_before_backward'), view=False, format='png')
print(f"Graph 'basic_usage_L_before_backward.png' saved in '{output_dir}/'")

# Perform backward pass
L.backward()

# Visualize the graph after backward pass
dot_after = draw_dot(L)
dot_after.render(os.path.join(output_dir, 'basic_usage_L_after_backward'), view=False, format='png')
print(f"Graph 'basic_usage_L_after_backward.png' saved in '{output_dir}/'")

# Print gradients
print("\nGradients:")
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")
print(f"d.grad: {d.grad}")
print(f"e.grad: {e.grad}")
print(f"f.grad: {f.grad}")
print(f"L.grad: {L.grad}")

print("\n--- Manual Gradient Step Example ---")

print(f"Initial a.data: {a.data}, a.grad: {a.grad}")
print(f"Initial b.data: {b.data}, b.grad: {b.grad}")
print(f"Initial c.data: {c.data}, c.grad: {c.grad}")
print(f"Initial f.data: {f.data}, f.grad: {f.grad}")


a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

# Recompute L with new data values
e_new = a * b
d_new = e_new + c
L_new = d_new * f

print(f"New L.data after one step: {L_new.data}") 