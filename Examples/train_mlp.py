# examples/train_mlp.py
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from micrograd import MLP, Value 
import random   

# Define the MLP
n = MLP(3, [4, 4, 1]) # 3 inputs, 2 hidden layers of 4 neurons, 1 output neuron

# Training data
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # Desired targets

print(f"Number of parameters: {len(n.parameters())}")

# Training loop
learning_rate = 0.1 
epochs = 1000  

losses = []

for k in range(epochs):
  # Forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  # Backward pass
  # Reset gradients for all parameters
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # Update parameters (gradient descent)
  for p in n.parameters():
    p.data += -learning_rate * p.grad

  losses.append(loss.data)
  if k % 100 == 0 or k == epochs - 1: # Print every 100 epochs and the last one
      print(f"Epoch {k}: Loss = {loss.data}")

# Final predictions
print("\nFinal predictions after training:")
final_ypred = [n(x) for x in xs]
for i, (x_input, y_target, y_pred_val) in enumerate(zip(xs, ys, final_ypred)):
    print(f"Input: {x_input}, Target: {y_target}, Prediction: {y_pred_val.data:.4f}")


plt.plot(losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('graph_outputs/training_loss.png')
print("Plot 'MLP_training_loss.png' saved in 'graph_outputs/'")