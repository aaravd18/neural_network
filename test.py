import numpy as np
from network import Network  # Importing the Network class

import numpy as np

# Define the XOR dataset
training_data = [
    (np.array([[0, 0]]), np.array([[0]])),
    (np.array([[0, 1]]), np.array([[1]])),
    (np.array([[1, 0]]), np.array([[1]])),
    (np.array([[1, 1]]), np.array([[0]]))
]

# Initialize the network with 2 input neurons, 8 neurons in one hidden layer, and 1 output neuron
# This is a simple architecture to learn XOR
network = Network([2, 8, 1])

# Define parameters for training
epochs = 500
mini_batch_size = 1
learning_rate = 0.08

# Train the network
network.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate)

# Test the network on the training data
for x, y in training_data:
    prediction = network.feed_forward(x)
    print(f"Input: {x.flatten()}, Expected Output: {y.flatten()}, Predicted Output: {prediction.flatten()}")
