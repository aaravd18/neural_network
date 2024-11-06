import numpy as np
import random
import time

class Network:
    def __init__(self, dimensions):  
        # Dimensions is a list representing the nodes in each layer
        # E.g [5,3,1] is a 3-layer network with 5 nodes in first layer, 3 in second, 1 in last.
        self.layers = dimensions
        self.num_layers = len(dimensions)

        # Initialise weights of each layer as a 2d matrix with random values from a standard normal distribution
        self.weights = [
            np.random.randn(self.layers[i+1], self.layers[i]) * 0.01 for i in range(len(self.layers) - 1) 
        ]

        # Initialise bias of each layer as a zero vector
        self.biases = [
            np.zeros((1, self.layers[i+1])) for i in range(len(self.layers) - 1)
        ]

    
    def feed_forward(self, x):
        """
        Returns the output of the network for each sample in mini-batch x
        """
        for w, b in zip(self.weights, self.biases):
            x = leaky_ReLU(np.dot(x, w.T) + b)  # Apply weights and biases
        return x

    def back_propagation(self, batch, expected_output):
        """Returns the error in each weight and bias in each sample"""
        # Batch is a matrix where each row is an input sample
        # Store the activations of each layer 
        self.activations = [batch]  
        z_values = []  
        activation = batch
    
        for w, b in zip(self.weights, self.biases):
            # w has shape (layers[i+1], layers[i])
            # b has shape (1, layers[i+1])
            # activation has shape (batch_size, layers[i])
            z = np.dot(activation, w.T) + b  # z has shape (batch_size, layers[i+1])
            z_values.append(z)
            activation = leaky_ReLU(z)  # activation has shape (batch_size, layers[i+1])
            self.activations.append(activation)

        # Calculate error in the output layer
        delta_L = (self.activations[-1] - expected_output) * leaky_ReLU_prime(z_values[-1])

        # Initialize gradients
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        # Calculate gradients for the last layer
        # batch.shape[0] represents the batch size so we are averaging the gradient of each weight/bias over the batch
        delta_w[-1] = np.dot(delta_L.T, self.activations[-2]) / batch.shape[0]  
        delta_b[-1] = np.mean(delta_L, axis=0, keepdims=True)  # (1, layers[-1])

        # Backpropagate the error to previous layers
        for l in range(2, len(self.weights) + 1):
            z = z_values[-l]
            delta_L = np.dot(delta_L, self.weights[-l + 1]) * leaky_ReLU_prime(z)  # dimension (batch_size, layers[-l])
            
            # Compute gradients for layer l
            delta_w[-l] = np.dot(delta_L.T, self.activations[-l - 1]) / expected_output.shape[0]  # dimension (layers[-l], layers[-l-1])
            delta_b[-l] = np.mean(delta_L, axis=0, keepdims=True)  # dimension (1, layers[-l])
        
        return delta_w, delta_b
    
    def mini_batch_update(self, mini_batch, learning_rate):
        # Initialize accumulators for the gradients
        total_delta_w = [np.zeros(w.shape) for w in self.weights]
        total_delta_b = [np.zeros(b.shape) for b in self.biases]

        # For each (x, y) in the mini-batch
        for x, y in mini_batch:
            # Compute gradients for the current sample using backpropagation
            delta_w, delta_b = self.back_propagation(x, y)

            # Accumulate gradients
            total_delta_w = [tw + dw for tw, dw in zip(total_delta_w, delta_w)]
            total_delta_b = [tb + db for tb, db in zip(total_delta_b, delta_b)]

        # Update weights and biases
        self.weights = [w - (learning_rate / len(mini_batch)) * tw for w, tw in zip(self.weights, total_delta_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * tb for b, tb in zip(self.biases, total_delta_b)]
        
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate):
        for epoch in range(epochs):
            time1 = time.time()
            # Shuffle the training data
            random.shuffle(training_data)

            # Divide the data into mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]

            # Perform mini-batch updates
            for mini_batch in mini_batches:
                self.mini_batch_update(mini_batch, learning_rate)

            # Optionally, print progress
            time2 = time.time()
            print(f"Epoch {epoch + 1} complete in {time2-time1} seconds")


def ReLU(x):
    """Rectified Linear Unit Non-Linearity"""
    return np.maximum(0, x)

def ReLU_prime(x):
    """Derivative of ReLU"""
    return np.where(x > 0, 1, 0)

def leaky_ReLU(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_ReLU_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
        
    
# ReLU function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the ReLU function
def sigmoid_prime(x):
    return ReLU(x) * (1 - ReLU(x))
        
