# Neural Network from Scratch

This project is a simple neural network built entirely from scratch using Python and only the `numpy` library. My goal was to understand the fundamentals of neural networks by implementing each component manually, including backpropagation and gradient descent. Below, I outline the key parts of the project, what I did, and some insights I gained along the way. Credit to 3Blue1Brown and Michael Nielsen's "Neural Networks and Deep Learning" book for teaching the relevant mathematics, logic, and some code inspiration

## What I Did

1. **Backpropagation with Mini-Batch Optimization**  
   I implemented backpropagation, optimized to work on matrices of mini-batches rather than single inputs. By using Hadamard products (element-wise multiplications), I could calculate gradients for an entire mini-batch in one go, significantly speeding up the process. This approach is far more efficient than handling each input vector individually, especially for larger datasets.

2. **Stochastic Gradient Descent (SGD)**  
   I used stochastic gradient descent to update weights, helping the model converge effectively. Combined with mini-batch processing, SGD allowed for faster and more reliable training on varied datasets.

## What I Learned

- **Mathematics of Backpropagation**  
   Diving into the proofs and derivations of backpropagation formulas was both challenging and rewarding. This deepened my understanding of how errors move backward through layers, which in turn helped me implement these formulas correctly in Python.

- **Optimizing with Matrix Operations**  
   Translating these mathematical formulas into code and optimizing them for matrices was a valuable experience. Using numpy's broadcasting capabilities, I applied these operations on batch matrices, which required precise handling to ensure both efficiency and accuracy in the computations.

- **Activation Functions and Neuron Challenges**  
   Working with different activation functions like ReLU highlighted some interesting challenges. ReLU can lead to "dead neurons" (when a neuron’s output is zero and the gradient vanishes), making it difficult for the network to learn. I explored solutions such as:
   - **He Initialization**: To mitigate vanishing gradients, especially in deeper networks.
   - **Leaky ReLU**: To keep neurons active and avoid zero gradients, ensuring they continue to contribute to learning.

This project provided a solid foundation in both the theoretical and practical aspects of neural networks, along with key strategies for optimization and troubleshooting. 

