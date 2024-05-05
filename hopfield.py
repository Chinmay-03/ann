
import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        # Initialize weights matrix as zero matrix of size n_neurons x n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        # Train the network with the provided patterns using Hebbian learning rule
        for pattern in patterns:
            # Outer product of pattern with itself, then accumulate to weights
            self.weights += np.outer(pattern, pattern)
        # Set diagonal weights to zero to prevent self-feedback
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern):
        # Predict function to retrieve a pattern based on current weights
        # Using sign function to update the network state
        return np.sign(pattern @ self.weights)


# Define patterns to be stored in the Hopfield Network
patterns = np.array([[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]])
n_neurons = 4  # Number of neurons in the network

# Initialize the Hopfield Network
network = HopfieldNetwork(n_neurons)

# Train the network with the defined patterns
network.train(patterns)

# Test the network to see if it can recall stored patterns
for pattern in patterns:
    prediction = network.predict(pattern)
    print("Input pattern:", pattern)
    print("Predicted pattern:", prediction)
