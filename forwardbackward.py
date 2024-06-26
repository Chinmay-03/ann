import numpy as np
# Sigmoid activation function and its derivative
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)
# Input and output datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(1)
w1, w2 = 2 * np.random.random((2, 4)) - 1, 2 * np.random.random((4, 1)) - 1
# Train the neural network
for _ in range(10000):
    in_layer, hid_layer = X, sigmoid(np.dot(X, w1))
    out_layer = sigmoid(np.dot(hid_layer, w2))

    # Backpropagation
    out_error = y - out_layer
    out_delta = out_error * sigmoid_derivative(out_layer)
    hid_error = out_delta.dot(w2.T)
    hid_delta = hid_error * sigmoid_derivative(hid_layer)

    # Update weights
    w2 += hid_layer.T.dot(out_delta)
    w1 += in_layer.T.dot(hid_delta)
print("Output after training:")
print(out_layer)

# Implement Artificial Neural Network training process in Python by using forward propagation,
# backpropagation.

# error = target - output
# delta = error * sig_derivative (output of that layer)
# w = w + transpose_input @ that_layer_delta
# b = b + sum of columns of delta of that layer
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = self.output_delta @ self.W2.T
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)
        self.W1 += X.T @ self.hidden_delta
        self.b1 += np.sum(self.hidden_delta, axis=0)
        self.W2 += self.a1.T @ self.output_delta
        self.b2 += np.sum(self.output_delta, axis=0)

    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        # Make predictions for a given set of inputs
        return self.forward(X)


# Define the input and output datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 4 neurons in the hidden layer, and 1 output neuron
nn = NeuralNetwork(2, 4, 1)

# Train the neural network on the input and output datasets for 10000 epochs with a learning rate of 0.1
nn.train(X, y, epochs=10000)

# Use the trained neural network to make predictions on the same input dataset
predictions = nn.predict(X)

# Print the predictions
print(predictions)

