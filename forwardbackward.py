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
