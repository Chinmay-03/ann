from sklearn.linear_model import Perceptron
import numpy as np
# Training data for even and odd ASCII values
X_train = np.array([
    [48, 0], [50, 0], [52, 0], [54, 0], [56, 0],  # Even ASCII values: '0', '2', '4', '6', '8'
    [49, 1], [51, 1], [53, 1], [55, 1], [57, 1]   # Odd ASCII values: '1', '3', '5', '7', '9'
])


# Target labels (0 for even, 1 for odd)
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# Initialize and train the Perceptron model
model = Perceptron()
model.fit(X_train, y_train)

# Test the model with ASCII values of digits from 0 to 9
test_values = np.array([
    [53, 1], [50, 0], [48, 0], [55, 1], [52, 0]  # '5', '2', '0', '7', '4'
])


predictions = model.predict(test_values)
# Print the predictions
for value, prediction in zip(test_values[:, 0], predictions):
    if prediction == 0:
        print(f"ASCII value {value} is Even")
    else:
        print(f"ASCII value {value} is odd.")

####
import numpy as np

# Training data
training_inputs = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1],
    ]
)


training_labels = np.array(
    [
        [1],  # Even
        [0],  # Odd
        [0],  # Odd
        [1],  # Even
        [0],  # Odd
        [0],  # Odd
        [0],  # Odd
        [1],  # Even
        [0],  # Odd
        [0],  # Odd
    ]
)
# training_labels = np.array(
#     [
#         [1],  # Even
#         [-1],  # Odd
#         [-1],  # Odd
#         [1],  # Even
#         [-1],  # Odd
#         [-1],  # Odd
#         [1],  # Even
#         [-1],  # Odd
#         [1],  # Even
#         [-1],  # Odd
#     ]
# )


# Perceptron Neural Network class
class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros((num_inputs, 1))
        self.bias = 0

    def train(self, inputs, labels, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for input_data, label in zip(inputs, labels):
                prediction = self.predict(input_data)
                error = label - prediction

                self.weights += learning_rate * error * input_data.reshape(-1, 1)
                self.bias += learning_rate * error

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation >= 0 else 0


# Training the perceptron
perceptron = Perceptron(num_inputs=6)
perceptron.train(training_inputs, training_labels, num_epochs=100, learning_rate=0.1)

# Testing the perceptron
test_inputs = np.array(
    [
        [1, 1, 0, 0, 1, 0],  # 0 (Even)
        # [0, 0, 1, 0, 1, 0, 0, 1, 1, 1],  # 9 (Odd)
        [1, 1, 0, 1, 1, 0],  # 6 (even)
        # [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],  # 8 (Even)
        # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1 (odd)
    ]
)

for input_data in test_inputs:
    prediction = perceptron.predict(input_data)
    number = "".join(map(str, input_data.tolist()))

    if prediction == 1:
        print(f"{number} is even.")
    else:
        print(f"{number} is odd.")
