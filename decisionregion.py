import numpy as np
import matplotlib.pyplot as plt

# Generate random data for two classes
np.random.seed(0)
class_1 = np.random.randn(50, 2) + [2, 2]
class_2 = np.random.randn(50, 2) + [-2, -2]

# Concatenate the data and labels
X = np.concatenate((class_1, class_2))
y = np.concatenate((np.ones(50), -np.ones(50)))

# Initialize the weights and bias
w = np.random.randn(2)
b = np.random.randn(1)

# Perceptron learning algorithm
learning_rate = 0.1
num_epochs = 20

for epoch in range(num_epochs):
    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # Compute the activation
        activation = np.dot(w, x) + b

        # Update the weights and bias
        if activation * target <= 0:
            w += learning_rate * target * x
            b += learning_rate * target

# Plot the decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', label='Class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], c='red', label='Class 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Perceptron Decision Regions')
plt.show()
