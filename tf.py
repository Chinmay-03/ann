import tensorflow as tf
from keras.datasets import mnist 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0   
X_test = X_test / 255.0
model = tf.keras.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)), 
     tf.keras.layers.Dense(128, activation='relu'), 
     tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test) 
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
# Take an image from the MNIST test set for prediction
image_index = 5 # Index of the image in the test set (change as needed)
image = X_test[image_index]
label = y_test[image_index]
# Reshape the image for model input
image = image.reshape((1, 28, 28)) #first dimension is 1, which implies that there is only one image in this array
# Use the trained model to predict the digit
prediction = np.argmax(model.predict(image), axis=-1)

print(f"The actual digit is: {label}")
print(f"The predicted digit is: {prediction}")



# Take an image from the MNIST test set for prediction
image_index = 1 # Index of the image in the test set (change as needed)
image = X_test[image_index]
label = y_test[image_index]

# Display the image
plt.imshow(image, cmap='gray')
plt.title(f"Actual digit: {label}")
plt.show()

# Reshape the image for model input
image = image.reshape((1, 28, 28)) #first dimension is 1, which implies that there is only one image in this array

# Use the trained model to predict the digit
prediction = np.argmax(model.predict(image), axis=-1)

print(f"The actual digit is: {label}")
print(f"The predicted digit is: {prediction}")
