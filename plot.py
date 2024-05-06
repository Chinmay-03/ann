
# 1.	Write a program to plot Sigmoid activation functions that are being used in neural networks.
# 2.	Write a program to plot Tanh activation functions that are being used in neural networks.
# 3.	Write a program to plot ReLu activation functions that are being used in neural networks.
# 4.	Write a program to plot Softmax activation functions that are being used in neural networks.
# 5.	Generate ANDNOT function using McCulloch-Pitts neural net.
# 6.	Write a program using Perceptron Neural Network to recognize even numbers. Given numbers are in ASCII from 0 to 9
# 7.	Write a program using Perceptron Neural Network to recognize odd numbers. Given numbers are in ASCII from 0 to 9
# 8.	With a suitable example demonstrate the perceptron learning law with its decision regions using python. Give the output in graphical form.
# 9.	Write a python Program for Bidirectional Associative Memory with two pairs of vectors.
# 10.	Implement Artificial Neural Network training process in Python by using Forward Propagation.
# 11.	Implement Artificial Neural Network training process in Python by using Back Propagation.
# 12.	Write a python program to illustrate ART neural network.
# 13.	Write a python program to design a Hopfield Network which stores 4 vectors
# 14.	How to Train a Neural Network with TensorFlow/Pytorch and evaluation of logistic regression using tensorflow.
# 15.	MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow.
# 16.	Mini project on any Object Detection

'''Write a Python program to plot a few activation functions that are being used in neural networks'''

import numpy as np
import matplotlib.pyplot as plt

def binaryStep(x):
    return np.heaviside(x,1)
    
def linear(x):   
    return x
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
  
def softmax(x):
    soft = np.exp(x - np.max(x))
    return soft / np.sum(soft)


x = np.linspace(-10, 10, 100)
y = softmax(x)

plt.plot(x, y)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Softmax Activation Function")
plt.grid(True)
plt.show()

    
while True:
	print('Menu\n1.Binary\n2.Linear\n3.Sigmoid\n4.ReLu\n5.Tanh\n6.Exit')
	h =  int(input('Enter Your Choice:'))
	if h == 1:    
		x = np.linspace(-10, 10)
		plt.plot(x, binaryStep(x))
		plt.axis('tight')
		plt.title('Activation Function :Binary Step')
		plt.show()

	elif h == 2:
		x = np.linspace(-10, 10)
		plt.plot(x, linear(x))
		plt.axis('tight')
		plt.title('Activation Function :Linear')
		plt.show()

	elif h == 3:  
		x = np.linspace(-10, 10)
		plt.plot(x, sigmoid(x))
		plt.axis('tight')
		plt.title('Activation Function :Sigmoid')
		plt.show()
	elif h == 4:
		x = np.linspace(-10, 10)
		plt.plot(x, relu(x))
		plt.axis('tight')
		plt.title('Activation Function :ReLu')
		plt.show()

	elif h == 5:
		x = np.linspace(-10, 10)
		plt.plot(x, tanh(x))
		plt.axis('tight')
		plt.title('Activation Function :Tanh')
		plt.show()
		 
	elif h == 6:
		print('Bye')
		break	
	
	else:
		print('Enter Correct Choice !')
