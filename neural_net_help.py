"""This file contains all helper functions that were used in the implementation of
the neural network
"""

# must include this to avoid backend error message with mac
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def create_weight_mat(size):
	"""Creates a matrix with randomly initialized values between
	-.5 and .5"""

	weight_mat = np.random.rand(size[0], size[1])
	weight_mat -= .5

	return weight_mat

def activate(values):
	"""applies a sigmoid activtion function to a given list of values
	and returns the values"""
	
	return expit(values)

def sig_deriv(values):
	"""Helper method to pass values through the derivative of the sigmoid
	function - used for backpropogation of neural network
	
	*Assumed that values has already been passed through sigmoid activation
	"""
	
	return values*(1 - values)

def visualize_cost(error_vals):
	"""Method for visualizing the cost function that the neural network
	is optimizing to ensure that gradient descent is working properly
	"""
	
	plt.title("Cost Function Visualization")
	plt.ylabel("Cost")
	plt.xlabel("Iteration")
	plt.plot(error_vals)
	plt.show()

if __name__ == '__main__':
	"""Use for simple tests"""

	mat = create_weight_mat((1,6))
	print(mat)
	
	mat_1 = create_weight_mat((3,4))
	print(mat_1)


