"""This file contains all helper functions that were used in the implementation of
the neural network
"""

import numpy as np
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

if __name__ == '__main__':
	"""Use for simple tests"""

	mat = create_weight_mat((1,6))
	print(mat)
	
	mat_1 = create_weight_mat((3,4))
	print(mat_1)


