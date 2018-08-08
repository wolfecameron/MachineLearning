"""This file contains my personal implementation of a simple neural network with
forward and backward propogation that I used for several machine learning projects
"""

import numpy as np

from neural_net_help import create_weight_mat, activate

class Neural_Network:
	"""Class for neural network object"""

	def __init__(self, layer_sizes):
		"""Constructor for neural net object, takes in a list
		of integers where each integer in the list represents
		the number of hiddens nodes in a certain layer (including
		input/output layers)"""

		self.layer_sizes = layer_sizes
		self.weights = []
		
		# create all initial weight matrices with proper sizes
		for ind in range(1, len(layer_sizes)):
			# matrix should have number columns equal to current input layer
			# and number rows equal to number of resulting output nodes
			rows = layer_sizes[ind]
			cols = layer_sizes[ind - 1] + 1 # must add extra weights for bias
			self.weights.append(create_weight_mat((rows, cols)))
	
	
	def forward_prop(self, inputs):
		"""Forward propogates the neural network and returns the
		resulting output value(s)

		Parameters:
		inputs -- the inputs into the neural network - type is expected
		to be a normal python list
		"""
		
		# make sure input is the correct length
		if(len(inputs) != self.layer_sizes[0]):
			print("Error: {0} inputs were expected and {1} were given. \
				Returning null...".format(str(self.layer_sizes[0]), 
				str(len(inputs))))
		
			return None

		#transform input into a numpy array that can be dot multiplied
		inputs = np.array([inputs])
		inputs = np.transpose(inputs)
		
		# forward activate the inputs through all of the weight matrices
		curr_vals = inputs
		for weight_mat in self.weights:
			# must stack a bias unit on top
			curr_vals = activate(np.dot(weight_mat, np.vstack([np.ones((1,1)),curr_vals])))
			
		
		return curr_vals


if __name__ == '__main__':
	"""Used for simple testing"""
	
	x = Neural_Network([3,1])
	print(x.forward_prop([1,1,1]))
	print(x.weights[0])
	
	
