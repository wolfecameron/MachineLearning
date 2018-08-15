"""This file contains my personal implementation of a simple neural network with
forward and backward propogation that I used for several machine learning projects
"""

import numpy as np

from neural_net_help import create_weight_mat, activate, sig_deriv, visualize_cost

class Neural_Network:
	"""Class for neural network object"""

	def __init__(self, layer_sizes):
		"""Constructor for neural net object, takes in a list
		of integers where each integer in the list represents
		the number of hiddens nodes in a certain layer (including
		input/output layers
		
		Parameters:
		layer_sizes-- list of number of nodes present in each layer
		"""

		self.layer_sizes = layer_sizes
		self.layers = len(layer_sizes)
		self.weights = []
		# use to store intermediate values when activating
		self.values = [] 

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
	
		# clear intermediate values so the list can be repopulated
		self.values = []

		#transform input into a numpy array that can be dot multiplied
		inputs = np.array([inputs])
		inputs = np.transpose(inputs)
		self.values.append(inputs)

		# forward activate the inputs through all of the weight matrices
		curr_vals = inputs
		for weight_mat in self.weights:
			# must stack a bias unit on top
			vals = activate(np.dot(weight_mat, np.vstack([np.ones((1,1)), curr_vals])))
			# store intermediate values to be used by backpropogation
			self.values.append(vals)
			curr_vals = vals
		
		return curr_vals

	def backward_prop(self, outputs, expected, alpha=.01):
		"""Backward propogates the neural network to update the weights
		based on a current training example given to the network

		Parameters:
		inputs -- the inputs for the current example (should be a np array)
		outputs -- the actual outputs of the neural network
		expected -- the expected outputs of the neural network
		NOTE: expected and outputs should both be (N X 1)
		"""
		
		# must track delta as you backpropogate
		delta = None
		for layer in reversed(range(1, self.layers)):
			# backprop for output layer
			if(layer == self.layers - 1):
				error = expected - outputs
				delta = np.multiply(error, sig_deriv(outputs))
				
				# add changes based on deltas into the weight matrix	
				updates = np.vstack([np.ones((1,1)), self.values[layer - 1]]).dot(delta.T).T
				self.weights[layer - 1] += np.multiply(alpha, updates)
			
			# backprop for all hidden layers
			else:
				# must eliminate bias from the error
				error = delta.dot(self.weights[layer][:, 1:]) # 1 X 2 - which way is better
				delta = np.multiply(error, sig_deriv(self.values[layer]).T) # 2 X 1 - multiply upper gradient by gradient at current node
				# find updates to weights and add values to weight matrix	
				updates = np.vstack([np.ones((1,1)),self.values[layer - 1]]).dot(delta).T
				self.weights[layer - 1] += np.multiply(alpha, updates)			
				

if  __name__ == '__main__':
	"""Used for simple testing"""
	
	nn = Neural_Network([2,4,1])
	inputs = [[1,1],[0,1],[1,0],[0,0]]
	expected = [0, 1, 1, 0]
	expected_np = np.array(expected, copy=True)
	loss_func_vals = []
	for x in range(100000):
		results = []
		for ins, exp in zip(inputs, expected):
			output = nn.forward_prop(ins)
			results.append(output)
			nn.backward_prop(output, np.array(exp).T, alpha=.05)
		result_np = np.array(results, copy=True)
		loss = np.sum(np.square(np.subtract(expected_np, result_np)))
		loss_func_vals.append(loss)	
		
	visualize_cost(loss_func_vals)	
	
	print(nn.forward_prop([0,0]))
	print(nn.forward_prop([1,0]))
	print(nn.forward_prop([0,1]))
	print(nn.forward_prop([1,1]))
		
