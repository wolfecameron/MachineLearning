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
		self.act_values = []

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
		self.act_values.append(inputs)

		# forward activate the inputs through all of the weight matrices
		curr_vals = inputs
		for weight_mat in self.weights:
			# must stack a bias unit on top
			vals = np.dot(weight_mat, np.vstack([np.ones((1,1)), curr_vals]))
			# store intermediate values to be used by backpropogation
			# must store both unactivated and activated values for backpropogation
			self.values.append(vals)
			self.act_values.append(activate(vals))
			curr_vals = vals
		
		return curr_vals

	def backward_prop(self, expected, alpha=.0001, grad_check=False, inputs=None):
		"""Backward propogates the neural network to update the weights
		based on a current training example given to the network

		Parameters:
		inputs -- the inputs for the current example (should be a np array)
		outputs -- the actual outputs of the neural network
		expected -- the expected outputs of the neural network
		NOTE: expected and outputs should both be (N X 1)
		
		a list of the numpy arrays containing the gradient for each matrix is
		returned, empty if grad_check is false
		"""
		
		# must track delta as you backpropogate
		delta = None
		for layer in reversed(range(1, self.layers)):
			gradients = []
			# backprop for output layer
			if(layer == self.layers - 1):

				error = expected - self.act_values[layer]
				delta = np.multiply(error, sig_deriv(self.values[layer]))
			
				# add changes based on deltas into the weight matrix
				updates = np.vstack([np.ones((1,1)), self.act_values[layer - 1]]).dot(delta.T).T
				self.weights[layer - 1] += np.multiply(alpha, updates)
				
				# if gradient is being checked, add to the current list
				if grad_check:
					grad_arr = np.zeros(self.weights[layer - 1].shape)
					for r in range(self.weights[layer - 1].shape[0]):
						for c in range(self.weights[layer - 1].shape[1]):
							grad_arr[r][c] = self.compute_numerical_gradient(inputs, expected, layer-1, r, c, .0001)		
					print("\n\nBACKPROP DELTA:\n")
					print(np.multiply(alpha, updates))
					print("\n\nGRADIENT CHECK\n")
					print(grad_arr)
					input()
				
			
			# backprop for all hidden layers
			else:
				# must eliminate bias from the error
				error = delta.dot(self.weights[layer][:, 1:]) # 1 X 2
				delta = np.multiply(error, sig_deriv(self.values[layer]).T) # 2 X 1 - multiply upper gradient by gradient at current node
				
				# find updates to weights and add values to weight matrix	
				updates = np.vstack([np.ones((1,1)), self.act_values[layer - 1]]).dot(delta).T
				self.weights[layer - 1] += np.multiply(alpha, updates)			
				
				# determine the gradient check matrix and check to see if they are the same
				if grad_check:
					grad_arr = np.zeros(self.weights[layer - 1].shape)
					for r in range(self.weights[layer - 1].shape[0]):
						for c in range(self.weights[layer - 1].shape[1]):
							grad_arr[r][c] = self.compute_numerical_gradient(inputs, expected, layer-1, r, c, .0001)
					print("\n\nBACKPROP DELTA:\n")
					print(updates)
					print("\n\nGRADIENT CHECK\n")
					print(grad_arr)
					input()

	def compute_numerical_gradient(self, ins, out, layer, r, c, epsilon):
		"""this method computes the numerical gradient of the neural network that
		can be compared to the actual gradient to ensure that backpropogation is
		working properly"""
		
		# perturb the weight upwards and compute the output
		self.weights[layer][r, c] += epsilon
		JUp = out - self.forward_prop(ins)
		
		# perturb the weight downwards and compute the output
		self.weights[layer][r, c] -= (2*epsilon)
		JLow = out - self.forward_prop(ins)

		# calculate numerical gradient and return
		self.weights[layer][r, c] += epsilon
		grad = (JUp - JLow)/(2*epsilon)
		return grad

if  __name__ == '__main__':
	"""Used for simple testing"""
	
	nn = Neural_Network([2,3,1])
	inputs = [[1,1],[0,1],[1,0],[0,0]]
	expected = [0, 1, 1, 0]
	expected_np = np.array(expected, copy=True)
	loss_func_vals = []
	for x in range(10000):
		results = []
		for ins, exp in zip(inputs, expected):
			output = nn.forward_prop(ins)
			results.append(output)
			nn.backward_prop(np.array(exp).T, alpha=.001, grad_check=False, inputs=ins)
		result_np = np.array(results, copy=True)
		loss = np.sum(np.square(np.subtract(expected_np, result_np)))
		loss_func_vals.append(loss)	
		
	visualize_cost(loss_func_vals)	
	
	print(nn.forward_prop([0,0]))
	print(nn.forward_prop([1,0]))
	print(nn.forward_prop([0,1]))
	print(nn.forward_prop([1,1]))
		
