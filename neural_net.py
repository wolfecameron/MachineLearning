"""This file contains my personal implementation of a simple neural network with
forward and backward propogation that I used for several machine learning projects
"""

import numpy as np

from neural_net_help import create_weight_mat

class Neural_Network:
	"""Class for neural network object"""

	def __init__(self, layer_sizes):
		"""Constructor for neural net object, takes in a list
		of integers where each integer in the list represents
		the number of hiddens nodes in a certain layer (including
		input/output layers)"""

		self.weights = []
		
		# create all initial weight matrices with proper sizes
		for ind in range(1, len(layer_sizes)):
			# matrix should have number columns equal to current input layer
			# and number rows equal to number of resulting output nodes
			rows = layer_sizes[ind]
			cols = layer_sizes[ind - 1]
			self.weights.append(create_weight_mat((rows, cols))
