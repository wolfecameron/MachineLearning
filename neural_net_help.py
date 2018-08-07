"""This file contains all helper functions that were used in the implementation of
the neural network
"""

import numpy as np


def create_weight_mat(size):
	"""Creates a matrix with randomly initialized values between
	-.5 and .5"""

	weight_mat = np.random.rand(size[0], size[1])
	weight_mat -= .5

	return weight_mat


if __name__ == '__main__':
	"""Use for simple tests"""

	mat = create_weight_mat((1,6))
	print(mat)
	
	mat_1 = create_weight_mat((3,4))
	print(mat_1)

