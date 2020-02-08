"""helper methods for implementing neural network"""

# must include this to avoid backend error message with mac
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

def create_weight_mat(size):
    """creates a weight matrix with kaiming initialization"""

    std = np.sqrt(2/size[0]) # fan in
    weights = np.random.normal(0, .4, size=size)
    return weights

def relu(values):
    """relu activation function"""	
    values[values < 0] = 0.
    return values

def d_drelu(grad):
    grad[grad < 0] = 0.
    grad[grad >= 0] = 1.
    return grad

def sigmoid(values):
    return 1./(1. + np.exp(-values))

def d_dsigmoid(act):
    """assumed that the input is activation values not pre-activation"""
    return act*(1. - act)

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
