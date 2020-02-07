"""simple implementation of neural network"""

import numpy as np
import random

from neural_net_help import (
    create_weight_mat,
    relu,
    d_drelu,
    sigmoid,
    d_dsigmoid,
    visualize_cost,
)

class FFNN:
    """Class for neural network object"""

    def __init__(self, layer_sizes):
        """Constructor for neural net object, takes in a list
        of integers where each integer in the list represents
        the number of hiddens nodes in a certain layer (including
        input/output layers)
		
        Parameters:
        layer_sizes-- list of number of nodes present in each layer
        """

        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.weights = []
		
        # use to store intermediate values when activating
        self.values = [] 
        self.act_values = []

        # create kaiming initialized weight matrices
        for ind in range(1, len(layer_sizes)):
            num_in = layer_sizes[ind - 1] + 1 # account for bias
            num_out = layer_sizes[ind]
            self.weights.append(create_weight_mat((num_in, num_out)))

        # store gradient in loss for backprop
        self.loss_grad = np.zeros((1, 1))

    def bce_loss(self, output, target):
        """binary cross entropy loss function -- stores the loss
        gradient for backpropogation

        output: output array of size B X 1
        target: target array of size B X 1
        """

        loss = -((1 - target)*np.log(1 - output)
                + target*np.log(output))
        self.loss_grad = (target - output)
        return float(loss)
        

    def forward(self, inputs):
        """Forward propogates the neural network and returns the
        resulting output value(s)

        Parameters:
        inputs -- the inputs into the neural network - type is expected
        to be a numpy array of size b x n
        """
	
        # clear intermediate values so the list can be repopulated
        # these values must be tracked for use in backward pass
        self.z_vals = []
        self.a_vals = []

        # transform into numpy array
        self.z_vals.append(inputs)

        # forward activate the inputs through all of the weight matrices
        curr_vals = inputs
        for weight_mat in self.weights:
            self.a_vals.append(curr_vals)
            
            # must stack a bias unit on top
            bias = np.ones((curr_vals.shape[0], 1))
            data_in = np.hstack((curr_vals, bias))
            z_val = np.dot(data_in, weight_mat)
            a_val = relu(z_val)
            self.z_vals.append(z_val)
            curr_vals = a_val
        return sigmoid(curr_vals)

    def backward(self, outputs, alpha:float=.1):
        error = self.loss_grad*d_dsigmoid(outputs)
        for ind in reversed(range(1, self.layers)):
            weights = self.weights[ind - 1]
            prev_acts = self.a_vals[ind - 1]
            input_vec = np.vstack((np.transpose(prev_acts), np.ones((1, 1))))
            weight_grad = np.dot(input_vec, error)
            self.weights[ind - 1] += alpha*weight_grad # update weights
            if ind > 1:
                prev_z = self.z_vals[ind - 1]
                act_grad = np.dot(error, np.transpose(weights))
                act_grad = act_grad[:, :-1] # remove bias
                error = act_grad*d_drelu(prev_z)

if  __name__ == '__main__':
    """Used for simple testing"""

    # train the neural net on xor	
    nn = FFNN([2, 8, 1])
    inputs = np.array([[1,1],[0,1],[1,0],[0,0]])
    expected = np.transpose(np.array([[0, 1, 1, 0]]))
    losses = []
    for x in range(1000):
        tmp_losses = 0.
        for ind in range(inputs.shape[0]):
            data_in = inputs[ind, :][None, :]
            target = expected[ind, :][None, :]
            output = nn.forward(data_in)
            tmp_loss = nn.bce_loss(output, target)
            nn.backward(output)
            tmp_losses += tmp_loss
        tmp_losses /= 4
        if x % 100 == 0:
            print(f'Loss: {tmp_losses}')
    print(f'0 --> {nn.forward(inputs[0][None, :])}')
    print(f'1 --> {nn.forward(inputs[1][None, :])}')
    print(f'1 --> {nn.forward(inputs[2][None, :])}')
    print(f'0 --> {nn.forward(inputs[3][None, :])}')
