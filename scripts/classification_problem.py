"""This file contains example script of training neural network for classification problem."""
import matplotlib.pyplot as plt
import numpy as np

from NaiveNeurals.MLP.functions import Tanh
from NaiveNeurals.MLP.network import NeuralNetwork
from NaiveNeurals.data.dataset import DataSet


def learn_xor() -> None:
    """Example function performing XOR learning for various network parameters."""

    nn = NeuralNetwork()
    input_data_size = 2
    output_data_size = 1
    hidden_layer_number_of_nodes = 8
    hidden_layer_bias = -0.6
    output_layer_bias = 0.05
    weights_range = 1
    hidden_layer_act_func = Tanh()
    output_layer_act_func = Tanh()

    nn.setup_network(input_data_size=input_data_size, output_data_size=output_data_size,
                     hidden_layer_number_of_nodes=hidden_layer_number_of_nodes,
                     hidden_layer_bias=hidden_layer_bias, output_layer_bias=output_layer_bias,
                     hidden_layer_act_func=hidden_layer_act_func, output_layer_act_func=output_layer_act_func,
                     weights_range=weights_range)
    nn.LEARNING_RATE = 0.015

    # every list in inputs represents one input and data pushed into network
    inputs = [[0, 0, 1, 1], [1, 0, 1, 0]]
    targets = [[1, 0, 0, 1]]

    data_set = DataSet(inputs, targets)
    nn.train(data_set)

    siz = 101
    x_input = []
    y_input = []
    for ind_x in np.linspace(0, 1, siz):
        for ind_y in np.linspace(0, 1, siz):
            x_input.append(ind_x)
            y_input.append(ind_y)

    res = [nn.forward([[ins[0]], [ins[1]]]) for ins in zip(x_input, y_input)]
    res = [_[0][0] for _ in res]
    res_unified = [1 if x > 0.5 else 0 for x in res]

    f1 = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(x_input, y_input[::-1], c=res, cmap='PiYG')

    ax2 = f2.add_subplot(111)
    ax2.scatter(x_input, y_input[::-1], c=res_unified, cmap='PiYG')

    ax3 = f3.add_subplot(111)
    plt.yscale('log')
    ax3.plot(nn.convergence_profile)

    plt.show()


if __name__ == '__main__':
    learn_xor()
