"""This file contains example script of training neural network for regression problem."""
import itertools

import matplotlib.pyplot as plt

from NaiveNeurals.MLP.functions import Linear, Tanh
from NaiveNeurals.MLP.network import NeuralNetwork
from NaiveNeurals.data.data_generators import data_generator_for_regression
from NaiveNeurals.data.dataset import TrainingDataSet
from NaiveNeurals.utils import DataSeries, ConvergenceError


def sine_regression() -> None:
    """Example of regression problem for SINE."""

    x_values, y_values = data_generator_for_regression(DataSeries.SINE)

    nn = NeuralNetwork()
    input_data_size = 1
    output_data_size = 1
    hidden_layer_number_of_nodes = 12
    hidden_layer_bias = -1.5
    output_layer_bias = -0.5
    weights_range = 1
    nn.LEARNING_RATE = 0.01
    nn.TARGETED_ERROR_RATE = 0.0025
    hidden_layer_act_func = Tanh()
    output_layer_act_func = Linear()

    nn.setup_network(input_data_size=input_data_size, output_data_size=output_data_size,
                     hidden_layer_number_of_nodes=hidden_layer_number_of_nodes,
                     hidden_layer_bias=hidden_layer_bias, output_layer_bias=output_layer_bias,
                     hidden_layer_act_func=hidden_layer_act_func, output_layer_act_func=output_layer_act_func,
                     weights_range=weights_range)

    train = x_values[0::2]
    test = x_values[1::4]
    valid = x_values[3::4]
    train_target = y_values[0::2]
    test_target = y_values[1::4]
    valid_target = y_values[3::4]
    normalization_factor = max(max(y_values), abs(min(y_values)))
    data_set = TrainingDataSet([train], [train_target/normalization_factor])
    try:
        nn.train(data_set)
    except ConvergenceError:
        pass

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)

    ax1.plot(x_values, y_values, '--', x_values, list(itertools.chain(*nn.forward([x_values]))), 'r*')
    plt.show()


if __name__ == '__main__':
    sine_regression()
