"""This file contains example script of training neural network for regression problem."""
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from NaiveNeurals.MLP.activation_functions import Linear, Tanh
from NaiveNeurals.MLP.network import NeuralNetwork, LearningConfiguration
from NaiveNeurals.data.data_generators import data_generator_for_regression
from NaiveNeurals.data.dataset import DataSet
from NaiveNeurals.utils import DataSeries, ConvergenceError


def sine_regression() -> None:
    """Example of regression problem for SINE."""

    x_values1, y_values1 = data_generator_for_regression(DataSeries.SINE_GAUSS, data_size=6)
    x_values2, y_values2 = data_generator_for_regression(DataSeries.SINE_GAUSS, data_size=8)
    x_values3, y_values3 = data_generator_for_regression(DataSeries.SINE_GAUSS, data_size=10)
    normalization_outputs = max(max(y_values1 + y_values2 + y_values3), abs(min(y_values1 + y_values2 + y_values3)))
    val_x, val_y = data_generator_for_regression(DataSeries.SINE, data_size=20)

    nn = NeuralNetwork()
    input_data_size = 1
    output_data_size = 1
    hidden_layer_number_of_nodes = 25
    hidden_layer_bias = 1
    output_layer_bias = -0.7
    weights_range = 1
    hidden_layer_act_func = Tanh()
    output_layer_act_func = Linear()

    learning_configuration = LearningConfiguration(learning_rate=0.01, target_error=0.003,
                                                   solver='GD_MOM', max_epochs=20_000)

    nn.setup_network(input_data_size=input_data_size, output_data_size=output_data_size,
                     hidden_layer_number_of_nodes=hidden_layer_number_of_nodes,
                     hidden_layer_bias=hidden_layer_bias, output_layer_bias=output_layer_bias,
                     hidden_layer_act_func=hidden_layer_act_func, output_layer_act_func=output_layer_act_func,
                     weights_range=weights_range)

    nn.set_learning_params(learning_configuration)

    train_data_set1 = DataSet([x_values1], [y_values1 / normalization_outputs])
    train_data_set2 = DataSet([x_values2], [y_values2 / normalization_outputs])
    train_data_set3 = DataSet([x_values3], [y_values3 / normalization_outputs])
    validation = DataSet([val_x], [val_y / normalization_outputs])
    try:
        nn.train_with_validation([train_data_set1, train_data_set2, train_data_set3], validation)
    except ConvergenceError:
        pass

    data_size = 50
    x = [val / data_size for val in range(data_size)]

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    data_set1_label = mpatches.Patch(color='cyan', label='data set 1')
    data_set2_label = mpatches.Patch(color='red', label='data set 2')
    data_set3_label = mpatches.Patch(color='blue', label='data set 3')
    validation_label = mpatches.Patch(color='magenta', label='validation set')
    trained_label = mpatches.Patch(color='black', label='trained network response')
    ax1.plot(x_values1, y_values1, 'c-',
             x_values2, y_values2, 'r-',
             x_values3, y_values3, 'b-',
             val_x, val_y, 'm-',
             x, list(itertools.chain(*nn.forward([x]))), 'k*')
    plt.legend(handles=[data_set1_label, data_set2_label, data_set3_label, validation_label, trained_label])
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(nn.convergence_profile, '-', nn.validation_profile, 'r--')
    plt.yscale('log')
    learning_err = mpatches.Patch(color='blue', label='learning err')
    validation_line = mpatches.Patch(color='red', label='validation err')
    plt.legend(handles=[learning_err, validation_line])
    plt.show()


if __name__ == '__main__':
    sine_regression()
