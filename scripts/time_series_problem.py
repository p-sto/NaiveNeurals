"""Example of network training for time series prediction using ozon layer dataset"""
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np


from NaiveNeurals.MLP.activation_functions import Linear, Sigmoid
from NaiveNeurals.MLP.network import NeuralNetwork, LearningConfiguration
from NaiveNeurals.data.data_manipulators import prepare_time_series_data, normalise_data, get_time_series_data_slice
from NaiveNeurals.data.dataset import DataSet
from NaiveNeurals.utils import ConvergenceError


def read_data() -> Tuple[List, List]:
    """Read data from source file"""
    dobson_reading = []
    sulphur_dioxide_level = []
    with open(os.path.abspath(os.path.dirname(__file__)) + '/datasets/ozon_layer.txt') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            dobson_units, sulphur_level = line.split()
            dobson_reading.append(float(dobson_units))
            sulphur_dioxide_level.append(float(sulphur_level))
    return dobson_reading, sulphur_dioxide_level


def train_model() -> None:
    """Train neural network for ozon layer data."""

    dobson_reading, sulphur_dioxide_level = read_data()
    dobson_reading = normalise_data(dobson_reading)

    input_vector_size = 3   # describes how many data samples should be used to predict output
    data_interval = 3       # describes what interval between data every sample in input data

    inputs, targets = prepare_time_series_data(dobson_reading, input_vector_size, data_interval)

    dobson_slice = np.array(dobson_reading[2200:2700])      # keep aligned with slice for test_data
    test_data, test_targets = get_time_series_data_slice(inputs, targets, 2200, 2700)
    training_data, training_targets = get_time_series_data_slice(inputs, targets, 0, 1500)
    validation_data, validation_targets = get_time_series_data_slice(inputs, targets,
                                                                     len(dobson_reading) - 1200,
                                                                     len(dobson_reading))

    test_dataset = DataSet(test_data, test_targets)
    training_dataset = DataSet(training_data, training_targets)
    validation_dataset = DataSet(validation_data, validation_targets)

    nn = NeuralNetwork()
    output_data_size = 1
    hidden_layer_number_of_nodes = 6
    hidden_layer_bias = 0.09
    output_layer_bias = 0
    hidden_layer_act_func = Sigmoid()
    output_layer_act_func = Linear()

    learning_configuration = LearningConfiguration(learning_rate=0.0005, target_error=0.002,
                                                   solver='GD_MOM', max_epochs=200)

    nn.setup_network(input_data_size=input_vector_size, output_data_size=output_data_size,
                     hidden_layer_number_of_nodes=hidden_layer_number_of_nodes,
                     hidden_layer_bias=hidden_layer_bias, output_layer_bias=output_layer_bias,
                     hidden_layer_act_func=hidden_layer_act_func, output_layer_act_func=output_layer_act_func)

    nn.set_learning_params(learning_configuration)

    try:
        nn.train_with_validation([training_dataset], validation_dataset)
    except ConvergenceError:
        pass


    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(dobson_slice, 'b.')
    ax1.plot(nn.forward(test_dataset.inputs), 'cx')
    learning_err = mpatches.Patch(color='blue', label='real data')
    validation_line = mpatches.Patch(color='cyan', label='predicted values')
    plt.legend(handles=[learning_err, validation_line])
    plt.show()


if __name__ == '__main__':
    train_model()
