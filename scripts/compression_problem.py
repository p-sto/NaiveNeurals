"""Contains example of compression problem."""
import itertools
import os
from typing import Tuple, List
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from NaiveNeurals.MLP.activation_functions import Linear, Tanh
from NaiveNeurals.MLP.network import NeuralNetwork, LearningConfiguration
from NaiveNeurals.data.data_manipulators import normalise_data
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


def compression() -> None:
    """Compression with neural network is achieved with auto-associative networks.
    This networks are trained to predict targets being the same as inputs, though hidden layer
    has less nodes than input/output layer.

    TODO: Better results would be achieved with more than 1 hidden layer.
    """

    dobson_reading, sulphur_dioxide_level = read_data()
    dobson_reading = normalise_data(dobson_reading)

    input_vector_size = 18
    number_of_test_points = 700

    data_chunks_count = int(len(dobson_reading)/input_vector_size) - 1
    inputs = []
    real = []

    for ind in range(input_vector_size):
        inputs.append(dobson_reading[data_chunks_count * ind:data_chunks_count*(ind+1)])
    for ind in range(data_chunks_count):
        if ind < int(number_of_test_points/input_vector_size):
            real.append(dobson_reading[input_vector_size * ind:input_vector_size*(ind+1)])

    training = np.array(inputs)[:input_vector_size, :50]
    training2 = np.array(inputs)[:input_vector_size, 100:200]
    validation = np.array(inputs)[:input_vector_size, -200:]

    training_set = DataSet(training, training)
    training_set2 = DataSet(training2, training2)
    validation_set = DataSet(validation, validation)

    nn = NeuralNetwork()
    output_data_size = input_vector_size
    hidden_layer_number_of_nodes = 16
    hidden_layer_bias = 0
    output_layer_bias = 0
    hidden_layer_act_func = Tanh()
    output_layer_act_func = Linear()

    learning_configuration = LearningConfiguration(learning_rate=0.005, target_error=0.002,
                                                   solver='GD_MOM', max_epochs=1000)

    nn.setup_network(input_data_size=input_vector_size, output_data_size=output_data_size,
                     hidden_layer_number_of_nodes=hidden_layer_number_of_nodes,
                     hidden_layer_bias=hidden_layer_bias, output_layer_bias=output_layer_bias,
                     hidden_layer_act_func=hidden_layer_act_func, output_layer_act_func=output_layer_act_func)

    nn.set_learning_params(learning_configuration)

    try:
        nn.train_with_validation([training_set, training_set2], validation_set)
    except ConvergenceError:
        pass

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)

    predicted = list(itertools.chain(*([nn.forward(chunk) for chunk in real])))
    ax1.plot(list(itertools.chain(*real)), 'b.')
    ax1.plot(predicted, 'cx')
    learning_err = mpatches.Patch(color='blue', label='real data')
    validation_line = mpatches.Patch(color='cyan', label='predicted values')
    plt.legend(handles=[learning_err, validation_line])
    plt.show()


if __name__ == '__main__':
    compression()
