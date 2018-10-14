"""Module contains functional testing pf implemented neural network"""
import logging

from NaiveNeurals.MLP.classes import NeuralNetwork
from NaiveNeurals.training.misc import TrainingData

logger = logging.getLogger('classes')
logger.setLevel(logging.DEBUG)


def test_neural_network():
    # XOR gate
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_outputs = [[0], [1], [1], [0]]
    training_data = TrainingData(training_inputs, expected_outputs)
    neural_network = NeuralNetwork(input_layer_nodes_number=2, hidden_layers_number_of_nodes=[3])
    neural_network.initialize(data_size=training_data.input_size, output_data_size=training_data.output_size)
    _i = 0
    while neural_network.error_rate([0, 0], [0]) > 0.05:
        for random_set in training_data.get_randoms():
            print('Err rate = {}'.format(neural_network.error_rate([0, 0], [0])))
            neural_network.train(random_set.training_input, random_set.targeted_output)

    neural_network.forward([0, 0])
    print(neural_network.output_vector)
    neural_network.forward([0, 1])
    print(neural_network.output_vector)
    neural_network.forward([1, 0])
    print(neural_network.output_vector)
    neural_network.forward([1, 1])
    print(neural_network.output_vector)
