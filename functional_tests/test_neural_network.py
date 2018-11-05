"""Module contains functional testing pf implemented neural network"""
import logging

import pytest

from NaiveNeurals.MLP.classes import NeuralNetwork
from NaiveNeurals.model.functions import load_model
from NaiveNeurals.training.misc import TrainingData

logger = logging.getLogger('classes')
logger.setLevel(logging.DEBUG)


def test_basic_pass_network():
    """ This test one forward pass and one backpropagation for example given here:

    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    Neural network is being created from provided model.
    """
    testing_model = {
        "input": {
            "node_0": {
                "weight_0": 1
            },
            "node_1": {
                "weight_0": 1
            }
        },
        "hidden_1": {
            "node_0": {
                "bias": 0.35,
                "weight_0": 0.15,
                "weight_1": 0.20,
            },
            "node_1": {
                "bias": 0.35,
                "weight_0": 0.25,
                "weight_1": 0.3
            }
        },
        "output": {
            "node_0": {
                "bias": 0.6,
                "weight_0": 0.4,
                "weight_1": 0.45
            },
            "node_1": {
                "bias": 0.6,
                "weight_0": 0.5,
                "weight_1": 0.55
            }
        }
    }
    testing_model = {
        "input": {
            "node_0": {
                "weight_0": 1
            },
            "node_1": {
                "weight_0": 1
            }
        },
        "hidden_1": {
            "node_0": {
                "bias": 1,
                "weight_0": 0.2,
                "weight_1": 0.6,
            },
            "node_1": {
                "bias": 1,
                "weight_0": 0.1,
                "weight_1": 0.8,
            },
            "node_2": {
                "bias": 1,
                "weight_0": 0.3,
                "weight_1": 0.7,
            }
        },
        "output": {
            "node_0": {
                "bias": 1,
                "weight_0": 0.4,
                "weight_1": 0.5,
                "weight_2": 0.9,
            },
        }
    }
    inputs = [[0.05], [0.1]]
    targets = [[0.01]]
    inputs = [[2, 9], [1, 5]]
    targets = [[92], [86]]


    network = load_model(testing_model)
    network.forward(inputs)
    network._backpropagate(targets)

    assert network.output_vector[0][0] == pytest.approx(0.7513650, 0.0001)    # calculated  0.75136507
    assert network.output_vector[1][0] == pytest.approx(0.7729284, 0.0001)    # calculated  0.772928465


def test_neural_network_xor():
    # XOR gate
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_outputs = [[0], [1], [1], [0]]
    training_data = TrainingData(training_inputs, expected_outputs)
    neural_network = NeuralNetwork(data_size=training_data.input_size,
                                   output_data_size=training_data.output_size,
                                   hidden_layers_number_of_nodes=[2])
    neural_network.initialize()
    _i = 0
    for random_set in training_data.get_randoms(max_count=10000):
        _i += 1
        neural_network.train(random_set.training_input, random_set.targeted_output)
    print('Finished within {} iters'.format(_i))
    print(neural_network.forward([0, 0]))
    print(neural_network.forward([0, 1]))
    print(neural_network.forward([1, 0]))
    print(neural_network.forward([1, 1]))
