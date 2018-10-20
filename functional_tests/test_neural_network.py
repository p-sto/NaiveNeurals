"""Module contains functional testing pf implemented neural network"""
import logging

from NaiveNeurals.MLP.classes import NeuralNetwork
from NaiveNeurals.model import export_model_to_json
from NaiveNeurals.training.misc import TrainingData

logger = logging.getLogger('classes')
logger.setLevel(logging.DEBUG)


def test_basic_pass_network():
    """ This test one forward pass and one backpropagation for example given here:

    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    3 layer neural network is create with input layer, hidden layer and output layer
    """
    training_inputs = [[0.05, 0.1]]
    expected_outputs = [[0.01, 0.99]]
    training_data = TrainingData(training_inputs, expected_outputs)
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
                "weight_0": 0.15,
                "weight_1": 0.25
            },
            "node_1": {
                "bias": 1,
                "weight_0": 0.2,
                "weight_1": 0.3
            }
        },
        "output": {
            "node_0": {
                "bias": 1,
                "weight_0": 0.4,
                "weight_1": 0.5
            },
            "node_1": {
                "bias": 1,
                "weight_0": 0.45,
                "weight_1": 0.55
            }
        }
    }

    neural_network = NeuralNetwork(data_size=training_data.input_size, output_data_size=training_data.output_size,
                                   hidden_layers_number_of_nodes=[2])
    neural_network.initialize()
    export_model_to_json(neural_network)


def test_neural_network_xor():
    # XOR gate
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_outputs = [[0], [1], [1], [0]]
    training_data = TrainingData(training_inputs, expected_outputs)
    neural_network = NeuralNetwork(data_size=training_data.input_size, output_data_size=training_data.output_size,
                                   hidden_layers_number_of_nodes=[3])
    neural_network.initialize()
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
