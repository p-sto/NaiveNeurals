"""Module contains functional testing pf implemented neural network"""
import logging

import pytest

from NaiveNeurals.MLP.classes import NeuralNetwork
from NaiveNeurals.data.trained_models import XOR_MODEL_1

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
        },
        "hidden_act_func": "sigmoid",
        "output_act_func": "sigmoid",
    }
    inputs = [[0.05], [0.1]]

    nn = NeuralNetwork()
    nn.load_model(testing_model)
    nn.forward(inputs)

    assert nn.output[0][0] == pytest.approx(0.7513650, 0.0001)    # calculated  0.75136507
    assert nn.output[0][1] == pytest.approx(0.7729284, 0.0001)    # calculated  0.772928465


def test_neural_network_xor():
    # XOR gate
    nn = NeuralNetwork()
    nn.load_model(XOR_MODEL_1)
    assert nn.forward([[1], [1]])[0][0] == pytest.approx(0, abs=1e-1)
    assert nn.forward([[1], [0]])[0][0] == pytest.approx(1, abs=1e-1)
    assert nn.forward([[0], [1]])[0][0] == pytest.approx(1, abs=1e-1)
    assert nn.forward([[0], [0]])[0][0] == pytest.approx(0, abs=1e-1)
