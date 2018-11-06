"""Module contains test for MLP classes"""

from NaiveNeurals.MLP.network import NeuralNetwork

inputs = [[0, 0, 1, 1], [0, 1, 0, 1]]
targets = [[0, 1, 1, 0]]
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
            "bias": 0,
            "weight_0": 5,
            "weight_1": 5,
        },
        "node_1": {
            "bias": 0,
            "weight_0": -5,
            "weight_1": -5,
        },
        "node_2": {
            "bias": 0,
            "weight_0": -5,
            "weight_1": -5,
        },

    },
    "output": {
        "node_0": {
            "bias": -15,
            "weight_0": 10,
            "weight_1": 10,
            "weight_2": 10,
        },
    },
    "hidden_act_func": "sigmoid",
    "output_act_func": "sigmoid",
}


def test_neural_network():
    """Test NeuralNetwork"""
    nn = NeuralNetwork()
    nn.setup_network(input_data_size=2, output_data_size=2, hidden_layer_number_of_nodes=3,
                     hidden_layer_bias=-1, output_layer_bias=-1)
    assert nn.hidden_layer.number_of_nodes == 3
    assert nn.hidden_layer.weights.ndim == 2
    assert nn.hidden_layer.bias == -1
    assert nn.hidden_layer.activation_function.label == 'sigmoid'
    assert nn.output_layer.number_of_nodes == 2
    assert nn.output_layer.bias == -1
    assert nn.output_layer.activation_function.label == 'sigmoid'


def test_neural_network_load_model():
    """Test loading of a model"""
    nn = NeuralNetwork()
    nn.load_model(testing_model)
