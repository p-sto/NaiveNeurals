"""Module contains test for MLP classes"""
from unittest.mock import MagicMock

from NaiveNeurals.MLP.classes import Weight, Input, Bias, Output, PerceptronNode, MeshLayer, NeuralNetwork
from NaiveNeurals.MLP.functions import Sigmoid


def test_weight():
    """Test Weight object"""
    w = Weight(1.2)
    w.update(0.8)
    w.update(0.6)
    assert w.value == 0.6
    assert w.historical_values == [1.2, 0.8]


def test_perceptron_input():
    """Test perceptron's input object"""
    fake_perceptron = MagicMock()
    initial_weight = 1.2
    after_epoch_weight = 0.8
    inp = Input(fake_perceptron, initial_weight)
    inp.set_value(0.1)
    assert inp.value == 0.1
    assert inp.body == fake_perceptron
    assert inp.weight.value == initial_weight
    assert inp.is_connected is False
    inp.weight_value_after_epoch(after_epoch_weight)
    inp.update_weights()
    assert inp.weight.value == after_epoch_weight


def test_bias():
    """Test bias object"""
    bias = Bias(1.2)
    bias.update(0.8)
    bias.update(0.6)
    assert bias.value == 0.6
    assert bias.historical_values == [1.2, 0.8]


def test_output():
    """Test Output object"""
    fake_perceptron = MagicMock()
    fake_perceptron.net_value = 0.8
    fake_perceptron.activation_function = Sigmoid
    fake_perceptron_next = MagicMock()
    out = Output(fake_perceptron)
    out.connect(Input(fake_perceptron_next, 1.2))
    assert out.is_connected is True
    assert out.body == fake_perceptron
    assert out.value == Sigmoid.function(0.8)


def test_perceptron_node():
    """Test PerceptronNode object"""
    fake_mesh_layer = MagicMock
    input_vect = [0.1, 0.2, 0.3, 0.2, 0.1]
    node = PerceptronNode(5, -1, layer=fake_mesh_layer)
    node.feed_inputs(input_vect)
    expected_net_value = sum(input_vect) - 1
    assert node.activation_function == Sigmoid
    assert node.inputs[0].value == 0.1
    assert len(node.inputs) == 5
    assert node.net_value == expected_net_value
    assert node.output_value == Sigmoid.function(expected_net_value)
    assert [isinstance(elmn, Input) for elmn in node] == [True]*5


def test_mesh_layer():
    """Test MeshLayer object"""
    hidden_layer_nodes = 2
    output_layer_nodes = 1
    input_layer = MeshLayer('test_input_layer', data_size=5, nodes_number=3, is_input_layer=True)
    hidden_layer = MeshLayer('test_hidden_layer', data_size=len(input_layer.nodes), nodes_number=hidden_layer_nodes)
    output_layer = MeshLayer('test_output_layer', data_size=len(hidden_layer.nodes), nodes_number=output_layer_nodes)
    input_layer >> hidden_layer >> output_layer
    # if layer is configured as a input layer then it's input weights are set to 1
    assert input_layer.nodes[0].inputs[0].weight.value == 1
    assert input_layer.next_layer.next_layer == output_layer
    for node in input_layer:
        assert isinstance(node, PerceptronNode)
        assert node.output.is_connected is True
        assert len(node.output.following) == hidden_layer_nodes
    for node in hidden_layer:
        assert isinstance(node, PerceptronNode)
        assert node.output.is_connected is True
        assert len(node.output.following) == output_layer_nodes
        for inp in node:
            assert isinstance(inp, Input)
            assert inp.is_connected is True


def test_neural_network():
    """Test NeuralNetwork"""
    network = NeuralNetwork(data_size=2, output_data_size=1, hidden_layers_number_of_nodes=[2],
                            hidden_layer_bias=[0.4], output_layer_bias=0.8)
    network.initialize()
    assert [len(node.inputs) for node in network.input_layer.nodes] == [1, 1]
    assert [len(node.inputs) for node in network.hidden_layers[0].nodes] == [2, 2]
    assert [len(node.inputs) for node in network.output_layer.nodes] == [2]
