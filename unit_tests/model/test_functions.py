"""Module containing tests for functions,py"""
from NaiveNeurals.model.functions import load_model, export_model


def test_load_model():
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
    network = load_model(testing_model)

    assert len(network.hidden_layers[0].nodes) == 2
    assert network.hidden_layers[0].nodes[0].bias.value == 1
    assert network.hidden_layers[0].nodes[0].inputs[0].weight.value == 0.15
    assert network.hidden_layers[0].nodes[0].inputs[1].weight.value == 0.25
    assert network.hidden_layers[0].nodes[1].inputs[0].weight.value == 0.2
    assert network.hidden_layers[0].nodes[1].inputs[1].weight.value == 0.3
    assert len(network.output_layer.nodes) == 2
    assert network.output_layer.nodes[0].bias.value == 1
    assert network.output_layer.nodes[0].inputs[0].weight.value == 0.4
    assert network.output_layer.nodes[0].inputs[1].weight.value == 0.5
    assert network.output_layer.nodes[1].inputs[0].weight.value == 0.45
    assert network.output_layer.nodes[1].inputs[1].weight.value == 0.55


def test_export_model():
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
    network = load_model(testing_model)
    exported = export_model(network)
    assert exported['input']['node_0']['weight_0'] == testing_model['input']['node_0']['weight_0']
    assert exported['hidden_1']['node_1']['weight_0'] == testing_model['hidden_1']['node_1']['weight_0']
