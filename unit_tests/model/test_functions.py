"""Module containing tests for functions,py"""
from NaiveNeurals.model.functions import load_model, export_model_to_json


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
    export_model_to_json(network)
