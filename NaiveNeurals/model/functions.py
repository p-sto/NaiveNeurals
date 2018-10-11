"""Module containing definitions of functions for model related operations."""
import json
import logging
from collections import defaultdict
from typing import Dict

from Neurals.MLP.classes import NeuralNetwork


def export_model_to_json(neural_network: NeuralNetwork, file_name='model.json') -> None:
    model: Dict[str, Dict] = defaultdict(lambda: defaultdict(dict))
    for layer in neural_network:
        for ind, node in enumerate(layer):
            model['{}_layer'.format(layer.label)][ind]['bias'] = node.bias.value
            for iind, input in enumerate(node):
                model['{}_layer'.format(layer.label)][ind][iind] = input.weight.value
    logging.info('Extracting model to file: %s', file_name)
    with open(file_name, 'w+') as outfile:
        json.dump(model, outfile)
