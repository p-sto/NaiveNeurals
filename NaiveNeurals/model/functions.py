"""Module containing definitions of functions for model related operations."""
import json
import logging
from collections import defaultdict
from typing import Dict

from NaiveNeurals.MLP.classes import NeuralNetwork


def export_model(neural_network: NeuralNetwork, file_name: str='model.json', export_to_file: bool = False) -> Dict:
    """Export neural network model to json

    :param neural_network: NeuralNetwork object
    :param file_name: targeted json file name
    :param export_to_file: denotes whether model should be exported to json file
    :return: dictionary with model
    """
    model: Dict[str, Dict] = defaultdict(lambda: defaultdict(dict))
    for layer in neural_network:
        for ind, node in enumerate(layer):
            if not node.is_input_node:
                model['{}'.format(layer.label)]['node_{}'.format(ind)]['bias'] = node.bias.value
            for iind, inp in enumerate(node):
                model['{}'.format(layer.label)]['node_{}'.format(ind)]['weight_{}'.format(iind)] = inp.weight.value
    if export_to_file:
        logging.info('Extracting model to file: %s', file_name)
        with open(file_name, 'w+') as outfile:
            json.dump(model, outfile)
    return model


def load_model(model: Dict) -> NeuralNetwork:
    """Load data from model and return initialized network

    :param model: dict with model
    :return: NeuralNetwork object
    """
    input_layer_data = model.get('input')
    hidden_layers_names = sorted([layer for layer in model if layer.startswith('hidden_')])
    hidden_layers_model = [model[name] for name in hidden_layers_names]
    output_layer_data = model.get('output')
    assert input_layer_data, 'Could not retrieve input layer data from model'
    assert hidden_layers_model, 'Could not retrieve hidden layers data from model'
    assert output_layer_data, 'Could not retrieve output layer data from model'
    data_size = len(input_layer_data)
    output_data_size = len(output_layer_data)
    output_layer_bias = output_layer_data['node_0'].get('bias')
    assert output_layer_bias, 'No bias for output layer'
    hidden_layers_number_of_nodes = [len(layer) for layer in hidden_layers_model]
    hidden_layers_biases = [node['node_0'].get('bias') for node in hidden_layers_model]
    assert hidden_layers_biases, 'No bias for hidden layer'
    logging.info('Creating network for model with: {} hidden layer(s)'.format(len(hidden_layers_number_of_nodes)))
    logging.info('Data size = {}'.format(data_size))
    logging.info('Output data size = {}'.format(output_data_size))
    logging.info('Hidden layer(s) number of nodes = {}'.format(hidden_layers_number_of_nodes))
    logging.info('Hidden layer biases = {}'.format(hidden_layers_biases))
    logging.info('Output layer bias = {}'.format(output_layer_bias))
    network = NeuralNetwork(data_size=data_size, output_data_size=output_data_size,
                            hidden_layers_number_of_nodes=hidden_layers_number_of_nodes,
                            hidden_layer_bias=hidden_layers_biases, output_layer_bias=output_layer_bias)
    network.initialize()
    for model_h_layer_name in hidden_layers_names:
        model_layer_data = model[model_h_layer_name]
        network_layer = network.get_layer(model_h_layer_name)
        for nodel_node_data, network_node in zip(model_layer_data, network_layer.nodes):
            # update network value bias with value from imported model
            network_node.bias.update(model_layer_data[nodel_node_data]['bias'])
            for indx, inp in enumerate(network_node):
                inp.weight.update(model_layer_data[nodel_node_data]['weight_{}'.format(indx)])
    for indx, node in enumerate(network.output_layer.nodes):
        node.bias.update(output_layer_bias)
        for inp_indx, inp in enumerate(node.inputs):
            inp.weight.update(output_layer_data['node_{}'.format(indx)]['weight_{}'.format(inp_indx)])
    return network
