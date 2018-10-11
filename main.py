import logging

from NaiveNeurals.MLP.classes import NeuralNetwork
from NaiveNeurals.model import export_model_to_json

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    input_data_vector = [1, 1, 2, 1, 3, 1, 5, 6, 1, 2, 4, 8, 8, 6, 4]
    network = NeuralNetwork()
    network.initialize(len(input_data_vector))
    network.feed_with_data(input_data_vector)
    export_model_to_json(network, file_name='model.json')
