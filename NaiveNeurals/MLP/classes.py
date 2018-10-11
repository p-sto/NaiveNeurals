"""Implementation of perceptron neural network from scratch for educational purpose."""

import random
from collections import defaultdict

import numpy as np
import logging

from typing import List, Optional

from IteratorDecorator import iter_attribute
from NaiveNeurals.MLP.functions import Sigmoid, Function


class Weight:
    """Represents Weight object."""

    def __init__(self, init_value) -> None:
        """Initialize Weight object

        :param init_value: default initial value
        """
        self.value = init_value
        self.historical_values = []

    def update(self, new_val) -> None:
        """Update weight value

        Updates new weight to new value and stores previous value to statistical purposes

        :param new_val: new weight value
        """
        self.historical_values.append(self.value)
        self.value = new_val


class Input:
    """Represents perceptron input object.

    Every input has value multiplied by input's weight.

    -> Input value * Input weight value -> Perceptron

    """

    def __init__(self, body: 'PerceptronNode', initial_weight: float) -> None:
        """Initialize Input object

        :param body: reference to perceptron connected to Input
        :param initial_weight: float initial weight value
        """
        self.body = body
        self.value = None
        self.weight = Weight(initial_weight)
        self.value_to_be_set = None
        self.preceding: List[Output] = []   # list of Outputs from previous layer connected to this input

    @property
    def is_connected(self) -> bool:
        """Denotes whether outputs from previous layer are connected to this input.

        :return: True/False
        """
        return bool(self.preceding)

    def set_value(self, value: float) -> None:
        """Sets input value
        
        :param value: represents input value        
        """
        self.value = value

    def update_weights(self) -> None:
        """Update weight value related to input
        
        :param new_weigh: new value of weight
         
        """
        self.weight.update(self.value_to_be_set)

    def __repr__(self) -> str:
        return 'value={}, weight={}, is_connected={}, id={}'.format(self.value, self.weight.value,
                                                                    self.is_connected, id(self))


class Bias:
    """Represents BIAS input"""

    def __init__(self, initial_value: float) -> None:
        """Initialize Bias object

        :param initial_value: initial float value
        """
        self.value = initial_value
        self.historical_values = []

    def update(self, new_val: float) -> None:
        """Update bias value

        Updates new weight to new value and stores previous value to statistical purposes

        :param new_val: new weight value
        """
        self.historical_values.append(new_val)
        self.value = new_val


class Output:
    """Represents Perceptron's Output"""

    def __init__(self, body: 'PerceptronNode') -> None:
        """Initialize Output object

        :param body: reference to perceptron connected to Output
        """
        self.body = body
        self.following: List[Input] = []    # list of inputs of next layer connected to this output

    @property
    def is_connected(self):
        """Denotes whether output is connected to any input of next layer

        :return: True/False
        """
        return bool(self.following)

    @property
    def value(self) -> float:
        """Calculated output value

        :return: float output value
        """
        return self.body.activation_function.function(self.body.net_value)


@iter_attribute('inputs')
class PerceptronNode:
    """Perceptron model:

          ____________________
    -----| in1       |        |
    -----| in2       |        |
    **** |       NET |  OUT   | ---- out value
    -----| bias      |        |
         |___________|________|

    net_value = sum(inputs * weights)
    out_value = activation_function(net_value)

    """

    def __init__(self, input_data_size: int, bias_val: float = -1, activation_func: Function = Sigmoid,
                 layer: 'MeshLayer' = None, is_input_node: bool = False) -> None:
        self.inputs: List[Input] = [Input(self, random.uniform(-0.5, 0.5)) for _ in range(input_data_size)]
        self.output = Output(self)
        self.bias = Bias(bias_val) if bias_val else Bias(1.0)
        self.activation_function = activation_func
        self.layer = layer
        self._i = 0
        self.is_input_node = is_input_node
        # if it's input layer node then weights will be se to 1
        if self.is_input_node:
            [inp.weight.update(1) for inp in self.inputs]

    def feed_inputs(self, input_vec: List[float]) -> None:
        if len(input_vec) != len(self.inputs):
            description = 'Impedance mismatch!Data size = {}, inputs number = {}, ' \
                          'layer = {}!'.format(len(input_vec), len(self.inputs), self.layer)
            raise ValueError(description)
        [inp.set_value(val) for inp, val in zip(self.inputs, input_vec)]

    def connect_output(self, next_input: Input) -> None:
        self.output.following.append(next_input)
        next_input.preceding.append(self.output)

    @property
    def net_value(self) -> float:
        return sum([x.value for x in self.inputs]) + self.bias.value

    @property
    def output_value(self) -> float:
        return self.output.value

    def __str__(self) -> str:
        return 'Inputs: {} Following items: {}'.format(len(self.inputs), len(self.output.following))

    def __iter__(self) -> 'PerceptronNode':
        return self


@iter_attribute('nodes')
class MeshLayer:
    def __init__(self, label: str, data_size: int, nodes_number: int = 5,
                 activation_func: Function = Sigmoid, is_input_layer: bool = False, bias: Optional[float] = None):
        self.label = label
        self.inputs_in_node = data_size
        self.nodes = [PerceptronNode(input_data_size=data_size, activation_func=activation_func, layer=self,
                                     is_input_node=is_input_layer, bias_val=bias) for _ in range(nodes_number)]
        self.next_layer = None
        self.previous_layer = None
        self._i = 0
        self.activation_func = activation_func

    def __rshift__(self, other: 'MeshLayer'):
        assert isinstance(other, MeshLayer), 'Provided element is not instance of MeshLayer.'
        self.next_layer = other
        other.previous_layer = self
        for this_node in self.nodes:
            for next_node in other.nodes:
                for next_inp in next_node.inputs:
                    if not next_inp.is_connected:
                        this_node.connect_output(next_inp)
                        # ok, connect output of a node to first unconnected
                        # input from the next layer
                        break
        return other

    def forward(self, input_data: List[float]) -> None:
        if self.previous_layer is None:
            # this is first layer
            for in_data in input_data:
                for ind, node in enumerate(self.nodes):
                    node.feed_inputs([in_data])
        else:
            for ind, node in enumerate(self.nodes):
                node.feed_inputs(input_data)
        layer_output = [node.output_value for node in self.nodes]
        if self.next_layer:
            self.next_layer.forward(layer_output)

    def __str__(self) -> str:
        return '{} layer: {} node(s) with {} input(s), id={}'.format(self.label, len(self.nodes),
                                                                     self.inputs_in_node, id(self))

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self) -> 'MeshLayer':
        return self


@iter_attribute('_layers')
class NeuralNetwork:

    _MAX_EPOCHS = 100
    LEARNING_RATE = 0.15

    def __init__(self,
                 hidden_layers_number_of_nodes: List[Optional[int]] = None,
                 hidden_layer_bias: Optional[float] = None,
                 output_layer_bias: Optional[float] = None) -> None:
        """

        :param data_size:
        :param hidden_layers_number_of_nodes:  [2, 3] - hidden layer #1 - 2 nodes, hidden layer #2 - 3 nodes
        :param hidden_layer_bias:
        :param output_layer_bias:
        """
        self.input_layer: Optional[MeshLayer] = None
        self.hidden_layers: List[MeshLayer] = []
        self.output_layer: Optional[MeshLayer] = None
        self.total_epochs = 0
        self.error_rates = []
        self._layers: List[MeshLayer] = []
        self._i = 0
        self.hidden_layer_bias = hidden_layer_bias if hidden_layer_bias else -1
        self.output_layer_bias = output_layer_bias if output_layer_bias else -1
        self.hidden_layers_number_of_nodes = [5] if not hidden_layers_number_of_nodes else hidden_layers_number_of_nodes

    def initialize(self, data_size: int, output_data_size: int) -> None:
        self.hidden_layers = []
        self.input_layer = MeshLayer(label='input',
                                     data_size=1,
                                     nodes_number=data_size,
                                     is_input_layer=True,
                                     bias=1.0)
        prev_layer = self.input_layer
        for ind in range(len(self.hidden_layers_number_of_nodes)):
            hidden_layer = MeshLayer(label='hidden_{}'.format(ind + 1),
                                     data_size=len(prev_layer.nodes),
                                     nodes_number=self.hidden_layers_number_of_nodes[ind],
                                     bias=self.hidden_layer_bias)

            self.hidden_layers.append(hidden_layer)
            prev_layer >> hidden_layer
            prev_layer = hidden_layer
        self.output_layer = MeshLayer(label='output',
                                      data_size=len(prev_layer.nodes),
                                      nodes_number=output_data_size,
                                      bias=self.output_layer_bias)
        prev_layer >> self.output_layer
        self._layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        logging.debug('Initialized Neural Network with parameters:')
        logging.debug('Input layer nodes = %s', len(self.input_layer.nodes))
        logging.debug(' * Per node inputs: %s', len(self.input_layer.nodes[0].inputs))

        for hidden_layer in self.hidden_layers:
            logging.debug('%s nodes = %s', hidden_layer.label, len(hidden_layer.nodes))
            logging.debug(' * Per node inputs: %s', len(hidden_layer.nodes[0].inputs))
        logging.debug('Output layer nodes = %s', len(self.output_layer.nodes))
        logging.debug(' * Per node inputs: %s', len(self.output_layer.nodes[0].inputs))

    def forward(self, input_data: List[float]) -> np.array:
        self.input_layer.forward(input_data)
        return self.output_vector

    def _backpropagate(self, targets: List[float]):
        output_errors = []
        derivative_values = defaultdict(list)
        delta_values = {}
        for ind, node in enumerate(self.output_layer.nodes):
            output_errors.append(node.output_value - targets[ind])
            derivative_values[self.output_layer].append(node.activation_function.prime(node.output_value))
            delta_values[ind] = output_errors[ind] * derivative_values[self.output_layer][ind]
            for inp in node.inputs:
                # new weight value is derived from difference between current value and
                # multiplication of learning rate, calculated delta AND output from previous layer (so input val)
                inp.value_to_be_set = inp.weight.value - self.LEARNING_RATE * delta_values[ind] * inp.value
        for layer in self.hidden_layers:
            for node in layer.nodes:
                delta = 0
                # this is crazy!
                for ind, err_value in enumerate(output_errors):
                    delta += err_value * layer.next_layer.activation_func.function(node.output.following[ind].value) *\
                             node.output.following[ind].weight.value
                for inp in node.inputs:
                    delta = delta * layer.activation_func.function(node.output.value) * inp.value
                    inp.value_to_be_set = inp.weight.value - self.LEARNING_RATE * delta * inp.value

        # we have all weights calculated so now we can finally update them
        for node in self.output_layer:
            for inp in node:
                inp.update_weights()
        for layer in self.hidden_layers:
            for node in layer:
                for inp in node:
                    inp.update_weights()

    @property
    def output_vector(self) -> np.array:
        return np.asarray([node.output_value for node in self.output_layer.nodes])

    def train(self, sample: List[float], targets: List[float], error_rate: float = 0.2):
        self.forward(sample)
        self._backpropagate(targets)

    def __iter__(self) -> 'NeuralNetwork':
        return self

"""
training_sets = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]
training_inputs, training_outputs = random.choice(training_sets)
"""