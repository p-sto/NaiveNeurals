"""Implementation of perceptron neural network from scratch for educational purpose."""

import random
from collections import defaultdict

import numpy as np
import logging

from typing import List, Optional

from IteratorDecorator import iter_attribute
from NaiveNeurals.MLP.functions import Sigmoid, Function, calculate_error

logger = logging.getLogger('classes')
logger.setLevel(logging.WARNING)


class Weight:
    """Represents Weight object."""

    def __init__(self, init_value: float) -> None:
        """Initialize Weight object

        :param init_value: default initial value
        """
        self.value = init_value
        self.historical_values = []

    def update(self, new_val: float) -> None:
        """Update weight value

        Updates new weight to new value and stores previous value to statistical purposes

        :param new_val: new weight value
        """
        self.historical_values.append(self.value)
        self.value = new_val

    def __str__(self) -> str:
        """Override default string representation

        :return: string
        """
        return '{}-{}'.format('Weigh', id(self))


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
        self._value_to_be_set = None
        self.preceding: List[Output] = []   # list of Outputs from previous layer connected to this input

    @property
    def is_connected(self) -> bool:
        """Denotes whether outputs from previous layer are connected to this input.

        :return: True/False
        """
        return bool(self.preceding)

    def set_value(self, value: float) -> None:
        """Sets input value.

        :param value: represents input value
        """
        self.value = value

    def weight_value_after_epoch(self, value_to_be_set: float) -> None:
        """Register weigh value which will be updated after training epoch"""
        self._value_to_be_set = value_to_be_set

    def update_weights(self) -> None:
        """Update weight value related to input"""
        self.weight.update(self._value_to_be_set)

    def __repr__(self) -> str:
        return 'value={}, weight={}, is_connected={}, id={}'.format(self.value, self.weight.value,
                                                                    self.is_connected, id(self))


class Bias(Weight):
    """Represents BIAS input"""

    def __str__(self) -> str:
        """Override string representation

        :return: string
        """
        return '{}-{}'.format('Bias', id(self))


class Output:
    """Represents Perceptron's Output"""

    def __init__(self, body: 'PerceptronNode') -> None:
        """Initialize Output object

        :param body: reference to perceptron connected to Output
        """
        self.body = body
        self.following: List[Input] = []    # list of inputs of next layer connected to this output

    def connect(self, inp: Input) -> None:
        """Connect outut to input of next perceptron

        :param inp: Input of next connected perceptron
        """
        self.following.append(inp)

    @property
    def is_connected(self) -> bool:
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

    # pylint: disable=too-many-arguments
    def __init__(self, input_data_size: int, bias_val: float = -1, activation_func: Function = Sigmoid,
                 layer: 'MeshLayer' = None, is_input_node: bool = False) -> None:
        """Initialize PerceptronNode object

        :param input_data_size: length of input data vector
        :param bias_val: bias value
        :param activation_func: type of activation function
        :param layer: MeshLayer object to which this Percetron belongs
        :param is_input_node: boolean value denoting whether this perceptron is an input node
        """
        self.inputs: List[Input] = [Input(self, random.uniform(-0.5, 0.5)) for _ in range(input_data_size)]
        self.output = Output(self)
        self.bias = Bias(bias_val) if bias_val else Bias(1.0)
        self.activation_function = activation_func
        self.layer = layer
        self.is_input_node = is_input_node
        # if it's input layer node then weights will be se to 1
        if self.is_input_node:
            _ = [inp.weight.update(1) for inp in self.inputs]

    def feed_inputs(self, input_vec: List[float]) -> None:
        """Fill inputs with data

        :param input_vec: input vector of floats
        """
        if len(input_vec) != len(self.inputs):
            description = 'Impedance mismatch! Data size = {}, inputs number = {}, ' \
                          'layer = {}!'.format(len(input_vec), len(self.inputs), self.layer)
            raise ValueError(description)
        _ = [inp.set_value(val) for inp, val in zip(self.inputs, input_vec)]

    def connect_output(self, next_input: Input) -> None:
        """Connect output to next inputs

        :param next_input:
        """
        self.output.connect(next_input)
        next_input.preceding.append(self.output)

    @property
    def net_value(self) -> float:
        """Return net value

        :return: float net value
        """
        return sum([x.value for x in self.inputs]) + self.bias.value

    @property
    def output_value(self) -> float:
        """Return output value

        :return: float value
        """
        return self.output.value

    def __str__(self) -> str:
        """Override string representation

        :return: string
        """
        return 'Inputs: {} Following items: {}'.format(len(self.inputs), len(self.output.following))


@iter_attribute('nodes')
class MeshLayer:
    """MeshLayer consists of several nodes"""

    # pylint: disable=too-many-arguments
    def __init__(self, label: str, data_size: int, nodes_number: int = 5, activation_func: Function = Sigmoid,
                 is_input_layer: bool = False, bias: Optional[float] = None) -> None:
        """Initialize MeshLayer object

        :param label: name of layer
        :param data_size: size of input vector length
        :param nodes_number: number of perceptron nodes in layer
        :param activation_func: activation function
        :param is_input_layer: denote whether this is input layer or not
        :param bias: optional bias input value
        """
        self.label = label
        self.inputs_in_node = data_size
        self.nodes = [PerceptronNode(input_data_size=data_size, activation_func=activation_func, layer=self,
                                     is_input_node=is_input_layer, bias_val=bias) for _ in range(nodes_number)]
        self.next_layer: Optional['MeshLayer'] = None
        self.previous_layer = None
        self._i = 0
        self.activation_func = activation_func

    def __rshift__(self, other: 'MeshLayer') -> 'MeshLayer':
        """Override right shift operator to allow layer easy connection.

        input_layer >> hidden_layer >> output_layer

        :param other: MeshLayer which inputs has to be connected to this layer output
        :return: MeshLayer layer object
        """
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
        """Push data into network

        :param input_data:
        :return:
        """
        logger.debug('Layer = %s, input data is: %s', self.label, input_data)
        for ind, node in enumerate(self.nodes):
            node.feed_inputs(input_data)
        layer_output = [node.output_value for node in self.nodes]
        if self.next_layer:
            self.next_layer.forward(layer_output)

    def __str__(self) -> str:
        """Override string representation

        :return: string
        """
        return '{} layer: {} node(s) with {} input(s), id={}'.format(self.label, len(self.nodes),
                                                                     self.inputs_in_node, id(self))

    def __repr__(self) -> str:
        """Override __repr__ representation

        :return: string
        """
        return self.__str__()


@iter_attribute('_layers')
class NeuralNetwork:
    """Connects multiple mesh layers into one network. Delivers methods for learning."""

    _MAX_EPOCHS = 100
    LEARNING_RATE = 0.15

    def __init__(self,
                 input_layer_nodes_number: int,
                 hidden_layers_number_of_nodes: Optional[List[int]] = None,
                 hidden_layer_bias: Optional[List[float]] = None,
                 output_layer_bias: Optional[float] = None) -> None:
        """Initialize NeuralNetwork object

        :param input_layer_nodes_number: number of nodes in input layer
        :param hidden_layers_number_of_nodes:  [2, 3] - hidden layer #1 - 2 nodes, hidden layer #2 - 3 nodes
        :param hidden_layer_bias: bias values for hidden layers
        :param output_layer_bias: bias value for output layer
        """
        self.input_layer_nodes_number = input_layer_nodes_number
        self.input_layer: Optional[MeshLayer] = None
        self.hidden_layers: List[MeshLayer] = []
        self.output_layer: Optional[MeshLayer] = None
        self.total_epochs = 0
        self.error_rates = []
        self._layers: List[MeshLayer] = []
        self.hidden_layer_bias = hidden_layer_bias if hidden_layer_bias else [-1]
        self.output_layer_bias = output_layer_bias if output_layer_bias else -1
        self.hidden_layers_number_of_nodes = [5] if not hidden_layers_number_of_nodes else hidden_layers_number_of_nodes
        assert len(self.hidden_layer_bias) == len(self.hidden_layers_number_of_nodes)

    def initialize(self, data_size: int, output_data_size: int) -> None:
        """Initialize functionally NeuralNetwork

        :param data_size: size of input vector
        :param output_data_size: size of output vector
        """
        self.hidden_layers = []
        self.input_layer = MeshLayer(label='input',
                                     data_size=data_size,
                                     nodes_number=self.input_layer_nodes_number,
                                     is_input_layer=True,
                                     bias=1.0)
        prev_layer = self.input_layer
        for ind in range(len(self.hidden_layers_number_of_nodes)):
            hidden_layer = MeshLayer(label='hidden_{}'.format(ind + 1),
                                     data_size=len(prev_layer.nodes),
                                     nodes_number=self.hidden_layers_number_of_nodes[ind],
                                     bias=self.hidden_layer_bias[ind])

            self.hidden_layers.append(hidden_layer)
            prev_layer >> hidden_layer
            prev_layer = hidden_layer
        self.output_layer = MeshLayer(label='output',
                                      data_size=len(prev_layer.nodes),
                                      nodes_number=output_data_size,
                                      bias=self.output_layer_bias)
        prev_layer >> self.output_layer
        self._layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        if logger.level == logging.DEBUG:
            logger.debug('Initialized Neural Network with parameters:')
            logger.debug('Input layer nodes = %s', len(self.input_layer.nodes))
            logger.debug(' * Per node inputs: %s', len(self.input_layer.nodes[0].inputs))

            for hidden_layer in self.hidden_layers:
                logger.debug('%s nodes = %s', hidden_layer.label, len(hidden_layer.nodes))
                logger.debug(' * Per node inputs: %s', len(hidden_layer.nodes[0].inputs))
            logger.debug('Output layer nodes = %s', len(self.output_layer.nodes))
            logger.debug(' * Per node inputs: %s', len(self.output_layer.nodes[0].inputs))

    def forward(self, input_data: List[float]) -> np.array:
        """Push data forward through neural network

        :param input_data: data from input vector
        :return: output vector representing values calculated by network
        """
        self.input_layer.forward(input_data)
        return self.output_vector

    def _backpropagate(self, targets: List[float]) -> None:
        """Perform back-propagation

        :param targets: list of targeted values used for learning
        """
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
                inp.weight_value_after_epoch(inp.weight.value - self.LEARNING_RATE * delta_values[ind] * inp.value)
        for layer in self.hidden_layers:
            for node in layer.nodes:
                delta = 0
                # this is crazy!
                for ind, err_value in enumerate(output_errors):
                    delta += err_value * layer.next_layer.activation_func.function(node.output.following[ind].value) *\
                             node.output.following[ind].weight.value
                for inp in node.inputs:
                    delta = delta * layer.activation_func.function(node.output.value) * inp.value
                    inp.weight_value_after_epoch(inp.weight.value - self.LEARNING_RATE * delta * inp.value)

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
        """Return calculated output vector

        :return: np.array object
        """
        return np.asarray([node.output_value for node in self.output_layer.nodes])

    def train(self, sample: List[float], targets: List[float], error_rate: float = 0.2) -> None:
        """Perform training of neural network

        :param sample: data to be pushed forward through neural network
        :param targets: targeted values to be achieved for given input data
        :param error_rate: targeted error rate between output vector and input data
        """
        self.forward(sample)
        self._backpropagate(targets)

    def error_rate(self, inputs: List[float], targeted: List[float]) -> float:
        """Get error rate between provided input and output

        :param inputs: vector of inputs
        :param targeted: vector of outputs
        :return: float number
        """
        self.forward(inputs)
        return calculate_error(self.output_vector, np.array(targeted))
