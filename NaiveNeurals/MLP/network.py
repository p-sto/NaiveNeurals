"""Implementation of perceptron neural network from scratch for educational purpose."""
import logging
from numbers import Real

from typing import Optional, Dict, Any, Union, List, cast
import numpy as np

from NaiveNeurals.MLP.functions import Sigmoid, Function, calculate_error, get_activation_function
from NaiveNeurals.data.dataset import TrainingDataSet
from NaiveNeurals.utils import ErrorAlgorithm, InitialisationError

logging.basicConfig()
logger = logging.getLogger('network')
logger.setLevel(logging.INFO)
np.random.seed(1)


class Layer:
    """Represents neural network's layer"""

    def __init__(self, number_of_nodes: int, bias: float,
                 weights: np.array, activation_function: Function) -> None:
        """Initialise Layer object

        :param number_of_nodes: number of nodes within layer
        :param bias: bias value
        :param activation_function: type of activation function
        :param weights: array with weights
        """
        self.number_of_nodes = number_of_nodes
        self.bias = bias
        self.activation_function = activation_function
        self.weights = weights
        self.node_values = None
        self.output = None


class NeuralNetwork:
    """Represents neural network object.

    Perceptron node consists of inputs (dendrites) with weights, net (soma) and out (axon). In biology, connection
    between one perceptron and next one is called synapse.

            ____________________
    --W1---| in1       |        |
    --W2---| in2       |        |
     ***** |       NET |  OUT   | ---- out value -> connection (synapse) -> input_1 of next perceptron
    --WN---| inN       |        |
      ---->|bias(fixed)|        |
           |___________|________|

    Each perceptron consists of N inputs and 1 output.

    Perceptron's NET values is calculated as:
        net_value = sum(inputs * weights) + bias

        bias is a fixed value added to sum of inputs, it's the same for all nodes within one layer.

    Percetpron's OUTPUT can be calculated using formula:
        out_value = activation_function(net_value)

    Characteristics of activation_function contributes greatly to output of neuron. E.g. Sigmoid function is
    useful for classification problems (where output is supposed to be in one of known "states" like 0, 1).

    This Neural Network is implemented as 3 layer network.

    Input layer - is only responsible for passing values to every node of a hidden layer.
                    Every node within this layer has only one input, so number of nodes in input layer
                    is equal to data size (if we wish to implement XOR gate with two values then there will be 2 nodes
                    in input layer and every node will have 1 input).
                    This layer does not modify input value in any way (weights can be treated as equal to 1).

    Hidden layer - Each node has N inputs connected with OUTPUTS of every node from INPUT layer (thus each
                    preceptron within HIDDEN layer has as many inputs as input layer data size is).

    Output layer - Amount of perceptron nodes is equal to size of output data vector. Every node within OUTPUT layer
                    has Z inputs (which is equal to amount of nodes in HIDDEN layer).

    If we would wish to implement XOR gate for 2 input values X, Y, with 2 nodes in hidden layer, we will obtain:
    1. Input layer with 2 nodes (one for X and one for Y), each will have one input.
    2. Hidden layer with 2 nodes, each node will have 2 inputs (since there are 2 nodes in input layer) and every input
        will have it's weight. Each node will also have bias input.
    3. Output layer with 1 node (as output of a gate is singular value). Node will have two inputs (plus bias as third).
    """

    LEARNING_RATE = 0.3
    MAX_EPOCHS = 10_000
    TARGETED_ERROR_RATE = 0.001

    def __init__(self) -> None:
        """Initialise neural network"""
        self._was_initialized = False

        # not defining input layer as it's only passes values to hidden layer
        self.hidden_layer: Optional[Layer] = None
        self.output_layer: Optional[Layer] = None
        self.input_data_size: Optional[int] = None
        self._convergence_profile: List[float] = []

    @property
    def output(self) -> np.array:
        """Return neural network output vector.

        :return: vector of floats
        """
        if not self.output_layer:
            raise InitialisationError
        return self.output_layer.output

    @property
    def convergence_profile(self) -> List[float]:
        """Get convergence profile

        :return: list of floats
        """
        return self._convergence_profile

    def errors(self, targets: np.array) -> np.array:
        """Return vector of errors for particular learning data.

        :param targets: array with target values
        :return: vector of floats
        """
        return calculate_error(self.output, targets.T, func_type=ErrorAlgorithm.SQR)

    def cumulative_error_rate(self, targets: np.array) -> float:
        """Return cumulative error rate for all training data.

        :param targets: array with target values
        :return: float value
        """
        return sum(self.errors(targets))

    # pylint: disable=too-many-locals
    def load_model(self, _model: Dict) -> None:
        """Load neural network parameters from model.

        :param _model: dictionary representing model
        """
        model = dict(_model)
        hidden_layer_act_func = get_activation_function(model.pop('hidden_act_func'))
        output_layer_act_func = get_activation_function(model.pop('output_act_func'))
        input_layer_data = model.get('input')

        hidden_layers_model = model.get('hidden_1')
        output_layer_data = model.get('output')
        assert input_layer_data, 'Could not retrieve input layer data from model'
        assert hidden_layers_model, 'Could not retrieve hidden layers data from model'
        assert output_layer_data, 'Could not retrieve output layer data from model'
        data_size = len(input_layer_data)
        output_data_size = len(output_layer_data)
        output_layer_bias = output_layer_data['node_0'].get('bias')
        assert output_layer_bias is not None, 'No bias for output layer'
        hidden_layers_number_of_nodes = len(hidden_layers_model)
        hidden_layers_biases = hidden_layers_model['node_0']['bias']
        assert hidden_layers_biases is not None, 'No bias for hidden layer'
        logger.info('Data size = %s', data_size)
        logger.info('Output data size = %s', output_data_size)
        logger.info('Hidden layer number of nodes = %s', hidden_layers_number_of_nodes)
        logger.info('Hidden layer biases = %s', hidden_layers_biases)
        logger.info('Output layer bias = %s', output_layer_bias)

        hidden_layer_weights = np.zeros((data_size, hidden_layers_number_of_nodes))
        output_layer_weights = np.zeros((hidden_layers_number_of_nodes, output_data_size))

        self.input_data_size = data_size
        self.hidden_layer = Layer(hidden_layers_number_of_nodes, hidden_layers_biases,
                                  hidden_layer_weights, hidden_layer_act_func)
        self.output_layer = Layer(output_data_size, output_layer_bias, output_layer_weights,
                                  output_layer_act_func)

        for n_ind, node in enumerate(model['hidden_1']):
            weights = sorted([x for x in model['hidden_1'][node] if x != 'bias'])
            for w_ind, weight in enumerate(weights):
                self.hidden_layer.weights[w_ind][n_ind] = model['hidden_1'][node][weight]

        for n_ind, node in enumerate(model['output']):
            weights = sorted([x for x in model['output'][node] if x != 'bias'])
            for w_ind, weight in enumerate(weights):
                self.output_layer.weights[w_ind][n_ind] = model['output'][node][weight]

        self._was_initialized = True

    def export_model(self) -> Dict[str, Any]:
        """Export model to dictionary.

        :return: dictionary with model
        """
        if not self._was_initialized:
            raise InitialisationError
        self.hidden_layer = cast(Layer, self.hidden_layer)
        self.output_layer = cast(Layer, self.output_layer)
        assert self.input_data_size
        model: Dict = {'input': {}, 'hidden_1': {}, 'output': {}}

        input_weights = {}
        for ind in range(self.input_data_size):
            input_weights['node_{}'.format(ind)] = {'weight_0': 1}
        model['input'].update(input_weights)

        hidden_weights = {}
        for ind in range(self.hidden_layer.weights.T.shape[0]):
            hidden_weights['node_{}'.format(ind)] = {'bias': self.hidden_layer.bias}
        for col_ind, column in enumerate(self.hidden_layer.weights.T):
            for row_ind, val in enumerate(column):
                hidden_weights['node_{}'.format(col_ind)]['weight_{}'.format(row_ind)] = val
        model['hidden_1'].update(hidden_weights)

        output_weights = {}
        for ind in range(self.output_layer.weights.T.shape[0]):
            output_weights['node_{}'.format(ind)] = {'bias': self.output_layer.weights}
        for col_ind, column in enumerate(self.output_layer.weights.T):
            for row_ind, val in enumerate(column):
                output_weights['node_{}'.format(col_ind)]['weight_{}'.format(row_ind)] = val
        model['output'].update(output_weights)

        model['hidden_act_func'] = self.hidden_layer.activation_function.label
        model['output_act_func'] = self.output_layer.activation_function.label

        return model

    # pylint: disable=too-many-arguments
    def setup_network(self, input_data_size: int, output_data_size: int, hidden_layer_number_of_nodes: int,
                      hidden_layer_bias: float = 0.0, output_layer_bias: float = 0.0,
                      hidden_layer_act_func: Function = Sigmoid(), output_layer_act_func: Function = Sigmoid(),
                      weights_range: Union[int, float] = 1) -> None:
        """Setup neural network

        :param input_data_size: input data size aka how many inputs are in input layer
        :param output_data_size: defines size of output vector
        :param hidden_layer_number_of_nodes: number of nodes in hidden layer
        :param hidden_layer_bias: value of hidden layer bias
        :param output_layer_bias: value of output layer bias
        :param hidden_layer_act_func: hidden layer activation function
        :param output_layer_act_func: output layer activation function
        :param weights_range: denotes range of weights
        """
        if self._was_initialized:
            return

        self.input_data_size = input_data_size

        hidden_layer_weights = weights_range * (np.random.random((input_data_size,
                                                                  hidden_layer_number_of_nodes)) - 0.5)
        output_layer_weights = weights_range * (np.random.random((hidden_layer_number_of_nodes,
                                                                  output_data_size)) - 0.5)
        self.hidden_layer = Layer(hidden_layer_number_of_nodes, hidden_layer_bias,
                                  hidden_layer_weights, hidden_layer_act_func)
        self.output_layer = Layer(output_data_size, output_layer_bias, output_layer_weights,
                                  output_layer_act_func)

        self._was_initialized = True

    def forward(self, inputs: Union[np.array, List[List[Real]]]) -> np.array:
        """Push data forward through network

        :param inputs: vector of input data
        """
        if not self._was_initialized:
            raise InitialisationError
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        self.hidden_layer = cast(Layer, self.hidden_layer)
        self.output_layer = cast(Layer, self.output_layer)

        self.hidden_layer.node_values = inputs.T.dot(self.hidden_layer.weights) + self.hidden_layer.bias

        self.hidden_layer.output = self.hidden_layer.activation_function.function(self.hidden_layer.node_values)
        assert self.hidden_layer.output is not None
        self.output_layer.node_values = self.hidden_layer.output.dot(self.output_layer.weights)
        self.output_layer.node_values += self.output_layer.bias

        self.output_layer.output = self.output_layer.activation_function.function(self.output_layer.node_values)
        return self.output

    def _backwards(self, inputs: np.array, targets: np.array) -> None:
        """Perform back-propagation. Back-propagation in steps:
              ____________________
        -----| in1       |        |
        -----| in2       |        |
        **** |       NET |  OUT   | ---- out value
        -----| bias      |        |
             |___________|________|


        NET value = sum(in1 * w1 + in2 * w2 + ... + inN * wN) + bias
        OUT value = activation_function(NET)

        1. Calculate difference between every output and targeted value Eo1 = ((target_o1-out_o1)^2)/2
        2. Calculate cumulative error values E_tot = sum(Eo1, Eo2...EoN)
        3. Calculate partial derivative of every output's error with respect to total error (e.g. dEo1/dE_tot)
                - this allows to understand how total error depends on changes of particular outputs error
        4. For every NET calculate derivative of output value with respect of Net value.
        5. For every input's weights (w) in NET calculate partial derivative with respect to NET value (e.g. dw1h/dNET)
                - this allows to understand how weight of particular input corresponds to NET value
                    - denoted as w1_h as it is weight of 1st input of NET from output from hidden layer

        Point 5 can be calculated using delta rule[1]:
            dEtot/dw1_h = delta_o1 * out_1h
            whereas
            delta_o1 = -(target_o1 - out_o1) * out_o1(1 - out_o1)

        6. update wh_1 value with formula: wh_1 = wh_1 - learning_rate * dEtot/dw1_h

        Hidden layer nodes:

        7. Calculate partial derivatives for all outputs of hidden layer with respect to total error:
            E_tot = sum(Eo1, Eo2...EoN)

            For output 1 of hidden layer this would be:
            dE_tot/dout_h1 = sum(dE_o1/dout_h1 + dE_o2/dout_h1 + ... + dE_oN/dout_h1)

        Follow the same rules as in point 1...6

        In a nutshell, using delta rule:
        dE_tot/dw_1 = sum(delta_o1 * w_h0 + ... + delta_o1 * w_hN) * out_h1(1 - out_h1) * input value
        dE_tot/dw_1 = delta_h1 * input value

        Recommended sources:
        [1] https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        [2] https://iamtrask.github.io/2015/07/27/python-network-part2/

        When talking about implementation - it is FAR BETTER (trust me) to operate on matrices rather than trying
        to implement as it was written above...

        :param inputs: vector of input values
        :param targets: vector of target values
        """
        assert isinstance(self.hidden_layer, Layer)
        assert isinstance(self.output_layer, Layer)
        assert self.hidden_layer.output is not None
        output_layer_errors = self.output_layer.output - targets.T
        output_layer_delta = output_layer_errors * self.output_layer.activation_function.prime(self.output_layer.output)

        hidden_layer_error = output_layer_delta.dot(self.output_layer.weights.T)
        hidden_layer_delta = hidden_layer_error * self.hidden_layer.activation_function.prime(self.hidden_layer.output)
        hidden_layer_updates = self.LEARNING_RATE * inputs.dot(hidden_layer_delta)
        self.hidden_layer.weights -= hidden_layer_updates

        output_layer_weights_updates = self.LEARNING_RATE * self.hidden_layer.output.T.dot(output_layer_delta)
        self.output_layer.weights -= output_layer_weights_updates

    def train(self, dataset: TrainingDataSet) -> None:
        """Perform training of network for given inputs and targets

        :param dataset: TrainingData object with inputs and targets
        """
        if not self._was_initialized:
            raise InitialisationError('Network was not initialised!')
        _count = 0
        self.forward(dataset.inputs)
        err_rate = self.cumulative_error_rate(dataset.targets) / len(dataset.inputs[0])
        while err_rate > self.TARGETED_ERROR_RATE and _count < self.MAX_EPOCHS:
            self.forward(dataset.inputs)
            self._backwards(dataset.inputs, dataset.targets)
            _count += 1
            err_rate = self.cumulative_error_rate(dataset.targets) / len(dataset.inputs[0])
            self._convergence_profile.append(err_rate)
            if _count % 100 == 0:
                logger.info('[%s] iter, cumulative error rate is %s', _count, err_rate)
        if err_rate > self.TARGETED_ERROR_RATE:
            logger.error('Could not converge, error rate = %s', err_rate)
        logger.info('Convergence achieved in %s iterations. Cumulative error rate is %s', _count, err_rate)
