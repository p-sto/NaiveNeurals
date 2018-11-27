NaiveNeurals
============

Naive implementation of perceptron neural network. Under heavy development.


- [X] Implement 3 layer MLP network with SGD back-propagation algorithm
- [ ] Test coverage at least 80%
- [X] Allow model export/import to json
- [X] Prepare network learning examples and analysis for:
    - [X] Classification problem
    - [X] Regression problem
    - [X] Time series problem
    - [X] Data compression
- [ ] Use MLP network for MNIST dataset
- [ ] Implement various activation function
    - [X] Tanh
    - [X] Softmax
    - [X] Softplus
    - [ ] Gaussian
- [ ] Explore back-propagation algorithms:
    - [X] SGD with momentum
    - [ ] ADAM
    - [ ] Levenberg-Marquardt
- [ ] Add support for more than 1 hidden layer
- [ ] Create full documentation


Major inspiration for this work comes from book ``Machine Learning - An Algorithmic Perspective``.


Getting started
---------------

```bash
git clone https://github.com/stovorov/NaiveNeurals
cd NaiveNeurals
```


Prepare environment (using virtualenv)
--------------------------------------

Requires Python 3.6

```bash
source set_env.sh     # sets PYTHONPATH
make venv
source venv/bin/activate
make test
```

If you are using Ubuntu based system you must install tkinter

```bash
$ sudo apt-get install python3.6-tk
```

Usage
-----

### Simple training

```python
from NaiveNeurals.MLP.network import NeuralNetwork
from NaiveNeurals.data.dataset import DataSet

nn = NeuralNetwork()

nn.setup_network(input_data_size=2, output_data_size=1,
                 hidden_layer_number_of_nodes=5)

# every list in inputs represents one network input and data pushed into network
inputs =  [[0, 0, 1, 1], [1, 0, 1, 0]]
targets = [[1, 0, 0, 1]]

data_set = DataSet(inputs, targets)
nn.train(data_set)
```

If convergence is not achieved, ``ConvergenceError`` is raised.

### Network setup

There are 2 categories of network configuration parameters:

1. Network architecture (number of nodes, weights, activation functions etc.)

2. Learning configuration (algorithm: CG, CG_MOM, learning rate, target error rate etc.)

```python
from NaiveNeurals.MLP.network import NeuralNetwork, LearningConfiguration
from NaiveNeurals.MLP.activation_functions import Linear, Tanh

nn = NeuralNetwork()

# with LearningConfiguration one can set multiple parameters for solver algorithm 
learning_configuration = LearningConfiguration(learning_rate=0.01,
                                               target_error=0.003,
                                               solver='GD_MOM',
                                               max_epochs=1000,
                                               solver_params={'alpha': 0.95})

nn.setup_network(input_data_size=1,
                 output_data_size=1,
                 hidden_layer_number_of_nodes=25,
                 hidden_layer_bias=1,
                 output_layer_bias=-0.7,
                 hidden_layer_act_func=Tanh(),
                 output_layer_act_func=Linear())

nn.set_learning_params(learning_configuration)
```

### Batch learning with validation

In most cases it is recommended to split dataset into a few smaller subsets and validation set.
In batch mode network will switch training sets every 50 epochs and check error rate with validation data.

```python
from NaiveNeurals.MLP.network import NeuralNetwork
from NaiveNeurals.utils import ConvergenceError

train_data_set1 = ...
train_data_set2 = ...
train_data_set3 = ...
validation_set = ...

nn = NeuralNetwork()

try:
    nn.train_with_validation([train_data_set1, train_data_set2, train_data_set3], validation_set)
except ConvergenceError:
    pass
```

### Model export/import

Once model is trained it can be exported to dict and stored as json file using ``export_model`` method:

```python
from NaiveNeurals.MLP.network import NeuralNetwork
import json

nn = NeuralNetwork()

nn.setup_network(input_data_size=2, output_data_size=1,
                 hidden_layer_number_of_nodes=5)

# training procedure ...

with open('test.json', 'w+') as fil:
    fil.writelines(json.dumps(nn.export_model()))
```

Model can be imported using ``load_model`` method:

```python
from NaiveNeurals.MLP.network import NeuralNetwork

model_dict = ... # loaded model - Dict

nnn = NeuralNetwork()
nnn.load_model(model_dict)
```

Further Reading
---------------

If this project got your attention you can read about details below:

[Implementation details](docs/implementation_details.md)

[Classification problem example](docs/classification.md)

[Regression problem example](docs/regression.md)

[Time series problem example](docs/time_series.md)


References
----------

[Machine Learning - An Algorithmic Perspective (2nd edition)](https://www.amazon.com/Machine-Learning-Algorithmic-Perspective-Recognition/dp/1466583282/ref=dp_ob_title_bk)

[Stephen's Marsland homepage](https://seat.massey.ac.nz/personal/s.r.marsland/mlbook.html)

[Mat's Mazur Blog](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

[Python neural network pt.1](https://iamtrask.github.io/2015/07/12/basic-python-network/)

[Python neural network pt.2](https://iamtrask.github.io/2015/07/27/python-network-part2/)

[Activation function in neural networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

[Gradient Descent with Momentum](http://www.cs.bham.ac.uk/~jxb/NN/l8.pdf)

[SoftMax explained](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
