NaiveNeurals
============

Naive implementation of perceptron neural network. Under heavy development.


Goals

- [X] Implement 3 layer MLP network with GD back-propagation algorithm
- [ ] Test coverage at least 80%
- [X] Allow model export/import to json
- [ ] Add support for installing with Anaconda
- [ ] Prepare network learning examples and analysis:
    - [ ] Classification problem
    - [ ] Regression problem
- [ ] Use MLP network for MNIST dataset
- [ ] Implement various activation function
    - [x] Tanh
    - [ ] Softmax
    - [ ] ReLU
    - [ ] Gaussian
- [ ] Explore back-propagation algorithms:
    - [ ] Stochastic gradient descent
    - [ ] SGD with momentum
    - [ ] ADAM
    - [ ] Levenberg-Marquardt



Getting started
---------------

```bash
$ git clone https://github.com/stovorov/NaiveNeurals
$ cd NaiveNeurals
```


Prepare environment (using virtualenv)
--------------------------------------

```bash
$ source set_env.sh     # sets PYTHONPATH
$ make venv
$ source venv/bin/activate
$ make test

If you are using Ubuntu based system you must install tkinter

For Python < 3.6
$ sudo apt-get install python3-tk

For Python 3.6
$ sudo apt-get install python3.6-tk
```

Usage
-----


```python
from NaiveNeurals.MLP.network import NeuralNetwork
from NaiveNeurals.data.dataset import TrainingDataSet

input_data_size = 2
output_data_size = 1
hidden_layer_number_of_nodes = 5

nn = NeuralNetwork()
nn.setup_network(input_data_size=input_data_size, output_data_size=output_data_size,
                 hidden_layer_number_of_nodes=hidden_layer_number_of_nodes)

inputs =  [[0, 0, 1, 1], [1, 0, 1, 0]]
targets = [[1, 0, 0, 1]]

data_set = TrainingDataSet(inputs, targets)
nn.train(data_set, display_err=True)

nn.forward([[1], [0]])
```

References
----------

[Machine Learning - An Algorithmic Perspective (2nd edition)][1]

[Mat Mazur's blog][2]

[Basic python neural network][3]

[Basic python neural network pt2][4]

[Activation Functions: Neural Networks][5]

[1]: https://www.amazon.com/Machine-Learning-Algorithmic-Perspective-Recognition/dp/1466583282?undefined
[2]: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
[3]: https://iamtrask.github.io/2015/07/12/basic-python-network/
[4]: https://iamtrask.github.io/2015/07/27/python-network-part2/
[5]: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

