"""Contains definitions of resources needed for training"""
import random
from typing import List, Iterable


class TrainingSet:
    """Represents single training set consisting of targeted output for defined input"""

    def __init__(self, training_input: List[float], training_output: List[float]) -> None:
        """

        :param training_input:
        :param training_output:
        """
        self.training_input = training_input
        self.targeted_output = training_output


class TrainingData:
    """Represents training set"""

    def __init__(self, training_inputs: List[List[float]], targets: List[List[float]]) -> None:
        """Initialize TrainingData object

        :param training_inputs: List of inputs
        :param targets: targeted outputs for provided inputs
        """
        assert len(training_inputs) == len(targets)
        self.training_sets: List[TrainingSet] = []
        for input_data, target in zip(training_inputs, targets):
            self.training_sets.append(TrainingSet(input_data, target))

    @property
    def input_size(self):
        """Returns size of input data

        :return: integer value
        """
        return len(self.training_sets[0].training_input)

    @property
    def output_size(self):
        """Return size of output data

        :return: integer value
        """
        return len(self.training_sets[0].targeted_output)

    def get_randoms(self, max_count: int = 1) -> Iterable:
        """Return randomly selected training set for defined list

        :return: TrainingSet
        """
        count = 0
        while count < max_count:
            yield random.choice(self.training_sets)
            count += 1
