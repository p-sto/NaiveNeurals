"""Contains definitions of resources needed for training"""
from typing import List, Union
import numpy as np


class DataSet:
    """Represents training set"""

    def __init__(self, training_inputs: Union[List[List], np.array], targets: Union[List[List], np.array]) -> None:
        """Initialize TrainingData object

        :param training_inputs: List of inputs
        :param targets: targeted outputs for provided inputs
        """
        self._training_inputs = np.array(training_inputs) if isinstance(training_inputs, list) else training_inputs
        self._targets = np.array(targets) if isinstance(targets, list) else targets

    @property
    def inputs(self) -> np.array:
        """Return input data as numpy array"""
        return self._training_inputs

    @property
    def targets(self) -> np.array:
        """Return targets data as numpy array"""
        return self._targets
