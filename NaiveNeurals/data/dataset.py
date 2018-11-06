"""Contains definitions of resources needed for training"""
from typing import List
import numpy as np


class TrainingDataSet:
    """Represents training set"""

    def __init__(self, training_inputs: List[List[float]], targets: List[List[float]]) -> None:
        """Initialize TrainingData object

        :param training_inputs: List of inputs
        :param targets: targeted outputs for provided inputs
        """
        self._training_inputs = training_inputs
        self._targets = targets

    @property
    def inputs(self) -> np.array:
        """Return input data as numpy array"""
        return np.array(self._training_inputs)

    @property
    def targets(self) -> np.array:
        """Return targets data as numpy array"""
        return np.array(self._targets)
