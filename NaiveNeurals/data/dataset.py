"""Contains definitions of resources needed for training"""
from typing import List, Union
import numpy as np


class DataSet:
    """Represents training set"""

    def __init__(self, training_inputs: List[List[Union[int, float]]], targets: List[List[Union[int, float]]],
                 auto_normalise: bool = False) -> None:
        """Initialize TrainingData object

        :param training_inputs: List of inputs
        :param targets: targeted outputs for provided inputs
        :param auto_normalise: set to true if target data must be automatically normalised
        """
        self._training_inputs = training_inputs
        self._targets = targets
        self._normalised = auto_normalise

    @property
    def inputs(self) -> np.array:
        """Return input data as numpy array"""
        return np.array(self._training_inputs)

    @property
    def targets(self) -> np.array:
        """Return targets data as numpy array"""
        return np.array(self._targets)
