"""Module containing definitions of arithmetic functions used by perceptrons"""

import math
from abc import ABC, abstractmethod

import numpy as np

from NaiveNeurals.utils import ErrorAlgorithm


class Function(ABC):
    """Abstract function for defining functions"""

    @staticmethod
    @abstractmethod
    def function(arg: float) -> float:
        """Implementation of function

        :param arg: float
        :return: float
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def prime(cls, arg: float) -> float:
        """First derivative of implemented function

        :param arg: float
        :return: float
        """
        raise NotImplementedError()


class Sigmoid(Function):
    """Represents sigmoid function and its derivative"""

    @staticmethod
    def function(arg: float) -> float:
        """Calculate sigmoid(arg)

        :param arg: float input value
        :return: float sig(arg) value
        """
        return 1 / (1 + math.exp(-arg))

    @classmethod
    def prime(cls, arg: float) -> float:
        """Calculate value of sigmoid's prime derivative for given arg

        :param arg: float input value
        :return: float value
        """
        return cls.function(arg) * (1 - cls.function(arg))


def calculate_error(target: np.array, actual: np.array, func_type: ErrorAlgorithm = ErrorAlgorithm.SQR) -> float:
    """Calculates error for provided actual and targeted data.

    :param target: target data
    :param actual: actual training data
    :param func_type: denotes type of used function for error
    :return: calculated error
    """
    if func_type == ErrorAlgorithm.SQR:
        return 0.5 * pow(sum(actual - target), 2)
    raise NotImplementedError()
