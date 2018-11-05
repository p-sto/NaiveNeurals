"""Module containing definitions of arithmetic functions used by perceptrons"""

from abc import ABC, abstractmethod

import numpy as np

from NaiveNeurals.utils import ErrorAlgorithm


class Function(ABC):
    """Abstract function for defining functions"""

    label = None

    @staticmethod
    @abstractmethod
    def function(arg: np.array) -> np.array:
        """Implementation of function

        :param arg: float
        :return: float
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def prime(cls, arg: np.array) -> np.array:
        """First derivative of implemented function

        :param arg: float
        :return: float
        """
        raise NotImplementedError()


class Sigmoid(Function):
    """Represents sigmoid function and its derivative"""

    label = 'sigmoid'

    @staticmethod
    def function(arg: np.array) -> np.array:
        """Calculate sigmoid(arg)

        :param arg: float input value
        :return: float sig(arg) value
        """
        return 1 / (1 + np.exp(-arg))

    @classmethod
    def prime(cls, arg: np.array) -> np.array:
        """Calculate value of sigmoid's prime derivative for given arg

        :param arg: float input value
        :return: float value
        """
        return cls.function(arg) * (1 - cls.function(arg))


def get_activation_function(label: str) -> Function:
    """Get activation function by label

    :param label: string denoting function
    :return: callable function
    """
    if label == 'sigmoid':
        return Sigmoid
    return Sigmoid


def calculate_error(target: np.array, actual: np.array,
                    func_type: ErrorAlgorithm = ErrorAlgorithm.SQR) -> np.array:
    """Calculates error for provided actual and targeted data.

    :param target: target data
    :param actual: actual training data
    :param func_type: denotes type of used function for error
    :return: calculated error
    """
    if func_type == ErrorAlgorithm.SQR:
        return np.sum(0.5 * np.power(actual - target, 2), axis=1)
    raise NotImplementedError()
