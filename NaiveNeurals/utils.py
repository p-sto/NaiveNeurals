"""Contains definitions of util functions and classes"""
from enum import Enum


class DataSeries(Enum):
    """Enum storing definitions of possible data series being generated"""
    SINE = 'sine'
    GAUSS_SINE = 'gauss_sine'


class ErrorAlgorithm(Enum):
    """Enum storing definition of possible error algorithms"""
    SQR = 'square_root'
    CE = 'cross_entropy'


class InitialisationError(Exception):
    """Initialisation error definition"""
    pass


class ConvergenceError(Exception):
    """Convergence error definition"""
    pass
