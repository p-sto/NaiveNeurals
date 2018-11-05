"""Contains definitions of util functions and classes"""
from enum import Enum


class DataSeries(Enum):
    """Enum storing definitions of possible data series being generated"""
    GAUSS_SINE = 'gauss_sine'


class ErrorAlgorithm(Enum):
    """Enum storing definition of possible error algorithms"""
    SQR = 'square_root'


class InitialisationError(Exception):
    """Initialisation error definition"""
    pass
