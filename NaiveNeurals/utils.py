"""Contains definitions of util functions and classes"""
from enum import Enum


class DataSeries(Enum):
    GAUSS_SINE = 'gauss_sine'


class ErrorAlgorithm(Enum):
    SQR = 'square_root'
