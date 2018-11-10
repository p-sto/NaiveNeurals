"""Module contains generators for learning data"""

from typing import Tuple

import numpy as np
from NaiveNeurals.utils import DataSeries


def data_generator_for_regression(data_series: DataSeries, data_size: int = 60) -> Tuple[np.array, np.array]:
    """Generate series for regression testing

    Source: Machine Learning - An Algorithmic Perspective

    :param data_series: type of data series
    :param data_size: generated data size
    :return: generated data
    """
    x = [val / data_size for val in range(data_size)]
    if data_series == DataSeries.SINE:
        y = [np.sin(2 * np.pi * val) for val in x]
        return x, y
    elif data_series == DataSeries.GAUSS_SINE:
        y = [np.sin(2 * np.pi * val) + np.cos(4 * np.pi * val) for val in x]
        return x, y
    raise NotImplementedError
