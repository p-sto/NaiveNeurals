"""Module contains generators for learning data"""

from typing import Optional, Tuple

import numpy as np
from NaiveNeurals.utils import DataSeries


def data_generator_for_regression(data_series: DataSeries, data_size: Optional[int] = 40) -> Tuple[np.array, np.array]:
    """Generate series for regression testing

    Source: Machine Learning - An Algorithmic Perspective

    :param data_series: type of data series
    :param data_size: generated data size
    :return: generated data
    """
    if data_series == DataSeries.GAUSS_SINE:
        x = np.ones((1, data_size)) * np.linspace(0, 1, data_size)
        y = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(data_size) * 0.15
        return x.T, y.T
    raise NotImplementedError
