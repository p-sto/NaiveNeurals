"""Module contains generators for learning data"""
from typing import Optional

import numpy as np
from NaiveNeurals.utils import DataSeries


def data_generator_for_regression(data_series: DataSeries, data_size: Optional[int] = 40):
    if data_series == DataSeries.GAUSS_SINE:
        x = np.ones((1, data_size)) * np.linspace(0, 1, data_size)
        y = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(data_size) * 0.15
        return x.T, y.T
    raise NotImplemented
