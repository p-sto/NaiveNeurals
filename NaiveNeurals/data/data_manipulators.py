"""Contains mechanisms for data manipulation"""
from enum import Enum
from typing import List, Tuple, Union
import numpy as np


class FeatureScaling(Enum):
    """Aggregate definitions of all possible feature scaling algorithms."""
    RESCALE = 'rescale'
    MEAN_NORM = 'mean_norm'


def prepare_time_series_data(source_data: List[float], network_input_size: int,
                             data_interval: int) -> Tuple[List[List[float]], List[float]]:
    """Split raw time series data into inputs and targets for neural network

    Example:

    source_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    network_input_size = 3
    data_interval = 2

    results:
    inputs = [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
    targets = [[9], [10], [11], [12]]

    For inputs values 0, 3, 6 targeted output will be 9
    For inputs values 1, 4, 7 targeted output will be 10
    etc...

    :param source_data: time series source data
    :param network_input_size: neural network input size
    :param data_interval: interval between data used for prediction
    :return:
    """
    inputs: List[List[float]] = [[] for _ in range(network_input_size)]
    targets: List[float] = []

    last_element_ind = len(source_data) - network_input_size * (data_interval + 1) - 1
    for _ind in range(network_input_size):
        inputs[_ind] = source_data[_ind*(data_interval + 1):last_element_ind+_ind*(data_interval+1) + 1]
    for elmn_ind in range(last_element_ind + 1):
        targets.append(source_data[elmn_ind + network_input_size * (data_interval + 1)])
    return inputs, targets


def get_time_series_data_slice(source_data_series: List[List[float]], target_data_series: List[float],
                               start: int, stop: int) -> Tuple[List[List[float]], List[List[float]]]:
    """Get time series data slice for provided source and targets.

    :param source_data_series: source data from which slice has to be obtained
    :param target_data_series: target data from which slice has to be obtained
    :param start: slice start index
    :param stop: slice end index
    :return: sliced source data and target data
    """
    to_return_data, to_return_targets = [], []
    for elmn in source_data_series:
        to_return_data.append(elmn[start:stop])
    for ind, elmn in enumerate(target_data_series): # type: ignore
        if start <= ind < stop:
            to_return_targets.append(elmn)
    return to_return_data, [to_return_targets]  # type: ignore


def normalise_data(arr: Union[np.array, List], algorithm: FeatureScaling = FeatureScaling.RESCALE) -> np.array:
    """Normalise provided data

    :param arr: numpy array object
    :param algorithm: allows to set different algorithms for data normalisation
    :return: numpy array object normalised
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if algorithm == FeatureScaling.RESCALE:
        normalised = (arr - arr.min()) / (arr.max() - arr.min())
        return list(normalised)
    if algorithm == FeatureScaling.MEAN_NORM:
        normalised = (arr - arr.mean()) / (arr.max() - arr.min())
        return list(normalised)
    raise AttributeError('There is no definition for algorithms {}'.format(algorithm))
