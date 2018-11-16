"""Stores implementations of solvers"""
from enum import Enum, unique
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from NaiveNeurals.MLP.network import LearningConfiguration      # pylint: disable=cyclic-import, unused-import


@unique
class Solver(Enum):
    """Stores all supported solvers definitions."""
    GD = 'gradient_descent'
    GD_MOM = 'gradient_descent_momentum'


@unique
class SolversDefaults(Enum):
    """Stores configuration for solvers defaults values"""
    GD_MOM = {'alpha': 0.5}


class Cache:
    """Cache object for SGD Momentum"""
    stored_value: Dict[str, float] = {}


@lru_cache(32)
def get_solver_param(key: str, configuration: 'LearningConfiguration') -> Any:
    """Get parameter value from configuration class

    :param key: attribute name for which parameter is supposed to be obtained
    :param configuration: configuration class
    :return:
    """
    solver = configuration.solver
    if solver == Solver.GD_MOM:
        if key not in SolversDefaults[solver.name].value:
            raise AttributeError('No such parameter as `{}` for solver {}'.format(key, solver))
        return configuration.solver_setup.get(key, SolversDefaults[solver.name].value[key])
    return None


def calculate_weights(learning_rate: float, rhs: np.array, configuration: 'LearningConfiguration',
                      layer_label: str) -> np.array:
    """Calculate weights values based on provided solver algorithm

    :param learning_rate: learning rate parameter
    :param rhs: right-hands-side
    :param configuration: Network configuration object
    :param layer_label: denotes for which layer weights are calculated
    :return: weights update
    """
    if configuration.solver == Solver.GD:
        return learning_rate * rhs
    if configuration.solver == Solver.GD_MOM:
        if Cache.stored_value.get(layer_label) is None:
            Cache.stored_value[layer_label] = learning_rate * rhs
            return Cache.stored_value[layer_label]
        alpha = get_solver_param('alpha', configuration)
        to_update = learning_rate * rhs + alpha * Cache.stored_value[layer_label]
        Cache.stored_value[layer_label] = to_update
        return to_update
    raise NotImplementedError
