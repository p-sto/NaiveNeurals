"""Module containing tests for functions,py"""
import numpy as np
import pytest

from NaiveNeurals.MLP.functions import get_activation_function, Sigmoid, calculate_error


def test_get_activation_function():
    assert get_activation_function('sigmoid') == Sigmoid


def test_calculate_error():
    targets = np.array([[1, 0, 0, 1]])
    inputs = np.array([[.99, .1, .1, .99]])
    assert calculate_error(targets, inputs)[0] == pytest.approx(0.0101, 0.0001)
