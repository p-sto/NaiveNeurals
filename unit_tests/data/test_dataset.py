"""Contains tests for dataset.py"""

from NaiveNeurals.data.dataset import TrainingDataSet


def test_dataset():
    inputs = [[0, 0, 1, 1], [0, 1, 0, 1]]
    targets = [[1, 0, 0, 1]]
    data_set = TrainingDataSet(inputs, targets)
    assert data_set.inputs[0][0] == 0
    assert data_set.targets[0][0] == 1
