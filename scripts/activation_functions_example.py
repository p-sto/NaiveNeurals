"""Contains example scripts presenting various activation functions characteristics."""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import inspect
from typing import Dict, Type

import NaiveNeurals.MLP.activation_functions as functions_module


def plot_characteristics():
    """Plots various activation functions characteristics"""
    functions: Dict[str, Type[functions_module.ActivationFunction]] = {}
    for _, obj in inspect.getmembers(functions_module):
        if obj == functions_module.ActivationFunction:
            continue
        if inspect.isclass(obj) and issubclass(obj, functions_module.ActivationFunction):
            functions[obj.label] = obj

    plot_marks = ['m-', 'k-', 'r--', 'b-', 'c-']
    colors = ['magenta', 'black', 'red', 'blue', 'cyan']
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    legend_handlers = []
    for fn_label, fn in functions.items():
        if fn_label == 'lin':
            x_vals = np.linspace(-1, 1, 21)
        elif fn_label == 'softplus':
            x_vals = np.linspace(-2, 2, 21)
        else:
            x_vals = np.linspace(-4, 4, 101)
        plot_label = mpatches.Patch(color=colors.pop(), label=fn_label)
        legend_handlers.append(plot_label)
        ax1.plot(x_vals, fn.function(x_vals), plot_marks.pop())

    plt.legend(handles=legend_handlers)
    plt.savefig(os.path.abspath(os.path.dirname(__file__)) + '/../docs/graphs/activation_functions.png')
    plt.show()


if __name__ == '__main__':
    plot_characteristics()
