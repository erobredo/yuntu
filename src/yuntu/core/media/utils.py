import numpy as np


def pad_array(array, widths, mode, constant_values):
    if mode == 'constant':
        return np.pad(
            array,
            widths,
            mode=mode,
            constant_values=constant_values)

    return np.pad(array, widths, mode=mode)
