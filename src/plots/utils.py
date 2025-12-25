import numpy as np


def moving_average(data, windowsize=None):
    assert type(windowsize) == int, "The Type of 'windowsize' must be int."
    window = np.ones(int(windowsize)) / float(windowsize)
    smooth_data = []
    for interval in data:
        re = np.convolve(interval, window, "valid")
        smooth_data.append(re)
    return smooth_data
