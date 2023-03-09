import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.synergy import read
del sys.path[-1]

n_markers = 27


def pad_sequence(data, pad_length, padval):
    length = data.shape[0]
    dof = data.shape[1]

    data_pad = np.empty((pad_length, dof)).astype(np.float32)
    data_pad[0:length, :] = data

    if padval == "last":
        data_pad[length:, :] = data[-1, :]
    elif padval == "zero":
        data_pad[length:, :] = 0.0

    return data_pad


def convert_writing(markers):
    import itertools

    idx = np.argmax(np.sum(markers, axis=0))

    for i, (l1, l2, l3) in enumerate(itertools.product([-1, 0, 1], repeat=3)):
        if i == idx:
            command = np.array([l1, l2, l3])
            command = np.tile(command, (markers.shape[0], 1)).astype(np.float32)

    return command
