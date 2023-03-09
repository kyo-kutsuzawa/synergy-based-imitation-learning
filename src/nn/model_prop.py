import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataio
import mylstm


class Mymodel_train(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_layers, synergies_shape, offset=-6.0):
        super().__init__()

        self.lstms = mylstm.MyLSTM(in_dim, n_units, n_layers)
        self.fc1 = nn.Linear(n_units, out_dim)
        self.dec = nn.Conv1d(synergies_shape[0], synergies_shape[2], synergies_shape[1], bias=False, padding=synergies_shape[1] - 1)

        self.n_units = n_units
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.offset = offset

    def forward(self, x):
        if self.h is not None:
            h_out, (self.h, self.c) = self.lstms(x, self.h, self.c)
        else:
            h_out, (self.h, self.c) = self.lstms(x)
        N = h_out.shape[0]
        L = h_out.shape[1]
        h_out = torch.reshape(h_out, (N*L, -1))

        y = self.fc1(h_out)
        y = torch.reshape(y, (N, L, -1))

        y = y + self.offset
        y = F.softplus(y)

        v = torch.transpose(y, 1, 2)
        v = self.dec(v)[:, :, :y.shape[1]]
        v = v[:, :2] - v[:, 2:]
        v = torch.cumsum(v, dim=2)
        v = torch.transpose(v, 1, 2)

        return y, v

    def reset_state(self):
        self.h = None
        self.c = None

    def register_dec(self, synergies):
        synergies = np.transpose(synergies, (2, 0, 1))
        synergies = synergies[:, :, ::-1].copy()
        synergies = torch.tensor(synergies.astype(np.float32))
        self.dec.weight = nn.Parameter(synergies)
        self.dec.weight.requires_grad = False


class Mymodel(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_layers, synergies=None, offset=-6.0):
        super().__init__()

        self.lstms = mylstm.MyLSTMCells(in_dim, n_units, n_layers)
        self.fc1 = nn.Linear(n_units, out_dim)
        self.dec = nn.Conv1d(synergies.shape[0], synergies.shape[2], synergies.shape[1], bias=False, padding=synergies.shape[1] - 1)

        self.n_units = n_units
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.offset = offset

        self.synergies = synergies

    def forward(self, x):
        h = self.lstms(x)
        y = self.fc1(h)

        y = y + self.offset

        self.activity = F.softplus(y)

        # Decode synergy activities to positions
        self.activities.append(self.activity.numpy().reshape((1, -1)).copy())
        self.activities.pop(0)
        if self.pos is None:
            self.pos = x.detach().numpy().flatten()[0:2]
        self.pos = decode_step(self.pos, self.activities, self.synergies)

        return self.pos.copy()

    def reset_state(self, batchsize=1):
        self.lstms.reset_state(batchsize)

        self.activities = [np.zeros((1, self.out_dim), dtype=np.float32) for _ in range(self.synergies.shape[1])]
        self.pos = None

    def get_internal_state(self):
        return self.lstms.get_internal_state()


def convert_model(model_dict):
    return mylstm.convert_model(model_dict)


def load_dataset(dirname, use_torch=False, padding=False):
    import os, glob

    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    datalist = [np.loadtxt(filename, delimiter=",", dtype=np.float32) for filename in filelist]

    length_max = max([data.shape[0] for data in datalist])

    dataset = []
    for data in datalist:
        # Create input data
        position = dataio.read(data, "position", dataio.n_markers)
        markers = dataio.read(data, "markers", dataio.n_markers)
        markers = dataio.convert_writing(markers)
        data_in = np.concatenate([position, markers], axis=1)
        data_in = data_in[:-1]

        # Create output data
        data_out = dataio.read(data, "position", dataio.n_markers)
        data_out = data_out[1:]

        # Pad the data if specified
        if padding:
            data_in  = dataio.pad_sequence(data_in, length_max, padval="last")
            data_out = dataio.pad_sequence(data_out, length_max, padval="last")

        # Convert to torch.Tensor if specified
        if use_torch:
            import torch
            data_in  = torch.tensor(data_in)
            data_out = torch.tensor(data_out)

        dataset.append((data_in, data_out))

    return dataset


def loss_func(y_predicted, position_truth, alpha=1.0):
    activity_predicted, position_predicted = y_predicted
    position_predicted += position_truth[:, 0:1, :]

    l1 = F.mse_loss(position_predicted, position_truth)
    l2 = torch.mean(activity_predicted)
    loss = l1 + l2 * alpha

    return loss


def decode_step(pos, activities, synergies):
    T = synergies.shape[1]
    dev = np.zeros((1, 4))
    for tau in range(T):
        dev += np.dot(activities[tau], synergies[:, T - tau - 1, :])
    pos += dev[0, :2] - dev[0, 2:]

    return pos
