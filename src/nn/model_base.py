import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataio
import mylstm


class Mymodel_train(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_layers):
        super().__init__()

        self.lstms = mylstm.MyLSTM(in_dim, n_units, n_layers)
        self.fc1 = nn.Linear(n_units, out_dim)

        self.n_units = n_units
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim

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

        y = torch.cumsum(y, dim=1)

        return y

    def reset_state(self):
        self.h = None
        self.c = None

    def unchain(self):
        self.h = self.h.detach()
        self.c = self.c.detach()


class Mymodel(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_layers, synergies=None):
        super().__init__()

        self.lstms = mylstm.MyLSTMCells(in_dim, n_units, n_layers)
        self.fc1 = nn.Linear(n_units, out_dim)

        self.n_units = n_units
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.synergies = synergies

    def forward(self, x):
        h = self.lstms(x)
        y = self.fc1(h)

        self.activity = y

        if self.pos is None:
            self.pos = x.detach().numpy().flatten()[0:2]

        self.pos += y.detach().numpy().flatten()

        return self.pos.copy()

    def reset_state(self, batchsize=1):
        self.lstms.reset_state(batchsize)

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


def loss_func(y_predicted, y_truth):
    y_predicted += y_truth[:, 0:1, :]
    loss = F.mse_loss(y_predicted, y_truth)

    return loss
