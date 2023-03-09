from typing import OrderedDict
import torch
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()

        self.lstms = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.lstm_h = torch.nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.lstm_c = torch.nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, h=None, c=None):
        batchsize = x.shape[0]
        if h is None:
            h = torch.broadcast_to(self.lstm_h, [self.num_layers, batchsize, self.hidden_size]).contiguous()
        if c is None:
            c = torch.broadcast_to(self.lstm_c, [self.num_layers, batchsize, self.hidden_size]).contiguous()

        return self.lstms(x, (h, c))

    def reset_state(self):
        pass


class MyLSTMCells(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTMCells, self).__init__()

        self.lstms = []
        self.lstms.append(nn.LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.lstms.append(nn.LSTMCell(hidden_size, hidden_size))
        self.lstms = nn.ModuleList(self.lstms)

        self.lstm_h = torch.nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.lstm_c = torch.nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            self.state[i] = self.lstms[i](h, self.state[i])
            h = self.state[i][0]

        return h

    def reset_state(self, batchsize=1):
        h = torch.broadcast_to(self.lstm_h, [self.num_layers, batchsize, self.hidden_size]).contiguous()
        c = torch.broadcast_to(self.lstm_c, [self.num_layers, batchsize, self.hidden_size]).contiguous()

        self.state = [(h[i], c[i]) for i in range(self.num_layers)]

    def get_internal_state(self):
        c = [s[1] for s in self.state]
        c = torch.stack(c)

        return c


def convert_model(model_dict):
    import re

    template = "(.+?\.?)?lstms\.(.*?)_l(\d)"
    new_dict = OrderedDict()

    for k in model_dict.keys():
        matched = re.match(template, k)
        if matched:
            parent_name = matched.group(1)
            param_name = matched.group(2)
            if parent_name is None:
                parent_name = ""

            l = int(matched.group(3))
            new_key = "{}lstms.{}.{}".format(parent_name, l, param_name)
            new_dict[new_key] = model_dict[k].clone()
        else:
            new_dict[k] = model_dict[k]

    return new_dict
