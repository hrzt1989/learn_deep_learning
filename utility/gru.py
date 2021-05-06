import numpy as np
import torch
from torch import nn
class GRU(nn.Module):

    def __init__(self, inputs, hiddens, outputs):
        super(GRU, self).__init__()
        self.inputs = inputs
        self.hiddens = hiddens

        weight_xr, weiht_hr, bias_r = self.init_param(inputs, hiddens)
        weight_xz, weight_hz, bias_z = self.init_param(inputs, hiddens)
        weight_xh, weight_hh, bias_h = self.init_param(inputs, hiddens)

        self.weight_xr = weight_xr
        self.weight_hr = weiht_hr
        self.bias_r = bias_r

        self.weight_xz = weight_xz
        self.weight_hz = weight_hz
        self.bias_z = bias_z

        self.weight_xh = weight_xh
        self.weight_hh = weight_hh
        self.bias_h = bias_h

        self.weight_result = self.init_weight(hiddens, outputs)
        self.bias_result = self.init_bias(outputs)

    def init_weight(self, inputs, outputs):
        weight = torch.tensor(np.random.normal(0, 0.01, (inputs, outputs)), dtype=torch.float32, requires_grad=True)
        return nn.Parameter(weight)

    def init_bias(self, outputs):
        bias = torch.zeros(outputs, dtype=torch.float32, requires_grad=True)
        return nn.Parameter(bias)

    def init_param(self, inputs, outputs):
        weight_x = self.init_weight(inputs, outputs)
        weight_h = self.init_weight(outputs, outputs)
        bias = self.init_bias(outputs)
        return weight_x, weight_h, bias

    def active_matmul(self, x, h, weight_x, weight_h, bias, active_func):
        result =active_func(torch.matmul(x, weight_x) + torch.matmul(h, weight_h) + bias)
        return result

    def compute_one_step(self, inputs):
        x, h = inputs
        result_r = self.active_matmul(x, h, self.weight_xr, self.weight_hr, self.bias_r, torch.sigmoid)
        result_z = self.active_matmul(x, h, self.weight_xz, self.weight_hz, self.bias_z, torch.sigmoid)
        h_inputs = result_r * h
        result = self.active_matmul(x, h_inputs, self.weight_xh, self.weight_hh, self.bias_h, torch.tanh)
        result_h = h * result_z + (1 - result_z) * result
        return torch.matmul(result, self.weight_result) + self.bias_result, result_h

    def forward(self, inputs):
        x_steps, h = inputs
        result_list = []
        for x in x_steps:
            result, h = self.compute_one_step((x, h))
            result_list.append(result)
        return torch.cat(result_list, dim=0), h