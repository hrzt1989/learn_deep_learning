import numpy as np
import torch
from torch import nn
def default_active(input):
    return input

class LSTMUnit(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(LSTMUnit, self).__init__()
        self.params = nn.ParameterList()
        self.param_dict = {
            'f_gate' : self.init_param(inputs, hiddens),
            'i_gate' : self.init_param(inputs, hiddens),
            'c_delta' : self.init_param(inputs, hiddens),
            'o_gate' : self.init_param(inputs, hiddens),
            'output' : self.init_output_param(hiddens, outputs)
        }
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs

    def init_output_param(self, inputs, outputs):
        weight = self.init_weight(inputs, outputs)
        bias = self.init_bias(outputs)
        self.params.append(weight)
        self.params.append(bias)
        return (weight, None, bias)


    def init_weight(self, inputs, outputs):
        weight = torch.tensor(np.random.normal(0, 0.01, (inputs, outputs)), dtype=torch.float32, requires_grad=True)
        return nn.Parameter(weight)

    def init_bias(self, outputs):
        bias = torch.zeros(outputs, dtype=torch.float32, requires_grad=True)
        return nn.Parameter(bias)

    def init_param(self, inputs, hiddens):
        input_weight = self.init_weight(inputs, hiddens)
        hiddens_weight = self.init_weight(hiddens, hiddens)
        bias = self.init_bias(hiddens)
        self.params.append(input_weight)
        self.params.append(hiddens_weight)
        self.params.append(bias)
        return (input_weight, hiddens_weight, bias)

    def active(self, inputs, params, active_func = default_active):
        x, h = inputs
        weight_x, weight_h, bias = params
        result = torch.matmul(x, weight_x) + bias
        if weight_h is not None:
            result += torch.matmul(h, weight_h)
        return active_func(result)

    def forward(self, inputs):
        outputs = []
        steps, h, c = inputs
        for x in steps:
            f_gate_param = self.param_dict['f_gate']
            i_gate_param = self.param_dict['i_gate']
            c_delta_param = self.param_dict['c_delta']
            o_gate_param = self.param_dict['o_gate']
            output_param = self.param_dict['output']

            F = self.active((x, h), f_gate_param, torch.sigmoid)

            I = self.active((x, h), i_gate_param, torch.sigmoid)
            c_delta = self.active((x, h), c_delta_param, torch.tanh)

            O = self.active((x,h), o_gate_param, torch.sigmoid)
            c = F * c + I * c_delta
            h = torch.tanh(c) * O

            y = self.active((h, None), output_param)
            outputs.append(y)
        return torch.cat(outputs, dim=0), h, c