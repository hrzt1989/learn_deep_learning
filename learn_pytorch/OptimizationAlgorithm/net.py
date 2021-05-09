from torch import nn
class LinearReg(nn.Module):
    def __init__(self, inputs, outputs):
        super(LinearReg, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.net = nn.Linear(inputs, outputs)

    def forward(self, x):
        return self.net(x)