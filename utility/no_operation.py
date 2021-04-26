from torch import nn
class NoOperation(nn.Module):
    def __init__(self):
        super(NoOperation, self).__init__()

    def forward(self, x):
        return x