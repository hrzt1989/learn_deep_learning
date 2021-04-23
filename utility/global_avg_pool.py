from torch import nn
from torch.nn import functional

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return functional.avg_pool2d(x, kernel_size = x.shape[2:])