import torch
from torch import nn
from utility.no_operation import NoOperation
class Residual(nn.Module):
    def __init__(self, input_channel, output_channel, use_1x1conv = False, stride = 1):
        super(Residual, self).__init__()
        self.residual_net = self.gen_residual_net(input_channel, output_channel, stride)
        self.input_net = self.gen_input_net(input_channel, output_channel, use_1x1conv, stride)

    def gen_residual_net(self, input_channel, output_channel, stride = 1):
        block_list = []
        block_list.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=stride))
        block_list.append(nn.BatchNorm2d(output_channel))
        block_list.append(nn.ReLU())
        block_list.append(nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1))
        block_list.append(nn.BatchNorm2d(output_channel))
        return nn.Sequential(*block_list)

    def gen_input_net(self, input_channel, output_channel, use_1x1conv, stride = 1):
        if use_1x1conv:
            return nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)
        return NoOperation()

    def forward(self, x):
        return self.residual_net(x) + self.input_net(x)

if '__main__' == __name__:
    net = Residual(5, 5, True, stride=2)
    x = torch.ones([5, 5, 5, 5])
    print(x.shape)
    print(net(x).shape)