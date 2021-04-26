from torch import nn
class Cov33Block(nn.Module):
    def __init__(self, input_channel, output_channle):
        super(Cov33Block, self).__init__()
        self.net = self.gen_net(input_channel, output_channle)

    def gen_net(self, input_channel, output_channel):
        return nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)