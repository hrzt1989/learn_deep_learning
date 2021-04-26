from torch import nn
class TransitionBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TransitionBlock, self).__init__()
        self.net = self.gen_net(input_channel, output_channel)

    def gen_net(self, input_channel, output_channel):
        return nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.net(x)