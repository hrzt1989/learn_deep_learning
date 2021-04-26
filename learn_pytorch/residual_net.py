import torch
from torch import nn
from utility.Residual import Residual
from utility.global_avg_pool import GlobalAvgPool
from utility.flatten_layer import FlattenLayer
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.net = self.gen_net()

    def gen_resnet_block(self, input_channel, output_channel, num_residuals, first_block = False):
        block_list = []
        for index in range(num_residuals):
            if index == 0 and not first_block:
                block_list.append(Residual(input_channel, output_channel, use_1x1conv=True, stride=2))
            else:
                block_list.append(Residual(output_channel, output_channel))
        return nn.Sequential(*block_list)

    def gen_net(self):
        net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        net.add_module("resnet_block1", self.gen_resnet_block(64, 64, 2, first_block=True))
        net.add_module("resnet_block2", self.gen_resnet_block(64, 128, 2))
        net.add_module("resnet_block3", self.gen_resnet_block(128, 256, 2))
        net.add_module("resnet_block4", self.gen_resnet_block(256, 512, 2))
        net.add_module("global_avg_pool", GlobalAvgPool())
        net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))
        return net


    def forward(self, x):
        return self.net(x)

if '__main__' == __name__:
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader
    from utility.model_train import train_device

    lr = 0.001
    num_epoch = 5
    batch_size = 64

    train_mnst, test_mnst = get_fashion_MNST(resize=112)
    train_iter= data_loader(train_mnst, 5, batch_size)
    test_iter = data_loader(test_mnst, 5, batch_size)

    net = ResidualNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_device(train_iter, test_iter, net, nn.CrossEntropyLoss(), lr, num_epoch, device, 'Adam')