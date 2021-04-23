from torch import nn
from utility.global_avg_pool import GlobalAvgPool
from utility.flatten_layer import FlattenLayer
class NIN(nn.Module):
    def __init__(self, nin_blocks, num_output):
        super(NIN, self).__init__()
        self.net = self.gen_nin_net(nin_blocks, num_output)

    def forward(self, image):
        return self.net(image)

    def gen_nin_block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size=1),
            nn.ReLU()
        )
    def gen_nin_net(self, nin_blocks, num_output):
        last_output_channel = 0
        net = nn.Sequential()
        for index, (input_channel, output_channel, kernnel_size, stride, padding) in enumerate(nin_blocks):
            one_block = self.gen_nin_block(input_channel, output_channel, kernnel_size, stride, padding)
            net.add_module('nin_' + str(index), one_block)
            net.add_module('max_pool', nn.MaxPool2d(kernel_size=3, stride=2))
            last_output_channel = output_channel

        net.add_module('dropout', nn.Dropout(0.5))
        block = self.gen_nin_block(last_output_channel, num_output, kernel_size=3, stride=1, padding=1)
        net.add_module('final_nin_block', block)
        net.add_module('global_avg_pool', GlobalAvgPool())
        net.add_module('flatten_layer', FlattenLayer())
        return net

if '__main__' == __name__:
    import torch
    from torch import nn
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader
    from utility.model_train import train_device


    batch_size = 64
    num_epoch = 5
    lr = 0.002

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_MNST, test_MNST = get_fashion_MNST(resize = 224)

    train_data_iter = data_loader(train_MNST, 5, batch_size)
    test_data_iter = data_loader(test_MNST, 5, batch_size)

    nin_blocks = [(1, 96, 11, 4, 0), (96, 256, 5, 1, 2), (256, 384, 3, 1, 1)]
    net = NIN(nin_blocks, 10)

    print(net)
    train_device(train_data_iter, test_data_iter, net, nn.CrossEntropyLoss(), lr, num_epoch, device, 'Adam')