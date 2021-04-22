from torch import nn
from utility.flatten_layer import FlattenLayer
class VGG(nn.Module):
    def __init__(self, vgg_blocks, image_shape, fc_hidden_units):
        super(VGG, self).__init__()
        self.net = self.gen_vgg_net(vgg_blocks, image_shape, fc_hidden_units)

    def forward(self, image):
        return self.net(image)

    def gen_vgg_block(self, num, input_channel, output_channel):
        vblk = []
        for i in range(num):
            if i == 0:
                conv = nn.Conv2d(input_channel, output_channel, 3, padding=1)
            else:
                conv = nn.Conv2d(output_channel, output_channel, 3, padding=1)
            vblk.append(conv)
            vblk.append(nn.ReLU())
        maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        vblk.append(maxp)
        return nn.Sequential(*vblk)

    def gen_vgg_net(self, vgg_blocks, image_shape, fc_hidden_units):
        image_high = image_shape[0]
        image_width = image_shape[1]
        net_sequential = nn.Sequential()
        vgg_block_num = len(vgg_blocks)
        last_output_channel = 0
        for index, (num, input_channel, output_channel) in enumerate(vgg_blocks):
            vgg_block = self.gen_vgg_block(num, input_channel, output_channel)
            net_sequential.add_module('vgg_block_' + str(index), vgg_block)
            last_output_channel = output_channel
        new_image_high = image_high // (2 ** vgg_block_num)
        new_image_width = image_width // (2 ** vgg_block_num)
        feature_num = new_image_high * new_image_width * last_output_channel
        net_sequential.add_module('fc',
            nn.Sequential(
                FlattenLayer(),

                nn.Linear(feature_num, fc_hidden_units),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(fc_hidden_units, fc_hidden_units),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(fc_hidden_units, 10)
            )
        )
        return net_sequential


if '__main__' == __name__:
    import torch
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader
    from utility.model_train import train_device

    batch_size = 64
    num_epoch = 5
    lr = 0.001
    ratio = 8

    train_MNIST, test_MNIST = get_fashion_MNST(resize=224)
    train_data_iter = data_loader(train_MNIST, 5, batch_size)
    test_data_iter = data_loader(test_MNIST, 5, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_blocks = [(1, 1, 64 // ratio),
                  (1, 64 // ratio, 128 // ratio),
                  (2, 128 // ratio, 256 // ratio),
                  (2, 256 // ratio, 512 // ratio),
                  (2, 512 // ratio, 512 // ratio)]
    image_shape = (224, 224)
    fc_hidden_units = 4096 // ratio

    net = VGG(vgg_blocks, image_shape, fc_hidden_units)
    print(net)
    train_device(train_data_iter,
                 test_data_iter,
                 net,
                 nn.CrossEntropyLoss(),
                 lr,
                 num_epoch,
                 device,
                 'Adam')