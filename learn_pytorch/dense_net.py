
if '__main__' == __name__:
    import torch
    from torch import nn
    from utility.dense_block import DenseBlock
    from utility.transition_block import TransitionBlock
    from utility.global_avg_pool import GlobalAvgPool
    from utility.flatten_layer import FlattenLayer
    from utility.model_train import train_device
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader

    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    number_channel = 64
    growth_rate = 32
    num_cov_in_block = [4, 4, 4, 4]
    temp_input_channel = number_channel
    for index, cov_num in enumerate(num_cov_in_block):
        one_dense_block = DenseBlock(cov_num, temp_input_channel, growth_rate)
        temp_input_channel = one_dense_block.output_channel
        net.add_module('dense_block_' + str(index), one_dense_block)
        if index != len(num_cov_in_block) - 1:
            one_transition_block = TransitionBlock(temp_input_channel, temp_input_channel // 2)
            net.add_module('transition_block_' + str(index), one_transition_block)
            temp_input_channel = temp_input_channel // 2
    net.add_module('BN', nn.BatchNorm2d(temp_input_channel))
    net.add_module('Relu', nn.ReLU())
    net.add_module('GlobalAvgPool', GlobalAvgPool())
    net.add_module('FlattenLayer', FlattenLayer())
    net.add_module('linear', nn.Linear(temp_input_channel, 10)) \

    print('net', net)
    batch_size = 128
    lr = 0.001
    num_epochs = 5

    train_mnist, test_mnist = get_fashion_MNST(resize=112)
    train_iter = data_loader(train_mnist, 5, batch_size)
    test_iter = data_loader(test_mnist, 5, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_device(train_iter, test_iter, net, nn.CrossEntropyLoss(), lr, num_epochs, device, 'Adam')