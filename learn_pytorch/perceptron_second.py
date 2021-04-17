import torch
from torch import nn
from torch.nn import init
from utility.model_train import train
from utility.flatten_layer import FlattenLayer
from collections import OrderedDict
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.data_loader import data_loader
if '__main__' == __name__:

    num_inputs = 784
    num_hidden = 256
    num_outputs = 10
    num_epochs = 5
    batch_size = 256
    lr = 0.1

    mnist_train, mnist_test = get_fashion_MNST()
    train_iter = data_loader(mnist_train, 5, batch_size)
    test_iter = data_loader(mnist_test, 5, batch_size)

    net = nn.Sequential(
        OrderedDict(
            [
                ('FlattenLayer', FlattenLayer()),
                ('Hidden', nn.Linear(num_inputs, num_hidden)),
                ('Relu', nn.ReLU()),
                ('Output', nn.Linear(num_hidden, num_outputs))
            ]
        )
    )

    init.normal_(net.Hidden.weight, mean=0, std=0.01)
    init.normal_(net.Output.weight, mean=0, std=0.01)

    init.constant_(net.Hidden.bias, val=0)
    init.constant_(net.Output.bias, val=0)

    train(train_iter, test_iter, net, nn.CrossEntropyLoss(), num_epochs, lr)

