import torch
from torch import nn
from torch.nn import init
from utility.model_train import train
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.data_loader import data_loader
from utility.flatten_layer import FlattenLayer
from collections import OrderedDict

if '__main__' == __name__:

    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    num_hiddens1 = 256
    num_hiddens2 = 256
    num_epochs = 10
    lr = 0.5

    mnist_train, mnist_test = get_fashion_MNST()
    data_train_iter = data_loader(mnist_train, 5, batch_size)
    data_test_iter = data_loader(mnist_test, 5, batch_size)

    net = nn.Sequential(
        OrderedDict([
            ('FlattenLayer', FlattenLayer()),

            ('hidden1', nn.Linear(num_inputs, num_hiddens1)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout()),

            ('hidden2', nn.Linear(num_hiddens1, num_hiddens2)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout()),

            ('output', nn.Linear(num_hiddens2, num_outputs)),
        ])
    )

    init.normal_(net.hidden1.weight, mean=0, std=0.01)
    init.normal_(net.hidden2.weight, mean=0, std=0.01)
    init.normal_(net.output.weight, mean=0, std=0.01)

    init.constant_(net.hidden1.bias, val=0)
    init.constant_(net.hidden2.bias, val=0)
    init.constant_(net.output.bias, val=0)


    train(data_train_iter,
          data_test_iter,
          net,
          nn.CrossEntropyLoss(),
          num_epochs,
          lr)