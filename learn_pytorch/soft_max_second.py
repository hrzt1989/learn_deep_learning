import torch
from torch import nn
from torch.nn import init
from torch import optim
from collections import OrderedDict
from utility.evaluate_accuracy import evaluate_accuracy_v2
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.data_loader import data_loader

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self,x):
        return x.view(x.shape[0], -1)

def train(train_data_iter, test_data_iter, net, loss, batch_size, learn_rate, num_epochs):
    optimizer = optim.SGD(net.parameters(), learn_rate)
    for epoch in range(num_epochs):
        for x, y in train_data_iter:
            y_hat = net(x)
            loss_sum = loss(y_hat, y).sum()
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        acc = evaluate_accuracy_v2(test_data_iter, net)
        print('epoch', epoch, 'acc', acc)


if '__main__' == __name__:
    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    num_epochs = 5
    lr = 0.1

    mnist_train, mnist_test = get_fashion_MNST()
    train_iter = data_loader(mnist_train, 5, batch_size)
    test_iter = data_loader(mnist_test, 5, batch_size)
    net = nn.Sequential(
        OrderedDict(
            [('FlattenLayer', FlattenLayer()),
             ('linear', nn.Linear(num_inputs, num_outputs))]
        )
    )
    init.normal_(net.linear.weight, mean = 0, std = 0.01)
    init.constant_(net.linear.bias, val=0)
    train(train_iter, test_iter, net, nn.CrossEntropyLoss(), batch_size, lr, num_epochs)