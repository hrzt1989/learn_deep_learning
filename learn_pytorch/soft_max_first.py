import torch
import numpy as np
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.sgd import sgd
from utility.cross_entropy import cross_entropy
from utility.evaluate_accuracy import evaluate_accuracy
from utility.soft_max import soft_max
from utility.data_loader import data_loader

def net(x, w, b):
    x_soft_max = soft_max(torch.mm(x.view(x.shape[0], w.shape[0]), w) + b)
    return x_soft_max


def train(data_train_iter, data_test_iter, net, w, b,loss, learn_rate,batch_size, num_epochs):
    for epoch in range(num_epochs):
        for x, y in data_train_iter:
            y_hat = net(x, w, b)
            l = loss(y_hat, y)
            l_sum = l.sum()
            l_sum.backward()
            sgd([w, b], learn_rate, batch_size)
            w.grad.data.zero_()
            b.grad.data.zero_()

        acc = evaluate_accuracy(data_test_iter, net, w, b)
        print('epoch', epoch, 'acc', acc)



if '__main__' ==__name__:

    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    num_epochs = 5
    lr = 0.1

    mnist_train, mnist_test = get_fashion_MNST()

    data_train_iter = data_loader(mnist_train, 5, batch_size)
    data_test_iter = data_loader(mnist_test, 5, batch_size)

    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32,requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)

    train(data_train_iter, data_test_iter, net, w, b, cross_entropy, lr, batch_size, num_epochs)