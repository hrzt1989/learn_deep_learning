import torch
import numpy as np
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.data_loader import data_loader
from utility.relu import relu
from utility.sgd import sgd
from utility.cross_entropy import cross_entropy
from utility.evaluate_accuracy import evaluate_accuracy_v1
from utility.soft_max import soft_max

def net(x, hidden_ws, hidden_bs, output_w, output_b):
    x_layer = x.view(x.shape[0], -1)
    for w, b in zip(hidden_ws, hidden_bs):
        x_layer = relu(torch.matmul(x_layer, w) + b)
    return soft_max(torch.matmul(x_layer, output_w) + output_b)

def train(train_data_iter,
          test_data_iter,
          net,
          loss,
          hidden_ws,
          hidden_bs,
          output_w,
          output_b,
          batch_size,
          learn_rate,
          num_epochs):

    params = []
    for one_w in hidden_ws:
        params.append(one_w)

    for one_b in hidden_bs:
        params.append(one_b)

    params.append(output_w)
    params.append(output_b)

    for epoch in range(num_epochs):
        for x, y in train_data_iter:

            y_hat = net(x, hidden_ws, hidden_bs, output_w, output_b)
            loss_result = loss(y_hat, y)

            loss_sum = loss_result.sum()
            loss_sum.backward()
            #print('loss_sum', loss_sum )
            sgd(params, learn_rate, batch_size)

            for one_param in params:
                one_param.grad.data.zero_()
        train_acc = evaluate_accuracy_v1(train_data_iter, net, hidden_ws, hidden_bs, output_w, output_b)
        acc = evaluate_accuracy_v1(test_data_iter, net, hidden_ws, hidden_bs, output_w, output_b)
        print('epoch', epoch, 'tain_acc', train_acc,'test_acc', acc)

if  '__main__' == __name__:
    batch_size = 256

    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256

    num_epochs = 10
    lr = 0.1

    mnist_train, mnist_test = get_fashion_MNST()
    data_train_iter = data_loader(mnist_train, 5, batch_size)
    data_test_iter = data_loader(mnist_test, 5, batch_size)

    hidden_w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float32, requires_grad=True)
    output_w = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float32, requires_grad=True)
    hidden_b = torch.zeros(num_hiddens, requires_grad=True)
    output_b = torch.zeros(num_outputs, requires_grad=True)

    hidden_w.requires_grad_()

    train(data_train_iter,
          data_test_iter,
          net,
          cross_entropy,
          [hidden_w],
          [hidden_b],
          output_w,
          output_b,
          batch_size,
          lr,
          num_epochs)
