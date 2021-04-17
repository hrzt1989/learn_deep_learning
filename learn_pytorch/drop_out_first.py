import torch
import numpy as np
from utility.load_fashion_MNIST import get_fashion_MNST
from utility.data_loader import data_loader
from utility.model_train import train_v0
from utility.relu import relu
from utility.drop_out import dropout
from utility.cross_entropy import cross_entropy
from utility.soft_max import soft_max

def net(x, hidden_ws, hidden_bs, output_w, output_b, is_training = False):
    drop_probs = [0.2, 0.5]
    input_feature = x.view(x.shape[0], -1)
    for one_w, one_b, one_prob in zip(hidden_ws, hidden_bs, drop_probs):

        input_feature = relu(torch.matmul(input=input_feature, other=one_w) + one_b)
        if is_training:
            input_feature = dropout(input_feature, one_prob)
    return soft_max(torch.matmul(input_feature, output_w) + output_b)

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

    hidden1_w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens1)), dtype=torch.float32, requires_grad=True)
    hidden2_w = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_hiddens2)), dtype=torch.float32, requires_grad=True)
    output_w = torch.tensor(np.random.normal(0, 0.01, (num_hiddens2, num_outputs)), dtype=torch.float32, requires_grad=True)

    hidden1_b = torch.zeros(num_hiddens1, dtype=torch.float32, requires_grad=True)
    hidden2_b = torch.zeros(num_hiddens2, dtype=torch.float32, requires_grad=True)
    output_b = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)

    train_v0(data_train_iter,
             data_test_iter,
             net,
             [hidden1_w, hidden2_w],
             [hidden1_b, hidden2_b],
             output_w,
             output_b,
             cross_entropy,
             batch_size,
             num_epochs,
             lr)