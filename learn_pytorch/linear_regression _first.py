import torch
import numpy
from utility.data_iter import data_iter
from learn_pytorch.generate_linear_data import genarate_linear_data

def sqrt_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

def linear_regression_model(features, w, b):
    return torch.mm(features, w) + b

def sgd(params, learn_rate, batch_size):
    for param in params:
        param.data -= param.grad * learn_rate / batch_size

if '__main__' == __name__:

    # test_yhat = torch.tensor([[3], [3]])
    # test_y = torch.tensor([[1], [1]])
    # test_loss = sqrt_loss(test_yhat, test_y).sum()
    lr = 0.4
    batch_size = 100
    num_epochs = 100
    original_w = torch.tensor([[1], [2]], dtype=torch.float32)
    original_b = 5

    w = torch.tensor(numpy.random.normal(0, 0.01, original_w.size()), dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    data, labels = genarate_linear_data(10000, original_w, original_b)
    for epoch in range(num_epochs):
        for batch_data, batch_lables in data_iter(batch_size, data, labels):
            loss = sqrt_loss(linear_regression_model(batch_data, w, b), batch_lables).sum()
            loss.backward()
            sgd([w, b], lr, batch_size)

            w.grad.data.zero_()
            b.grad.data.zero_()
            # last_loss = sqrt_loss(linear_regression_model(batch_data, w, b), batch_lables).sum()
        final_loss = sqrt_loss(linear_regression_model(data, w, b), labels).sum()
        print('loss', final_loss)
        print('w', w)
        print('b', b)
    print('w', w)
    print('b', b)