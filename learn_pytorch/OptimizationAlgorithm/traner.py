import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt

def train(optimization_fun,
          states,
          hyperparameter,
          net,
          loss_fun,
          features,
          labels,
          batch_size = 10,
          num_epoch = 2):

    loss_list = [loss_fun(net(features).view(-1), labels).cpu().item()]
    data_iter = DataLoader(TensorDataset(features, labels), batch_size, shuffle=True)
    for epoch in range(num_epoch):
        start = time.time()
        for batch_i, (x, y) in enumerate(data_iter):
            y_hat = net(x)
            loss = loss_fun(y_hat, y)
            params = net.parameters()
            for param in params:
                if param.grad is not None:
                    param.grad.data.zero_()
                else:
                    break

            loss.backward()
            optimization_fun(net.parameters(), states, hyperparameter)
            if (batch_i + 1) * batch_size % 100 == 0:
                loss_list.append(loss_fun(net(features), labels))

    print('loss: %f, %f sec per epoch' % (loss_list[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epoch, len(loss_list)), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def train_concise(optimization_name,
          hyperparameter,
          net,
          loss_fun,
          features,
          labels,
          batch_size = 10,
          num_epoch = 2):

    optim_func = getattr(optim, optimization_name)
    optimizer = optim_func(net.parameters(), **hyperparameter)

    loss_list = [loss_fun(net(features), labels.view(-1, 1)).cpu().item() / 2]

    data_iter = DataLoader(TensorDataset(features, labels), batch_size, shuffle=True)
    for epoch in range(num_epoch):
        start = time.time()
        for batch_i, (x, y) in enumerate(data_iter):
            y_hat = net(x)
            loss = loss_fun(y_hat, y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                loss_list.append(loss_fun(net(features), labels.view(-1,1)) / 2)

    print('loss: %f, %f sec per epoch' % (loss_list[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epoch, len(loss_list)), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()