from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy
from learn_pytorch.generate_linear_data import genarate_linear_data
if '__main__' == __name__:
    w = torch.tensor([[1], [2]], dtype=torch.float32)
    b = torch.tensor([1], dtype=torch.float32)
    features, labels = genarate_linear_data(10000, w, b)
    data_set = TensorDataset(features, labels)
    batch_size = 100
    num_epochs = 10
    num_inputs = 2
    lr = 0.04
    loss = nn.MSELoss()
    net = nn.Sequential(OrderedDict([
                            ('linear', nn.Linear(num_inputs, 1, bias=True))
                        ]))
    optimizer = optim.SGD(net.parameters(), lr)
    for epoch in range(0, num_epochs):
        data_iter = DataLoader(data_set, batch_size, shuffle=True)
        for feature, label in data_iter:
            outputs = net(feature)
            l = loss(outputs, label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

    dense = net[0]
    print(dense.weight)
    print(dense.bias)