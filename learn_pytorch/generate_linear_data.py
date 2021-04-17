import torch
import numpy
def genarate_linear_data(data_size, w, b):
    data = torch.randn(data_size, w.size()[0])
    labels = torch.mm(data, w) + b
    labels += torch.tensor(numpy.random.normal(0, 0.01, labels.size()), dtype=torch.float32)
    return data, labels

if '__main__' == __name__:
    w = torch.tensor([[1], [1]], dtype=torch.float32)
    b = 1
    data, labels = genarate_linear_data(5, w, b)
    print('data', data)
    print('labels', labels)