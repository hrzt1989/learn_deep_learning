import torch
from torch import nn
class BatchNorm1d(nn.Module):
    def __init__(self, feature_num, eps = 1e-5, momentum = 0.9):
        super(BatchNorm1d, self).__init__()
        self.eps = torch.tensor(eps)
        self.momentum = torch.tensor(momentum)

        self.moving_mean = torch.zeros([1, feature_num])
        self.moving_var = torch.zeros([1, feature_num])

        self.gamma = nn.Parameter(torch.ones([1, feature_num]))
        self.beta = nn.Parameter(torch.ones([1, feature_num]))

    def forward(self, x):
        if x.device != self.moving_var.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
            self.eps = self.eps.to(x.device)
            self.momentum = self.momentum.to(x.device)

        if not self.training:
            return self.gamma * ((x - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)) + self.beta

        temp_mean = x.mean(dim = 0, keepdim = True)
        temp_var = ((x - temp_mean) ** 2).mean(dim = 0, keepdim = True)

        self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * temp_mean
        self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * temp_var
        return self.gamma * ((x - temp_mean) / torch.sqrt(temp_var + self.eps)) + self.beta



class BatchNorm2d(nn.Module):
    def __init__(self, feature_num, eps = 1e-5, momentum = 0.9):
        super(BatchNorm2d, self).__init__()
        self.eps = torch.tensor(eps)
        self.momentum = torch.tensor(momentum)

        self.moving_mean = torch.zeros([1, feature_num, 1, 1])
        self.moving_var = torch.zeros([1, feature_num, 1, 1])

        self.gamma = nn.Parameter(torch.ones([1, feature_num, 1, 1]))
        self.beta = nn.Parameter(torch.ones([1, feature_num, 1, 1]))

    def forward(self, x):
        if x.device != self.moving_var.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
            self.eps = self.eps.to(x.device)
            self.momentum = self.momentum.to(x.device)

        if not self.training:
            return self.gamma * ((x - self.moving_mean) / torch.sqrt(self.moving_var + self.eps))+ self.beta

        temp_mean = x.mean(dim = 0, keepdim = True).\
                      mean(dim = 2 ,keepdim = True).\
                      mean(dim = 3, keepdim = True)

        temp_var = ((x - temp_mean) ** 2).\
            mean(dim = 0, keepdim = True).\
            mean(dim = 2, keepdim = True).\
            mean(dim = 3, keepdim = True)
        self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * temp_mean
        self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * temp_var
        return self.gamma * ((x - temp_mean) / torch.sqrt(temp_var + self.eps)) + self.beta

if '__main__' == __name__:
    x = torch.ones([2, 5,5, 5], dtype=torch.float)
    x = x.to('cuda')
    print('x.shape', x.shape)
    net = BatchNorm2d(5).to('cuda')
    result = net(x)
    print('result.shape', result.shape)
    print('net', net(x))