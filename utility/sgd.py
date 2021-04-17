import torch
def sgd(params, learn_rate, batch_size):
    for param in params:
        param.data -= param.grad * learn_rate / batch_size
        if True in torch.isnan(param):
            print('sgd', 'nan in param', param.shape)