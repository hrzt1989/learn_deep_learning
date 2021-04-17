import torch
def relu(x):
    return torch.max(input = x, other=torch.tensor(0.0))