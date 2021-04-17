import torch
def dropout(x, prob):
    x_float = x.float()
    keep_prob = 1 - prob
    if keep_prob == 0:
        return torch.zeros_like(x_float)
    mask = (torch.rand(x_float.shape) < keep_prob).float()
    return mask * x_float / keep_prob

