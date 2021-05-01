import torch
def gradient_clipping(params, theta, device):

    norm = torch.tensor(0, device=device, dtype=torch.float32)
    for one_param in params:
        norm += (one_param.grad ** 2).sum()

    if theta < torch.sqrt(norm) :
        ratio = theta / torch.sqrt(norm)
        for one_param in params:
            one_param.grad.data *= ratio