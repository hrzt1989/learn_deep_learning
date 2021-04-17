def soft_max(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim = 1, keepdim = True)
    return x_exp / partition