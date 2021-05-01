import torch
def one_hot(x, vocabulary_size):
    new_vector = torch.zeros((x.shape[0], vocabulary_size), dtype=torch.float32)
    new_vector.scatter_(1, x.view(-1, 1), 1)
    return new_vector