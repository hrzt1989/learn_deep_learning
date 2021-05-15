import torch
from torch.utils.data import Dataset
def trans_batch(data_batch):
    centers = []
    label = []
    mask = []
    all_context = []
    max_len = max(len(c) + len(n) for _, c, n in data_batch)
    for center, context, negative in data_batch:
        centers.append([center])
        all_context.append(context + negative + [0] * (max_len - len(context) - len(negative)))
        label.append([1] * len(context) + [0] * (max_len - len(context)))
        mask.append([1] * (len(context) + len(negative)) + [0] * (max_len - len(context) - len(negative)))

    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(all_context),
            torch.tensor(label),
            torch.tensor(mask))



class WordDataSet(Dataset):
    def __init__(self, centers, context, negative):
        self.centers = centers
        self.context = context
        self.negative = negative

    def __getitem__(self, index):
        return (self.centers[index], self.context[index], self.negative[index])

    def __len__(self):
        return len(self.centers)