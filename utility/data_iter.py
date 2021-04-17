import torch
from torch.utils.data import DataLoader
import random
import sys
def data_iter(batch_size, feature, label):
    num_samples = len(feature)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_samples)])
        yield feature.index_select(0, j), label.index_select(0, j)

def data_iter_upgrade(batch_size, data_set):

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    data_iter = DataLoader(data_set, batch_size, shuffle=True, num_workers=num_workers)
    return data_iter

if '__main__' == __name__:
    from utility.load_fashion_MNIST import get_fashion_MNST

    train_set, test_set = get_fashion_MNST()
    train_iter = data_iter_upgrade(256, train_set)
    test_iter = data_iter_upgrade(256, test_set)
    for feature, label in test_iter:
        print(label)