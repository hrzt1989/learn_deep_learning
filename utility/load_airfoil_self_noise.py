import numpy as np
import torch
def load_airfoil_self_noise(filepath = '../data/airfoil_self_noise.dat'):
    data = np.genfromtxt(filepath, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    feature = torch.tensor(data[:1500, :-1], dtype=torch.float32)
    label = torch.tensor(data[:1500, -1], dtype=torch.float32)
    return feature, label


if '__main__' == __name__:
    features, labels = load_airfoil_self_noise()
    print('features', features.shape)
    print('labels', labels.shape)