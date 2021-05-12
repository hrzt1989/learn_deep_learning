if '__main__' == __name__:
    from torch import nn
    from learn_pytorch.OptimizationAlgorithm.traner import train_concise
    from learn_pytorch.OptimizationAlgorithm.net import LinearReg
    from utility.load_airfoil_self_noise import load_airfoil_self_noise

    lr = 0.01
    batch_size = 10
    num_epoch = 2

    features, labels = load_airfoil_self_noise('../../data/airfoil_self_noise.dat')
    net = LinearReg(5, 1)
    train_concise('RMSprop',
                  {'lr': lr, 'alpha': 0.99},
                  net,
                  nn.MSELoss(),
                  features,
                  labels,
                  batch_size,
                  num_epoch
                  )