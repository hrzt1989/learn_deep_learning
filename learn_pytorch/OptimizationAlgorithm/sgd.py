def sgd(params, states, hyperparams):
    for param in params:
        param.data -= hyperparams['lr'] * param.grad.data

if '__main__' == __name__:
    from learn_pytorch.OptimizationAlgorithm.loss_func import sqrt_loss
    from learn_pytorch.OptimizationAlgorithm.traner import train
    from learn_pytorch.OptimizationAlgorithm.net import LinearReg
    from utility.load_airfoil_self_noise import load_airfoil_self_noise

    features, labels = load_airfoil_self_noise('../../data/airfoil_self_noise.dat')
    batch_size = 1
    lr = 0.005
    num_epoch = 2
    train(sgd,
          None,
          {'lr': lr},
          LinearReg(5, 1),
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch
          )
