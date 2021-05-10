import torch

def init_states(num_params):
    return [torch.zeros(1, dtype=torch.float32) for i in range(num_params)]

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        momentum = hyperparams['momentum']
        lr = hyperparams['lr']
        v.data = momentum * v.data + lr * p.grad.data
        p.data -= v.data

if '__main__' == __name__:
    from learn_pytorch.OptimizationAlgorithm.loss_func import sqrt_loss
    from learn_pytorch.OptimizationAlgorithm.traner import train
    from learn_pytorch.OptimizationAlgorithm.net import LinearReg
    from utility.load_airfoil_self_noise import load_airfoil_self_noise

    features, labels = load_airfoil_self_noise('../../data/airfoil_self_noise.dat')
    batch_size = 10
    lr = 0.02
    num_epoch = 2

    net = LinearReg(5, 1)
    params = net.parameters()

    states = init_states(len(list(params)))


    train(sgd_momentum,
          states,
          {'lr': lr, 'momentum' : 0.5},
          net,
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch
          )