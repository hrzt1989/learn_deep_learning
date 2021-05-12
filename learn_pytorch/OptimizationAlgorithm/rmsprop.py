import torch
from torch import optim
def init_states(params):
    return [torch.zeros(p.shape, dtype=torch.float32) for p in params]

def copy_params(params):
    new_params = []
    old_param = []
    for p in params:
        new_params.append(p.clone())
        old_param.append(p)
    return old_param, new_params

def RMSProp(params, states, hyperparams):
    lr = hyperparams['lr']
    alpha = hyperparams['alpha']
    err = 1e-6
    for p, s in zip(params, states):
        s.data += s.data * alpha + (1 - alpha) * (p.grad.data) ** 2
        p.data -= (p.grad.data * lr) / torch.sqrt(s + err)

if '__main__' == __name__:
    from learn_pytorch.OptimizationAlgorithm.loss_func import sqrt_loss
    from learn_pytorch.OptimizationAlgorithm.traner import train
    from learn_pytorch.OptimizationAlgorithm.net import LinearReg
    from utility.load_airfoil_self_noise import load_airfoil_self_noise

    features, labels = load_airfoil_self_noise('../../data/airfoil_self_noise.dat')
    batch_size = 10
    lr = 0.01
    num_epoch = 2

    net = LinearReg(5, 1)
    params = net.parameters()

    states = init_states(params)

    train(RMSProp,
          states,
          {'lr': lr, 'alpha' : 0.5},
          net,
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch,
          'RMSprop'
          )