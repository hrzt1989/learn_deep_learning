import torch

def init_states(params):
    states = []
    for p in params:
        states.append(torch.zeros(p.shape, dtype=torch.float32))
    return states

def AdaGrad(params, states, hyperparams):
    err = 1e-6
    lr = hyperparams['lr']
    for p, s in zip(params, states):
        s.data += p.grad.data ** 2
        p.data -= (p.grad * lr) / torch.sqrt(s + err)

if '__main__' == __name__:
    from learn_pytorch.OptimizationAlgorithm.loss_func import sqrt_loss
    from learn_pytorch.OptimizationAlgorithm.traner import train
    from learn_pytorch.OptimizationAlgorithm.net import LinearReg
    from utility.load_airfoil_self_noise import load_airfoil_self_noise

    features, labels = load_airfoil_self_noise('../../data/airfoil_self_noise.dat')
    batch_size = 10
    lr = 0.1
    num_epoch = 2

    net = LinearReg(5, 1)
    params = net.parameters()

    states = init_states(params)

    train(AdaGrad,
          states,
          {'lr': lr},
          net,
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch
          )
