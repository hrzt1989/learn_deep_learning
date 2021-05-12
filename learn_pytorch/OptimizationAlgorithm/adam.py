import torch

def init_states(params):
    v = []
    s = []
    for p in params:
        v.append(torch.zeros(p.shape, dtype=torch.float32))
        s.append(torch.zeros(p.shape, dtype=torch.float32))
    return {'V': v, 'S' : s}


def Adam(params, states, hyperparams):
    lr = hyperparams['lr']
    err = 1e-6
    beta1 =0.9
    beta2 = 0.99
    V = states['V']
    S = states['S']
    t = states.get('t', 1)
    for p, v, s in zip(params, V, S):
        v.data = beta1 * v.data + (1 - beta1) * p.grad.data
        s.data = beta2 * s.data + (1 - beta2) * (p.grad.data ** 2)
        v_hat = v / (1 - beta1 ** t)
        s_hat = s / (1 - beta2 ** t)
        p.data -= (v_hat * lr) / (torch.sqrt(s_hat) + err)
    states['t'] = t + 1

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

    train(Adam,
          states,
          {'lr': lr},
          net,
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch
          )
