import torch
def init_states(params):
    deltas = []
    s = []
    for p in params:
        deltas.append(torch.zeros(p.shape, dtype=torch.float32))
        s.append(torch.zeros(p.shape, dtype=torch.float32))

    return {'deltas' : deltas, 'S' : s}

def adadelta(params, states, hyperparams):
    err = 1e-6
    rho = hyperparams['rho']
    deltas = states['deltas']
    S = states['S']
    for p, d, s in zip(params, deltas, S):
        s.data = rho * s.data + (1 - rho) * (p.grad ** 2)
        new_g = torch.sqrt((d.data + err) / (s.data + err)) * p.grad.data
        p.data -= new_g
        d.data = rho * d.data + (1 - rho) * (new_g * new_g)


# def init_adadelta_states():
#     s_w, s_b = torch.zeros((1,features.shape[1]), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
#     delta_w, delta_b = torch.zeros((1,features.shape[1]), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
#     return ((s_w, delta_w), (s_b, delta_b))

# def adadelta(params, states, hyperparams):
#     rho, eps = hyperparams['rho'], 1e-6
#     for p, (s, delta) in zip(params, states):
#         s[:] = rho * s + (1 - rho) * (p.grad.data**2)
#         g =  p.grad.data * torch.sqrt((delta + eps) / (s + eps))
#         p.data -= g
#         delta[:] = rho * delta + (1 - rho) * g * g

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
    # states = init_adadelta_states()
    train(adadelta,
          states,
          {'rho': 0.99},
          net,
          sqrt_loss,
          features,
          labels,
          batch_size,
          num_epoch
          )