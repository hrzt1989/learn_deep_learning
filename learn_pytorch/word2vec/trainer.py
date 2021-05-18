import numpy as np
from torch import optim
import matplotlib.pyplot as plt
def train(data_iter,
          net,
          loss,
          num_epoch,
          learn_rate,
          batch_size,
          optimizer_name,
          device,
          show = False):
    net = net.to(device)
    optimfunc = getattr(optim, optimizer_name)
    optimizer = optimfunc(net.parameters(), learn_rate)

    loss_result_list = []
    beta = 0.9
    current_result = 0
    i = 0
    for epoch in range(num_epoch):
        for center, context, y, mask in data_iter:

            center_device = center.to(device)
            context_device = context.to(device)
            y_device = y.to(device)
            mask_device = mask.to(device)

            y_hat = net((center_device, context_device))
            loss_sum = loss(y_hat, y_device, mask_device)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            current_result = current_result * beta + loss_sum.view(-1).cpu().item() * (1 - beta)
            # print('loss_sum', loss_sum)

            i += 1
            if i * batch_size % 100 == 0:
                loss_result_list.append(current_result / (1 - beta ** i))
    if show:
        plt.plot(np.linspace(0, num_epoch, len(loss_result_list)), loss_result_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    return net