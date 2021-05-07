import math
import time
import torch
from torch import optim
from utility.char_data_iter import char_data_iter_random, char_data_iter_consecutive
from utility.grade_clipping import gradient_clipping
def train_rnn(corpus_data,
              index_to_char_list,
              char_to_index_dict,
              is_random_iter,
              learn_rate,
              loss,
              num_step,
              batch_size,
              num_epoch,
              theta,
              prefixes,
              net,
              pred_len,
              device,
              optimizer_name,
              ):
    net = net.to(device)
    char_iter_func = None
    if is_random_iter:
        char_iter_func = char_data_iter_random
    else:
        char_iter_func = char_data_iter_consecutive

    ooptm_func = getattr(optim, optimizer_name)
    optimizer = ooptm_func(net.parameters(), learn_rate)
    corpu_index = [char_to_index_dict[one_char] for one_char in corpus_data]
    state = None

    pred_period = 50

    for epoch in range(num_epoch):
        index_iter = char_iter_func(corpu_index, num_step, batch_size)
        state = None

        l_sum, n, start = 0.0, 0, time.time()

        for x, y in index_iter:

            if state is not None:
                if is_random_iter:
                    state = None
                else:
                    if isinstance(state, tuple):
                        for one_state in state:
                            one_state.detach_()
                    else:
                        state.detach_()

            y_hat, state = net(x, state)
            yy = torch.transpose(y, 0, 1).contiguous().view(-1)
            yy = yy.to(device)

            optimizer.zero_grad()
            loss_sum = loss(y_hat, yy).sum()
            # print('loss_sum', loss_sum)
            loss_sum.backward()
            gradient_clipping(net.parameters(), theta, device)
            optimizer.step()
            l_sum += loss_sum.cpu().item() * yy.shape[0]
            n += yy.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                prexs_index = [char_to_index_dict[one_char] for one_char in prefix]
                result_index = net.predict(prexs_index, pred_len)
                chars = [index_to_char_list[char_index] for char_index in result_index]
                print(' -', ''.join(chars))




