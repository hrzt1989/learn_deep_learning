import math
import time
import torch
import numpy as np
from utility.char_data_iter import char_data_iter_random, char_data_iter_consecutive
from utility.gen_char_index import gen_index_list, gen_char_index
from utility.grade_clipping import gradient_clipping
from utility.sgd import sgd
from utility.trans_vector import one_hot
class RNN():
    def __init__(self, inputs, outputs, hiddens, device):
        input_weight, state_weight, hidden_bias, output_weight, output_bias = self.get_init_param(inputs, outputs, hiddens, device)
        self.input_weight = input_weight
        self.state_weight = state_weight
        self.hidden_bias = hidden_bias
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.device = device

    def init_one_weight(self, inputs, outputs, device):
        return torch.tensor(np.random.normal(0, 0.01, (inputs, outputs)),
                            dtype=torch.float32,
                            device=device,
                            requires_grad=True)

    def init_state(self, inputs, outputs):
        return torch.zeros((inputs, outputs), dtype=torch.float32)

    def get_params(self):
        return (self.input_weight, self.state_weight, self.hidden_bias, self.output_weight, self.output_bias)

    def get_init_param(self, inputs, outputs, hiddens, device):
        input_weight = self.init_one_weight(inputs, hiddens, device)
        state_weight = self.init_one_weight(hiddens, hiddens, device)
        output_weight = self.init_one_weight(hiddens, outputs, device)
        hidden_bias = torch.zeros(hiddens, dtype=torch.float32, device=device, requires_grad=True)
        output_bias = torch.zeros(outputs, dtype=torch.float32, device=device, requires_grad=True)
        return (input_weight, state_weight, hidden_bias, output_weight, output_bias)

    def forward(self, inputs):
        x, state = inputs
        state_device = state.to(self.device)
        results = []
        for one_x in x:
            one_x_device = one_x.to(self.device)
            hidden_result = torch.tanh((torch.matmul(input = one_x_device, other= self.input_weight) +\
                                        torch.matmul(input = state_device, other=self.state_weight)) +\
                                       self.hidden_bias)
            state_device = hidden_result
            result = torch.matmul(hidden_result, self.output_weight) + self.output_bias
            results.append(result)
        return results, state_device

    def predict(self,
                prefix,
                num_chars,
                index_to_char_list,
                char_to_index_dict):
        state = self.init_state(1, len(self.hidden_bias))
        outputs = []
        outputs.append(prefix[0])
        i = 0
        for t in range(num_chars + len(prefix) -1):
            input_chars_index = torch.tensor([char_to_index_dict[outputs[-1]]], dtype=torch.int64).view(-1, 1)
            x = self.to_one_hot(input_chars_index, 1, len(index_to_char_list))
            inputs = (x, state)
            char_vecs, state = self.forward(inputs)

            if i < len(prefix) - 1:
                outputs.append(prefix[i + 1])
            else:
                for char_vec in char_vecs:
                    char_index = char_vec.argmax(dim = 1)
                    char_index_cpu = char_index.cpu()
                    temp_chars = [index_to_char_list[char_index_cpu[i].item()] for i in range(char_index_cpu.shape[0])]
                    outputs.extend(temp_chars)
            i += 1
        return ''.join(outputs)

    def to_one_hot(self, x, num_step, class_num):
        return [one_hot(x[:, i], class_num) for i in range(num_step)]

    def train(self,
              corpus_data,
              is_random_iter,
              learn_rate,
              loss,
              num_step,
              batch_size,
              num_epoch,
              prefixes,
              pred_len):

        index_to_char_list, char_to_index_dict = gen_char_index(corpus_data)
        vocab_size = len(index_to_char_list)
        corpus_chars_index = gen_index_list(corpus_data, char_to_index_dict)

        iter_func = char_data_iter_consecutive
        if is_random_iter:
            iter_func = char_data_iter_random

        state = self.init_state(batch_size, len(self.hidden_bias))


        pred_period = 50

        for epoch in range(num_epoch):
            n = 0
            l_sum = 0
            start = time.time()
            data_iter = iter_func(corpus_chars_index, num_step, batch_size)
            for x, y in data_iter:
                if is_random_iter:
                    state = self.init_state(batch_size, len(self.hidden_bias))
                else:
                    state.detach_()
                x_vector = self.to_one_hot(x, num_step, vocab_size)
                y_hat, state = self.forward((x_vector, state))
                y_hat = torch.cat(y_hat, dim=0)

                params = self.get_params()

                for one_param in params:
                    if one_param.grad is None:
                        break
                    one_param.grad.data.zero_()

                yy = torch.transpose(y, 0, 1).contiguous().view(-1)

                yy_device = yy.to(self.device)
                loss_sum = loss(y_hat, yy_device).sum()
                loss_sum.backward()

                gradient_clipping(params=params, theta=0.01, device=self.device)
                sgd(params, learn_rate, 1)
                n += yy.shape[0]
                l_sum += loss_sum.cpu().item() * yy.shape[0]
            if (epoch + 1) % pred_period == 0:
                print('epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, math.exp(l_sum / n), time.time() - start))
                for prefix in prefixes:
                    print(' -', self.predict(prefix, pred_len, index_to_char_list, char_to_index_dict))

if '__main__' == __name__:
    from torch import nn
    from utility.jaychou.load_jaychou_lyrics import get_jaychou_lyrics
    lr = 100
    corpus_char_list = get_jaychou_lyrics()
    index_to_char_list, char_to_index_dict = gen_char_index(corpus_char_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn = RNN(len(index_to_char_list), len(index_to_char_list), 256, device)
    rnn.train(corpus_char_list,
              False,
              lr,
              nn.CrossEntropyLoss(),
              35,
              32,
              1000,
              ['分开', '不分开'],
              50)



