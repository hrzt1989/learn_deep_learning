import torch
from torch import nn
from utility.gru import GRU
from utility.trans_vector import one_hot
class GatedRecurrentUnit(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(GatedRecurrentUnit, self).__init__()
        self.net = GRU(inputs, hiddens, outputs)

    def predict(self, prexs_index, predict_char_num):
        param = list(self.net.parameters())
        device = param[0].device
        outputs = [prexs_index[0]]
        state = torch.zeros(1, self.net.hiddens, dtype=torch.float32, device=device)

        for step in range(predict_char_num + len(prexs_index) - 1):
            x = torch.tensor(outputs[-1], dtype=torch.int64).view(1,1)
            x_vector = [one_hot(x[:, i], self.net.inputs, device) for i in range(x.shape[1])]
            inputs = (x_vector, state)
            result, state = self.net(inputs)
            if step < len(prexs_index) - 1:
                outputs.append(prexs_index[step + 1])
            else:
                result_idex = result.argmax(dim=1).view(-1).cpu().item()
                outputs.append(result_idex)
        return outputs

    def forward(self, x, state):
        param = list(self.net.parameters())
        device = param[0].device
        if state is None:
            state = torch.zeros(x.shape[0], self.net.hiddens, dtype=torch.float32, device=device)
        else:
            state = state.to(device)
        x_vectors = [one_hot(x[:, i], self.net.inputs, device) for i in range(x.shape[1])]
        result, new_state = self.net((x_vectors, state))
        return result, new_state

if '__main__' == __name__:
    from torch import nn
    from utility.jaychou.load_jaychou_lyrics import get_jaychou_lyrics
    from utility.train_rnn import train_rnn
    from utility.gen_char_index import gen_char_index

    epoch = 250
    lr = 0.01
    theta = 0.01
    hiddens = 256
    batch_size = 128
    num_step = 35
    num_epochs = 250
    pred_len = 50

    corpus_char_list = get_jaychou_lyrics()
    index_to_char_list, char_to_index_dict = gen_char_index(corpus_char_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = GatedRecurrentUnit(len(index_to_char_list), hiddens, len(index_to_char_list))
    train_rnn(corpus_char_list,
              index_to_char_list,
              char_to_index_dict,
              True,
              lr,
              nn.CrossEntropyLoss(),
              num_step,
              batch_size,
              num_epochs,
              theta,
              ['分开', '不分开'],
              net,
              pred_len,
              device,
              'Adam',
              )