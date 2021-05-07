import torch
from torch import nn
from utility.lstm_unit import LSTMUnit
from utility.trans_vector import one_hot
class LSTM(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(LSTM, self).__init__()
        self.net = LSTMUnit(inputs, hiddens, outputs)

    def init_state(self, inputs, outputs, device):
        return torch.zeros((inputs, outputs), dtype=torch.float32, device=device)

    def predict(self, prexs_index, predict_char_num):
        outputs = [prexs_index[0]]
        state = None
        for step in range(predict_char_num + len(prexs_index) - 1):
            x = torch.tensor(outputs[-1], dtype=torch.int64).view(1, 1)
            result, state = self.forward(x, state)
            if step < len(prexs_index) - 1:
                outputs.append(prexs_index[step + 1])
            else:
                y = result.argmax(dim = 1).view(-1).cpu().item()
                outputs.append(y)
        return outputs

    def forward(self, x, state):
        params = list(self.net.parameters())
        device = params[0].device
        if state == None:
            h = self.init_state(x.shape[0], self.net.hiddens, device)
            c = self.init_state(x.shape[0], self.net.hiddens, device)
        else:
            h, c = state
            h = h.to(device)
            c = c.to(device)
        x_vector = [one_hot(x[:, i], self.net.outputs, device) for i in range(x.shape[1])]
        y, h, c = self.net((x_vector, h, c))
        return y, (h, c)

if '__main__' == __name__:
    from utility.jaychou.load_jaychou_lyrics import get_jaychou_lyrics
    from utility.gen_char_index import gen_char_index
    from utility.train_rnn import train_rnn

    num_epoch = 250
    lr = 0.01
    theta = 0.01
    hiddens = 256
    batch_size = 128
    num_step = 35
    num_epochs = 250
    pred_len = 50

    corpus_data = get_jaychou_lyrics()
    index_to_char_list, char_to_index_dict = gen_char_index(corpus_data)
    net = LSTM(len(index_to_char_list), hiddens, len(index_to_char_list))
    params = list(net.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_rnn(
        corpus_data,
        index_to_char_list,
        char_to_index_dict,
        True,
        lr,
        nn.CrossEntropyLoss(),
        num_step,
        batch_size,
        num_epoch,
        theta,
        ['分开', '不分开'],
        net,
        pred_len,
        device,
        'Adam',
    )