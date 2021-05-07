import torch
from torch import nn
from utility.trans_vector import one_hot
class LSTM(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(LSTM, self).__init__()
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.lstm_layer = nn.LSTM(inputs, hiddens)
        self.linear_layer = nn.Linear(hiddens, outputs)

    def predict(self,  prexs_index, predict_char_num):
        outputs = [prexs_index[0]]
        state = None
        for step in range(predict_char_num + len(prexs_index) - 1):
            x = torch.tensor(outputs[-1], dtype=torch.int64).view(-1,1)
            result_vec, state = self.forward(x, state)
            if step < len(prexs_index) - 1:
                outputs.append(prexs_index[step + 1])
            else:
                y_index = result_vec.argmax(dim = 1).view(-1).cpu().item()
                outputs.append(y_index)

        return outputs

    def forward(self, x, state):
        device = list(self.parameters())[0].device
        x_vector = torch.stack([one_hot(x[:, i], self.inputs, device) for i in range(x.shape[1])])
        lstm_output, new_sate = self.lstm_layer(x_vector, state)
        output_vec = self.linear_layer(lstm_output.view(-1, self.hiddens))
        return output_vec, new_sate

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