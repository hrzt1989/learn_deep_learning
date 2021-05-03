import torch
from torch import nn
from utility.trans_vector import one_hot
class RNN(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hiddens,
                 class_num):

        super(RNN, self).__init__()
        self.class_num = class_num
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.rnn_layer = nn.RNN(input_size=class_num, hidden_size=num_hiddens)
        self.linear_layer = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x, state):
        paras = list(self.parameters())
        x_features = torch.stack([one_hot(x[:, i], self.class_num) for i in range(x.shape[1])])
        x_features = x_features.to(paras[0].device)
        if state is not None:
            state = state.to(paras[0].device)
        rnn_y, new_state = self.rnn_layer(x_features, state)
        result = self.linear_layer(rnn_y.view(-1, self.num_hiddens))
        return result, new_state

    def predict(self, prexs_index, predict_char_num):
        paras = list(self.parameters())
        device = paras[0].device
        outputs = [prexs_index[0]]
        state = None
        for i in range(predict_char_num + len(prexs_index) - 1):
            # char_index = torch.tensor(outputs[-1], dtype=torch.int64).view(1,1)
            # char_vec = one_hot(char_index, self.class_num)
            # char_vec = char_vec.to(device)
            result, state = self.forward(torch.tensor(outputs[-1], dtype=torch.int64).view(1,1), state)
            result_index = result.argmax(dim = 1).view(-1)
            if i < len(prexs_index) -1:
                outputs.append(prexs_index[i + 1])
            else:
                outputs.append(result_index.cpu().item())
        return outputs

if '__main__' == __name__:
    from torch import nn
    from utility.jaychou.load_jaychou_lyrics import get_jaychou_lyrics
    from utility.gen_char_index import gen_char_index
    from utility.train_rnn import train_rnn
    corpus_data = get_jaychou_lyrics()
    index_to_char_list, char_to_index_dict = gen_char_index(corpus_data)

    num_inputs = len(index_to_char_list)
    num_outputs = len(index_to_char_list)
    class_num = len(index_to_char_list)
    num_hiddens = 256

    net = RNN(num_inputs, num_outputs, num_hiddens, class_num)

    lr = 0.001
    theta = 0.01
    batch_size = 64
    num_step = 35
    num_epochs = 250
    pred_len = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_rnn(corpus_data,
              index_to_char_list,
              char_to_index_dict,
              False,
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
              'Adam')