import torch
from torch import nn
from utility.trans_vector import one_hot
class GatedRecurrentUnit(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(GatedRecurrentUnit, self).__init__()
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.gru_layer = nn.GRU(inputs, hiddens)
        self.linear_layer = nn.Linear(hiddens, outputs)

    def forward(self, x, state):
        params = list(self.parameters())
        device = params[0].device
        if state is not None:
            state = state.to(device)
        x_vector = torch.stack([one_hot(x[:, i], self.inputs, device) for i in range(x.shape[1])])
        gru_result, new_state = self.gru_layer(x_vector, state)
        result = self.linear_layer(gru_result.view(-1, self.hiddens))
        return result, new_state

    def predict(self, prexs_index, predict_char_num):
        outputs = []
        state = None
        outputs.append(prexs_index[0])
        for step in range(predict_char_num + len(prexs_index) - 1):
            x = torch.tensor(outputs[-1], dtype=torch.int64).view(1,1)
            result_vector, state = self.forward(x, state)
            if step < len(prexs_index) - 1:
                outputs.append(prexs_index[step + 1])
            else:
                result_index = result_vector.argmax(dim = 1).view(-1).cpu().item()
                outputs.append(result_index)
        return outputs

if '__main__' == __name__:
    from utility.jaychou.load_jaychou_lyrics import get_jaychou_lyrics
    from utility.gen_char_index import gen_char_index
    from utility.train_rnn import train_rnn

    lr = 0.01
    theta = 0.01
    num_epoch = 250
    batch_size = 32
    num_step = 35
    hiddens = 256
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
              num_epoch,
              theta,
              ['分开', '不分开'],
              net,
              pred_len,
              device,
              'Adam')