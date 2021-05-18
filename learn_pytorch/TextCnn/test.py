import torch
import pickle
def compare(word_vocab):
    stoi = word_vocab.stoi
    itos = word_vocab.itos
    for s in stoi:
        if s not in itos:
            print('error', s, word_vocab.stoi[s])
            pass
class EmbeddingVocab(object):
    def __init__(self, vectors, itos, stoi):
        self.vectors = vectors
        self.itos = itos
        self.stoi = stoi

def load_embedding_struct(embedding_path, index_path):
    embedding = torch.load(embedding_path)
    index_file = open(index_path, 'rb')
    index_to_word_list, word_to_index_dic    = pickle.load(index_file)
    return EmbeddingVocab(embedding, index_to_word_list, word_to_index_dic)

if '__main__' == __name__:
    import torchtext.vocab as Vocab
    from torch import nn
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    from utility.model_train import train_device

    from learn_pytorch.TextCnn.model import TextCnn
    from learn_pytorch.TextCnn.process_data import process_data, get_token, get_vocab, get_embedding
    from learn_pytorch.TextCnn.read_data import read_calimdb

    # glove_vocab = Vocab.GloVe(name='6B', dim=100,cache='D:\catche\glove')

    embedding_vocab = load_embedding_struct('../../data/word2vec', '../../data/index_file')

    batch_size = 64
    lr = 0.001
    num_epoch = 10
    train_data = read_calimdb('train')
    test_data = read_calimdb('test')
    word_vocab = get_vocab(train_data)
    embedding_count = len(word_vocab.stoi)

    train_token = get_token(train_data)
    test_token = get_token(test_data)
    train_feature, train_label = process_data(train_data, train_token, word_vocab)
    test_feature, test_label = process_data(test_data, test_token, word_vocab)

    embedding = get_embedding(word_vocab, embedding_vocab)

    train_iter = DataLoader(TensorDataset(train_feature, train_label), batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=True)

    net = TextCnn(embedding_count=embedding_count, embedding_size=100,convs=[(3, 100), (4, 100), (5, 100)], outputs=2)
    net.embedding.weight.data.copy_(embedding)
    net.embedding.weight.requires_grad = True

    net.embedding_const.weight.data.copy_(embedding)
    net.embedding.weight.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_device(train_iter, test_iter, net, nn.CrossEntropyLoss(), lr, num_epoch, device, 'Adam')