import collections
import torch
from torchtext import vocab

def get_token(data):

    def tokenizer(one_data):
        return [token.lower() for token in one_data.split(' ')]
    return [tokenizer(one_data) for one_data, _ in data]

def get_vocab(data):
    st_list = get_token(data)
    token_count_dict = collections.Counter([token for st in st_list for token in st])
    return vocab.Vocab(token_count_dict, min_freq=5)

def get_embedding(token_vocab, glove_vocab):
    embedding_len = 100
    token_count = len(token_vocab.itos)
    embedding = torch.zeros((token_count, embedding_len), dtype=torch.float32)
    err_count = 0
    for token in token_vocab.itos:
        try:
            index = token_vocab.stoi[token]
            glove_index = glove_vocab.stoi[token]
            embedding[index, :] = glove_vocab.vectors[glove_index]
        except KeyError:
            err_count += 1
    return embedding

def process_data(data, token_data, token_vocab):
    # token_data = get_token(data)
    # token_vocab = get_vocab(token_data)

    def pad(index_list):
        max_len = 500
        return index_list[:max_len] if len(index_list) >= max_len else index_list + [0] * (max_len - len(index_list))
    labels = [label for _, label in data]

    features = [pad([token_vocab.stoi[token] for token in st]) for st in token_data]
    return torch.tensor(features, dtype=torch.int64), torch.tensor(labels, dtype=torch.int64)

if '__main__' == __name__:
    pass