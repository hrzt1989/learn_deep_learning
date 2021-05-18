import torch
import sys
import pickle
from torch.utils.data import DataLoader
from learn_pytorch.word2vec.loss import BinaryCrossEntropyLoss
from learn_pytorch.word2vec.trainer import train
from learn_pytorch.word2vec.model import WordEmbedding
from learn_pytorch.word2vec.data_loader import trans_batch, WordDataSet
from learn_pytorch.word2vec.load_data import load_sentence
from learn_pytorch.word2vec.process_data import gen_word_count_dict
from learn_pytorch.word2vec.process_data import trans_word_to_index
from learn_pytorch.word2vec.process_data import sample_data_set
from learn_pytorch.word2vec.process_data import gen_center_context
from learn_pytorch.word2vec.process_data import negative_sample

def train_word2vec_embedding(data_set, embedding_path, index_patch):

    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4
    num_epoch = 10
    lr = 0.01
    word_count_dic, all_word_count = gen_word_count_dict(data_set)
    data_set_index, index_to_word_list, word_to_index_dic = trans_word_to_index(data_set, word_count_dic)
    sampled_data_set = sample_data_set(data_set_index, word_count_dic, index_to_word_list, all_word_count)
    centers, context = gen_center_context(sampled_data_set, max_window_size=5)
    negetive_context = negative_sample(context, word_count_dic, index_to_word_list)

    word_data_set = WordDataSet(centers, context, negetive_context)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    data_iter = DataLoader(word_data_set, batch_size, shuffle=True, collate_fn=trans_batch, num_workers=num_workers)
    net = WordEmbedding(len(index_to_word_list), 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = train(data_iter, net, BinaryCrossEntropyLoss(), num_epoch, lr, batch_size, 'Adam', device, show=True)
    net.to(torch.device('cpu'))
    torch.save(net.net[0].weight, embedding_path)
    index_file = open(index_patch, 'wb')
    pickle.dump((index_to_word_list, word_to_index_dic), index_file)
    index_file.close()

if '__main__' == __name__:
    pass
