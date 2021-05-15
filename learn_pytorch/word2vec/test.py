import torch
def get_similar_tokens(query_token, k, token_to_idx, idx_to_token,embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


if '__main__' == __name__:
    import sys
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

    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4
    num_epoch = 10
    lr = 0.01

    data_set = load_sentence('../../data/ptb/ptb.train.txt')
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
    net = train(data_iter, net, BinaryCrossEntropyLoss(), num_epoch, lr, batch_size, 'Adam', device)
    get_similar_tokens('intel', 3, word_to_index_dic, index_to_word_list, net.net[0])