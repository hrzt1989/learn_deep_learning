def get_st_list(data):
    result = []
    for one_data in data:
        result.append(one_data[0])
    return result

if '__main__' == __name__:

    from learn_pytorch.word2vec.train_word2vec import train_word2vec_embedding
    from learn_pytorch.TextCnn.read_data import read_calimdb

    train_data = read_calimdb(folder='train')
    test_data = read_calimdb(folder='test')
    st_lst = []
    st_lst.extend(get_st_list(train_data))
    st_lst.extend(get_st_list(test_data))

    train_word2vec_embedding(st_lst, '../../data/word2vec', '../../data/index_file')
