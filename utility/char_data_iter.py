import random
import torch

def get_char_index(corpus_chars_index, pos, step):
    return corpus_chars_index[pos:pos + step]

def char_data_iter_random(corpus_chars_index,
                          num_step,
                          batch_size):

    sample_num = (len(corpus_chars_index) - 1) // num_step
    samples = [index for index in range(sample_num)]
    random.shuffle(samples)
    epoch_num = len(samples) // batch_size
    for epoch in range(epoch_num):
        base = epoch * batch_size
        batch_samples = samples[base : base + batch_size]
        x = [get_char_index(corpus_chars_index = corpus_chars_index,
                            pos=sample_index * num_step, step=num_step) for sample_index in batch_samples]
        y = [get_char_index(corpus_chars_index = corpus_chars_index,
                            pos=sample_index * num_step + 1, step=num_step) for sample_index in batch_samples]
        yield torch.tensor(x,  dtype = torch.int64), torch.tensor(y, dtype = torch.int64)

def char_data_iter_consecutive(corpus_chars_index,
                               num_step,
                               batch_size):
    batch_len = len(corpus_chars_index) // batch_size
    num_epoch = (batch_len - 1) // num_step

    corpus_chars_index_tensor = torch.\
                                tensor(corpus_chars_index[0 : batch_len * batch_size], dtype=torch.int64).\
                                view(batch_size, -1)
    for epoch in range(num_epoch):
        base = epoch * num_step
        x = corpus_chars_index_tensor[:, base : base + num_step]
        y = corpus_chars_index_tensor[:, base + 1 : base + num_step + 1]
        yield x, y


if '__main__' == __name__:
    my_seq = list(range(30))
    for X, Y in char_data_iter_consecutive(my_seq, batch_size=2, num_step=6):
        print('X: ', X, '\nY:', Y, '\n')