import collections
import random
import math
from utility.gen_char_index import gen_char_index
def gen_word_count_dict(data_set, filter_count = 5):
    word_cout_dic = collections.Counter([one_word for st in data_set for one_word in st])
    word_count_dic = dict(filter(lambda x: x[1] > filter_count, word_cout_dic.items()))
    all_word_count = sum(list(word_count_dic.values()))
    return word_count_dic, all_word_count

def trans_word_to_index(data_set, word_cout_dic):
    index_to_char_list, char_to_index_dict = gen_char_index(set(word_cout_dic.keys()))
    data_set_index = [[char_to_index_dict[one_word] for one_word in line if one_word in word_cout_dic] for line in data_set]
    return data_set_index, index_to_char_list, char_to_index_dict

def sample_data_set(data_set_index, word_count_dict, index_to_word, all_word_count, t = 1e-4):
    sampled_data_set_index = []
    for st in data_set_index:
        sampled_st = []
        for one_word_index in st:
            one_word = index_to_word[one_word_index]
            word_count = word_count_dict[one_word]
            if random.uniform(0,1) < max(1 - math.sqrt(t / word_count * all_word_count), 0):
                continue
            sampled_st.append(one_word_index)
        sampled_data_set_index.append(sampled_st)
    return sampled_data_set_index

def compare_count(token_index, data_set_index, sampled_data_set_index):
    count = sum([st.count(token_index) for st in data_set_index])
    count_sampled = sum([st.count(token_index) for st in sampled_data_set_index])
    print('count', count, 'count_sampled', count_sampled)

def gen_center_context(sampled_data_set_index, max_window_size):
    centers = []
    context = []

    for st_index in sampled_data_set_index:
        if len(st_index) < 2:
            continue
        centers += st_index
        for center_id in range(len(st_index)):
            window_size = random.randint(1, max_window_size)
            context_ids = list(range(max(0, center_id - window_size), min(center_id + 1 + window_size, len(st_index))))
            context_ids.remove(center_id)
            context.append( [st_index[index] for index in context_ids])

    return centers, context

def negative_sample(context_list, word_count_dict, index_to_word_list, negative_num = 5):

    word_weight = [word_count_dict[one_word] ** 0.75 for one_word in index_to_word_list]
    candidates = list(range(len(word_weight)))
    all_negative = []
    neg_candidates = []
    i = 0
    for context in context_list:
        negative = []
        while len(negative) < len(context) * negative_num:
            if i == len(neg_candidates):
                neg_candidates = random.choices(candidates, word_weight, k=int(1e5))
                i = 0
            one_neg = neg_candidates[i]
            if one_neg not in context:
                negative.append(one_neg)
            i += 1
        all_negative.append(negative)
    return all_negative


if '__main__' == __name__:
    from learn_pytorch.word2vec.load_data import load_sentence
    data_set = load_sentence('../../data/ptb/ptb.train.txt')
    word_count_dic, all_word_count = gen_word_count_dict(data_set)
    data_set_index, index_to_word_list, word_to_index_dic = trans_word_to_index(data_set, word_count_dic)
    sampled_data_set = sample_data_set(data_set_index, word_count_dic, index_to_word_list, all_word_count)
    compare_count(word_to_index_dic['join'], data_set_index, sampled_data_set)
    centers, context = gen_center_context(sampled_data_set, max_window_size=5)
    negetive_context = negative_sample(context, word_count_dic, index_to_word_list)