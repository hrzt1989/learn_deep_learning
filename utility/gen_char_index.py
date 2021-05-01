def gen_char_index(char_data):
    index_to_char_list = list(set(char_data))
    char_to_index_dict = {}
    for index, char in enumerate(index_to_char_list):
        char_to_index_dict[char] = index
    return index_to_char_list, char_to_index_dict

def gen_index_list(char_data,char_to_index_dict):
    return [char_to_index_dict[one_char] for one_char in char_data]