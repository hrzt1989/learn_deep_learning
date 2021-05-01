import zipfile

def get_jaychou_lyrics(file_path = '../../data/jaychou_lyrics.txt.zip'):
    result = None
    with zipfile.ZipFile(file_path) as zn:
        with zn.open('jaychou_lyrics.txt') as f:
            result = f.read().decode('utf-8')
    if result is not None:
        result = result.replace('\n', ' ').replace('\r', ' ')
    return result

if '__main__' == __name__:
    from utility.gen_char_index import gen_char_index, gen_index_list
    from utility.char_data_iter import char_data_iter_random, char_data_iter_consecutive
    result = get_jaychou_lyrics()
    index_to_char_list, char_to_index_dict = gen_char_index(result)
    char_index_list = gen_char_index(result, char_to_index_dict)
