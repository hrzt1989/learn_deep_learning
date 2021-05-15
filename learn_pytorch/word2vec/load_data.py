def load_sentence(file_path):
    result = []
    with open(file_path) as file:
        for line in file:
            sentences = line.split()
            result.append(sentences)
    return result

if '__main__' == __name__:
    result = load_sentence('../../data/ptb/ptb.train.txt')
    print('result', len(result))
    print(result[0])