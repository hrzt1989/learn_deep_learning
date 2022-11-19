import os
import tarfile
import random
def decompression(file_path, file_name):
    fname = os.path.join(file_path, file_name)
    with tarfile.open(fname, 'r') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, file_path)

def read_calimdb(folder='train', data_root = '../../data/aclImdb'):
    result = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file),'rb') as one_file:
                review = one_file.read().decode('utf-8').replace('\n', ' ').lower()
                result.append([review, 1 if label == 'pos' else 0])
    random.shuffle(result)
    return result
if '__main__' == __name__:
    from learn_pytorch.TextCnn.process_data import get_vocab
    # decompression('../../data', 'aclImdb_v1.tar.gz')

    data = read_calimdb()
    vocab = get_vocab(data)
    print('data', len(data))