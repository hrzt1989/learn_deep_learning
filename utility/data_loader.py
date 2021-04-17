from torch.utils.data import DataLoader
import sys
def data_loader(datas, workers_num, batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = workers_num
    return DataLoader(datas, batch_size=batch_size, shuffle=True, num_workers=num_workers)