if '__main__' == __name__:
    import torch
    from torch import nn
    from utility.model_train import train_device
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader
    from utility.flatten_layer import FlattenLayer

    batch_size = 256
    lr = 0.001
    num_epochs = 5


    train_MNST, test_MNST = get_fashion_MNST()
    train_data_iter = data_loader(train_MNST, 5, batch_size)
    test_data_iter = data_loader(test_MNST, 5, batch_size)

    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),  # kernel_size, stride
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        FlattenLayer(),
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_device(train_data_iter,
                 test_data_iter,
                 net,
                 nn.CrossEntropyLoss(),
                 lr,
                 num_epochs,
                 device,
                 'Adam')