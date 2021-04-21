import torch
from torch import nn
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.full_connections = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )
    def forward(self, image):
        features = self.convolutions(image)
        return self.full_connections(features.view(features.shape[0], -1))

if '__main__' == __name__:
    from torch.nn import init
    from torch import optim
    from utility.load_fashion_MNIST import get_fashion_MNST
    from utility.data_loader import data_loader
    from utility.model_train import train_device

    batch_size = 128
    lr = 0.001
    num_epoch = 5

    train_MNIST, test_MNIST = get_fashion_MNST(resize=224)
    train_data_iter = data_loader(train_MNIST, 5, batch_size)
    test_data_iter = data_loader(test_MNIST, 5, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AlexNet()
    print(net)

    train_device(train_data_iter, test_data_iter, AlexNet(), nn.CrossEntropyLoss(), lr, num_epoch, device, 'Adam')

