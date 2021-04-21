import torchvision
from torchvision import transforms
def get_fashion_MNST(resize = None):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transforms = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms)
    return mnist_train, mnist_test


if '__main__' == __name__:
    mnist_train, mnist_test = get_fashion_MNST()
    print('mnist_train', mnist_train)
    print('mnist_test', mnist_test)