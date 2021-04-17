import torchvision
from torchvision import transforms
def get_fashion_MNST():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
    return mnist_train, mnist_test


if '__main__' == __name__:
    mnist_train, mnist_test = get_fashion_MNST()
    print('mnist_train', mnist_train)
    print('mnist_test', mnist_test)