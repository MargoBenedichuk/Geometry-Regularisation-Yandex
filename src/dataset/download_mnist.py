import torchvision.datasets as datasets

def download_mnist():
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    print("Датасет успешно скачан в папку ./data/MNIST/raw/")

