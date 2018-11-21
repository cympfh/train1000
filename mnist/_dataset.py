from .train1000 import mnist
import random
import torch
import torch.utils.data
import torchvision.transforms


class MNIST(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        """
        Parameters
        ----------
        x: numpy.array, float
        y: numpy.array, float
        """
        self.x = x.transpose((0, 3, 1, 2))  # 28x28x1 => 1x28x28
        self.x = self.x * 2.0 - 1.0  # [0, 1] => [-1, 1]
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: Index

        Returns
        -------
        (tensor, tensor)
        """
        x = torch.from_numpy(self.x[index])
        y = torch.from_numpy(self.y[index])
        if self.transform is not None:
            x, y = self.transform((x, y))
        return (x, y)


class RandomErasing:

    def __init__(self, num=1, size=3):
        self.num = num
        self.size = size

    def __call__(self, data):
        x, y = data
        for _ in range(self.num):
            h = x.size(1)
            w = x.size(2)
            i = random.randrange(h - self.size + 1)
            j = random.randrange(w - self.size + 1)
            x[0, i: i + self.size, j: j + self.size] = 0.0
        return (x, y)


class GaussianNoise:

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, data):
        x, y = data
        x += torch.randn(x.size()) * self.sigma
        return (x, y)


def load(aug=False, batch_size=30):

    (x_train, y_train), (x_test, y_test) = mnist()

    if aug:
        augmentation = torchvision.transforms.Compose([
            GaussianNoise()
        ])
    else:
        augmentation = torchvision.transforms.Compose([])

    set_train = MNIST(x_train, y_train, transform=augmentation)
    set_test = MNIST(x_test, y_test)

    loader_train = torch.utils.data.DataLoader(
        set_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    loader_test = torch.utils.data.DataLoader(
        set_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    return loader_train, loader_test
