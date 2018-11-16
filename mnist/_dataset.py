from .train1000 import mnist
import torch
import torch.utils.data


class MNIST(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        """
        Parameters
        ----------
        x: numpy.array, float
        y: numpy.array, float
        """
        self.x = x
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
            x, y = self.transform(x, y)
        return (x, y)


def load(batch_size=30):

    (x_train, y_train), (x_test, y_test) = mnist()

    set_train = MNIST(x_train, y_train)
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
