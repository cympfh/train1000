import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import numpy.random


pi = 3.14159265359


def shuffle(x: torch.Tensor):
    m = len(x)
    return x[numpy.random.permutation(m)]


class MixFeat(nn.Module):

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            r = numpy.random.normal(0, self.sigma)
            theta = numpy.random.uniform(-pi, pi)
            fx = shuffle(x)
            a = float(r * numpy.cos(theta))
            b = float(r * numpy.sin(theta))
            return x + a * x + b * fx
        else:
            return x

    def extra_repr(self):
        return f"sigma={self.sigma}"


class MixFeatConv(nn.Module):

    def __init__(self, sigma=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 10)
        self.mixfeat = MixFeat(sigma)

    def forward(self, x):
        x = F.elu(self.drop(self.bn1(F.max_pool2d(self.conv1(x), 2))))
        x = self.mixfeat(x)
        x = F.elu(self.drop(self.bn2(F.max_pool2d(self.conv2(x), 2))))
        x = self.mixfeat(x)
        x = x.view(-1, 20)
        y = self.fc2(F.elu(self.fc1(x)))
        return y
