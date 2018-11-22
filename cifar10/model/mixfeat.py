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
        raise NotImplemented
