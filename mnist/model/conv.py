import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2)
        self.lin = nn.Linear(256, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 256)
        y = self.lin(x)
        return y
