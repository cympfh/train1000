import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.elu(self.drop(self.bn1(F.max_pool2d(self.conv1(x), 2))))
        x = F.elu(self.drop(self.bn2(F.max_pool2d(self.conv2(x), 2))))
        x = x.view(-1, 20)
        y = self.fc2(F.elu(self.fc1(x)))
        return y
