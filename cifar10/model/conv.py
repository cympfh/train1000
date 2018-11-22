import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.elu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = self.drop(x)
        x = F.elu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = F.max_pool2d(x, 3)
        x = x.view(-1, 128)
        y = self.fc2(F.elu(self.fc1(x)))
        return y
