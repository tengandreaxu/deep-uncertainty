import torch
import torch.nn as nn
import torch.nn.functional as F


class MediumCNN(nn.Module):
    def __init__(self):

        super(MediumCNN, self).__init__()

        self.channels = (64, 128, 256, 256)
        self.pools = (2, 2, 2, 2)
        self.kernel_sizes = (3, 3, 3, 3)
        self.Hn = 32
        self.Wn = 32
        self.Cn = 3
        self.dropout = 0.1
        self.batch_size = 128
        self.epochs = 40
        self.learning_rate = 1.6e-3
        self.halving_epochs = 10

        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=self.channels[0],
            kernel_size=self.kernel_sizes[0],
            padding="same",
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
