import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.channels = (16, 32, 32)
        self.pools = (2, 2, 2)
        self.kernel_sizes = (3, 3, 3)

        self.dropout_ratio = 0.1
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 1.6e-3
        self.halving_epochs = 10

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.channels[0],
            kernel_size=self.kernel_sizes[0],
            padding="same",
            stride=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=self.kernel_sizes[1],
            padding="same",
            stride=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            kernel_size=self.kernel_sizes[2],
            padding="same",
            stride=1,
        )

        self.fc1 = nn.Linear(self.channels[2] * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x
