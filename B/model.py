import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN for 28x28 grayscale images
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (N, 64, 7, 7)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(1)
