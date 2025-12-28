import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network for 28x28 grayscale image classification.

    Architecture:
        - Conv2d(1, 32, 3x3) + MaxPool2d(2x2) -> 32 x 14 x 14
        - Conv2d(32, 64, 3x3) + MaxPool2d(2x2) -> 64 x 7 x 7
        - Fully connected layers: 3136 -> 128 -> 1
        - Output: Single logit for binary classification

    Attributes:
        conv1: First convolutional layer (1 -> 32 channels).
        conv2: Second convolutional layer (32 -> 64 channels).
        pool: Max pooling layer (2x2).
        fc1: First fully connected layer (3136 -> 128).
        fc2: Second fully connected layer (128 -> 1).
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (N, 1, 28, 28).

        Returns:
            Output logits of shape (N,) for binary classification.
        """
        # x: (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (N, 64, 7, 7)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(1)
