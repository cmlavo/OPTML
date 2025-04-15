import torch
import torch.nn as nn
import torch.nn.functional as F
import random

"""
Here we implement a few models
"""

# ConvNet for MNIST
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Second convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully connected layer: input from flattened 7x7x64, output to 128 hidden units
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Output layer: 128 -> 10 classes
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        #self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14 (for MNIST images)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7

        x = x.view(x.size(0), -1)  # Flatten

        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        #x = self.dropout(x)
        x = self.fc2(x)  # logits, no softmax

        return x
    


# Small ConvNet for MNIST
class SmallConvNet(nn.Module):
    def __init__(self):
        super(SmallConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 49)

        self.fc2 = nn.Linear(49, 10)

        # Dropout for regularization
        #self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7

        x = x.view(x.size(0), -1)  # Flatten

        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        #x = self.dropout(x)
        x = self.fc2(x)  # logits, no softmax

        return x
    