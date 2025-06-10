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
    
# Medium ConvNet for MNIST
class MediumConvNet(nn.Module):
    def __init__(self):
        super(MediumConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

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

class MediumConvNetCIFAR(nn.Module):
    """Medium ConvNet adapted for CIFAR-10 (3 channels, 32x32 images)"""
    def __init__(self):
        super(MediumConvNetCIFAR, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 16x16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 8x8
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 4x4

        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = self.pool(F.relu(self.conv4(x)))  # 4x4 -> 2x2

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
