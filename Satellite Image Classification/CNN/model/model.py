"""
model.py
--------
Defines the ConvolutionalNetwork used for satellite image classification.
The model takes 5-channel input images (R, G, B, R-G, G-B) and outputs 4 classes:
cloudy, desert, green area, water.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    """
    A CNN for satellite image classification using 5-channel input:
    - R, G, B
    - (R - G), (G - B)

    Features:
    - 3 convolutional blocks
    - 5 input channels
    - dynamic flatten dimension calculation (works with any input size)
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # 5 input channels now (R,G,B,R-G,G-B)
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Compute flatten dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 5, 224, 224)
            x = F.max_pool2d(F.relu(self.conv1(dummy)), 2, 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
            x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)
            self.flatten_dim = x.numel()

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
