import torch
import torch.nn as nn
import torch.nn.functional as F

class PHOCNet(nn.Module):
    def __init__(self, num_classes):
        super(PHOCNet, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)

        # Spatial pyramid pooling layers
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        # Fully connected layers
        self.fc1 = nn.Linear(7168, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv4(x))
        x = self.dropout1(x)

        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x)
        x_pool3 = self.pool3(x)

        x_pool1 = x_pool1.view(x_pool1.size(0), -1)
        x_pool2 = x_pool2.view(x_pool2.size(0), -1)
        x_pool3 = x_pool3.view(x_pool3.size(0), -1)

        x = torch.cat((x_pool1, x_pool2, x_pool3), dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x