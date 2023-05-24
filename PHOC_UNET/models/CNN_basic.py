import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_basic(nn.Module):

    def __init__(self, n_out, input_channels = 1):
        super(CNN_basic, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, n_out)

    def forward(self, x):
        out = self.conv_block1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = self.conv_block2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = self.conv_block3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = self.conv_block4(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        #out = F.dropout(out, p = 0.2, training = self.training)
        out = self.fc2(out)
        out = F.relu(out)
        #out = F.dropout(out, p = 0.2, training = self.training)
        out = self.fc3(out)
        out = torch.sigmoid(out)

        return out
