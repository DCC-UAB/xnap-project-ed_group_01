import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP_basic(nn.Module):

    def __init__(self, n_out):
        super(MLP_basic, self).__init__()

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, n_out)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)

        return out