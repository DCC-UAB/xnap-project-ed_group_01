import torch
import torch.nn as nn
from torchvision.models import resnet18

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()
        self.resnet = resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)