import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16, inception_v3

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes, type_model = "resnet"):
        super(CharacterClassifier, self).__init__()

        if type_model == "resnet":
            self.model = resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif type_model == "vggnet":
            self.model = vgg16(pretrained=True)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
        elif type_model == "inception":
            self.model = inception_v3(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError("Invalid type_model. Supported values: 'resnet', 'vggnet', 'inception'")

    def forward(self, x):
        return self.model(x)
