import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features + 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,1)

    def forward(self, aod, rh, x):
        features = self.densenet(x)
        input = torch.cat((aod, rh, features), dim=1)
        output = torch.relu(self.fc1(input))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        return output
