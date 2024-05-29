import torch
import torch.nn as nn
from torchvision import models


class DenseNetWithRH(nn.Module):
    def __init__(self):
        super(DenseNetWithRH, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove original classifier
        self.fc1 = nn.Linear(num_features + 1, 512)  # +1 for RH feature
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, rh):
        features = self.densenet(x)
        x = torch.cat((features, rh), dim=1)  # Concatenate features and RH
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
