import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class ResNet50(nn.Module):
    def __init__(self, num_classes=7, channels=3):
        super(ResNet50, self).__init__()
        # Initial layers
        self.conv_layer_s2_same = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Load pre-trained ResNet50 model
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Extract layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Fully connected layers
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layer_s2_same(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x