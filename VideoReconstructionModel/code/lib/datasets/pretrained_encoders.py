import torch.nn as nn
import torch
from torchvision import models
import numpy as np

class EncodingBackbone(nn.Module):
    def __init__(self, encoding_size=256):
        super(EncodingBackbone, self).__init__()

        # Load the pretrained ResNet-50 backbone
        #self.backbone = models.resnet50(pretrained=True)
        self.backbone = models.resnet18(pretrained=True)

        # Remove the fully connected layers (classification head) and average pooling layer from the pretrained ResNet-50
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add a global average pooling layer to reduce spatial dimensions
        #self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Linear layer for final encoding (to reduce the dimensionality of the output)
        #self.encoding_layer = nn.Linear(2048, encoding_size)

        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)

        # Reshape for the linear layer
        #x = x.view(x.size(0), -1)

        # Linear layer for final encoding
        #encoding = self.encoding_layer(x)

        return x