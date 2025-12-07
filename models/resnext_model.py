"""ResNext-based model for deepfake detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNextModel(nn.Module):
    """
    ResNext-50 model for deepfake detection.
    Uses grouped convolutions for better feature extraction.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize ResNext model.
        
        Args:
            num_classes (int): Number of output classes (fake/real)
            pretrained (bool): Whether to use ImageNet pretrained weights
        """
        super(ResNextModel, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNext50
        self.resnext = models.resnext50_32x4d(pretrained=pretrained)
        
        # Modify final layer
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Identity()  # Remove final layer
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        features = self.resnext(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """
        Extract feature vector before classification.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature vector
        """
        return self.resnext(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.resnext.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.resnext.parameters():
            param.requires_grad = True
