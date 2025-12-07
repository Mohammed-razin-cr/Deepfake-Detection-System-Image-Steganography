"""CNN-based model for deepfake detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for deepfake detection.
    Extracts spatial features from images/video frames.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize CNN model.
        
        Args:
            num_classes (int): Number of output classes (fake/real)
            pretrained (bool): Whether to use pretrained weights
        """
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual-like blocks
        self.conv2 = self._make_conv_block(64, 128, kernel_size=3)
        self.conv3 = self._make_conv_block(128, 256, kernel_size=3)
        self.conv4 = self._make_conv_block(256, 512, kernel_size=3)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size=3):
        """Create a convolutional block with batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                     stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """Extract feature vector before classification."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
