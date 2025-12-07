"""Vision Transformer model for deepfake detection."""

import torch
import torch.nn as nn
from torchvision.models import vision_transformer


class ViTModel(nn.Module):
    """
    Vision Transformer model for deepfake detection.
    Uses attention mechanism to capture global dependencies in images.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize Vision Transformer model.
        
        Args:
            num_classes (int): Number of output classes (fake/real)
            pretrained (bool): Whether to use ImageNet pretrained weights
        """
        super(ViTModel, self).__init__()
        self.num_classes = num_classes
        
        # Load Vision Transformer Base
        self.vit = vision_transformer.vit_b_16(pretrained=pretrained)
        
        # Get number of features from original head
        num_features = self.vit.heads.head.in_features
        
        # Replace classification head
        self.vit.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
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
        return self.vit(x)
    
    def get_attention_weights(self, x):
        """
        Extract attention weights from the transformer.
        Useful for interpretability and visualization.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Attention weights
        """
        # Reshape and permute image
        n, c, h, w = x.shape
        p = self.vit.patch_embed.patch_size
        torch._assert(h == self.vit.image_size, "Wrong image height")
        torch._assert(w == self.vit.image_size, "Wrong image width")
        n_h = h // p
        n_w = w // p
        
        # Patch embedding
        x = self.vit._process_input(x)
        n, _, c = x.shape
        
        # Expand class token
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Get attention from first layer
        x = self.vit.encoder(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract feature vector before classification.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature vector
        """
        # Process input
        x = self.vit._process_input(x)
        n, _, c = x.shape
        
        # Expand class token
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Pass through encoder
        x = self.vit.encoder(x)
        
        # Return class token
        return x[:, 0]
