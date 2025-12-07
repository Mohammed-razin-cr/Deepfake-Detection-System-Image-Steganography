"""LSTM-based model for temporal deepfake detection in videos."""

import torch
import torch.nn as nn
from torchvision import models


class LSTMModel(nn.Module):
    """
    LSTM model for temporal deepfake detection in videos.
    Processes sequence of frames to capture temporal inconsistencies.
    """
    
    def __init__(self, num_classes=2, sequence_length=16, pretrained=True):
        """
        Initialize LSTM model.
        
        Args:
            num_classes (int): Number of output classes (fake/real)
            sequence_length (int): Number of frames in sequence
            pretrained (bool): Whether to use pretrained CNN backbone
        """
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # CNN feature extractor (ResNet50)
        self.cnn_backbone = models.resnet50(pretrained=pretrained)
        cnn_features = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        lstm_out_size = 256 * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape 
                             (batch_size, seq_length, 3, H, W)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract CNN features from each frame
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use final hidden state for classification
        final_state = h_n[-1]  # Last layer's hidden state
        logits = self.classifier(final_state)
        
        return logits
    
    def get_temporal_features(self, x):
        """
        Extract temporal features for analysis.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_length, 3, H, W)
            
        Returns:
            torch.Tensor: LSTM output features
        """
        batch_size, seq_len, c, h, w = x.size()
        
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(cnn_features)
        return lstm_out
