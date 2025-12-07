"""Ensemble model combining multiple architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_model import CNNModel
from .resnext_model import ResNextModel
from .vision_transformer import ViTModel


class EnsembleDetector(nn.Module):
    """
    Ensemble model combining CNN, ResNext, and Vision Transformer.
    Uses weighted averaging and gating mechanism for final prediction.
    """
    
    def __init__(self, num_classes=2, use_lstm=False):
        """
        Initialize ensemble model.
        
        Args:
            num_classes (int): Number of output classes
            use_lstm (bool): Whether to include LSTM for video
        """
        super(EnsembleDetector, self).__init__()
        self.num_classes = num_classes
        self.use_lstm = use_lstm
        
        # Individual models
        self.cnn_model = CNNModel(num_classes=num_classes, pretrained=True)
        self.resnext_model = ResNextModel(num_classes=num_classes, pretrained=True)
        self.vit_model = ViTModel(num_classes=num_classes, pretrained=True)
        
        # Gating network for adaptive weighting (using logits instead of features)
        self.gating_network = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_ensemble_logits=False):
        """
        Forward pass through ensemble.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            return_ensemble_logits (bool): Return individual model predictions
            
        Returns:
            torch.Tensor: Final logits
            Optional[Tuple[torch.Tensor, ...]]: Individual model logits
        """
        # Get predictions from each model
        cnn_logits = self.cnn_model(x)
        resnext_logits = self.resnext_model(x)
        vit_logits = self.vit_model(x)
        
        # Concatenate logits for gating and fusion
        combined_logits = torch.cat([cnn_logits, resnext_logits, vit_logits], dim=1)
        
        # Get adaptive weights using logits
        weights = self.gating_network(combined_logits)  # Shape: (batch_size, 3)
        
        # Weighted combination of logits
        weighted_logits = (
            weights[:, 0:1] * cnn_logits +
            weights[:, 1:2] * resnext_logits +
            weights[:, 2:3] * vit_logits
        )
        
        # Fusion layer for final prediction
        final_logits = self.fusion(combined_logits)
        
        if return_ensemble_logits:
            return final_logits, (cnn_logits, resnext_logits, vit_logits)
        
        return final_logits
    
    def get_prediction_confidence(self, x):
        """
        Get prediction with confidence scores.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            dict: Prediction results with confidence scores
        """
        final_logits, ensemble_logits = self.forward(x, return_ensemble_logits=True)
        
        # Convert to probabilities
        final_probs = F.softmax(final_logits, dim=1)
        ensemble_probs = [
            F.softmax(logits, dim=1) for logits in ensemble_logits
        ]
        
        # Get predictions
        final_pred = torch.argmax(final_probs, dim=1)
        final_conf = torch.max(final_probs, dim=1)[0]
        
        # Individual model predictions
        ensemble_preds = [
            torch.argmax(probs, dim=1) for probs in ensemble_probs
        ]
        ensemble_confs = [
            torch.max(probs, dim=1)[0] for probs in ensemble_probs
        ]
        
        return {
            'prediction': final_pred.item(),
            'confidence': final_conf.item(),
            'is_fake': final_pred.item() == 1,
            'fake_probability': final_probs[0, 1].item(),
            'real_probability': final_probs[0, 0].item(),
            'ensemble_predictions': [p.item() for p in ensemble_preds],
            'ensemble_confidences': [c.item() for c in ensemble_confs]
        }
