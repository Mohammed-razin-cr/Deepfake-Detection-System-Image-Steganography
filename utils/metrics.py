"""Metrics calculation for model evaluation."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """
        Update metrics with batch results.
        
        Args:
            predictions (torch.Tensor or np.ndarray): Predicted labels
            targets (torch.Tensor or np.ndarray): Ground truth labels
            probabilities (torch.Tensor or np.ndarray): Prediction probabilities
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        if probabilities is not None:
            self.probabilities.extend(probabilities)
    
    def compute_metrics(self):
        """
        Compute all evaluation metrics.
        
        Returns:
            dict: Dictionary of computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted'),
        }
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm
        
        # ROC-AUC if probabilities available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            if probabilities.ndim > 1:
                probabilities = probabilities[:, 1]  # Get fake class probability
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            dict: Classification metrics
        """
        from sklearn.metrics import classification_report
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        report = classification_report(
            targets, predictions,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        
        return report
