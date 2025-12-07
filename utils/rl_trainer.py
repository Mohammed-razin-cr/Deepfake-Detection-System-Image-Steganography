"""Reinforcement Learning Trainer - Fine-tunes model based on user feedback."""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from pathlib import Path
from datetime import datetime


class RLTrainer:
    """Reinforcement Learning trainer using user feedback."""
    
    def __init__(self, model, device='cpu', learning_rate=0.0001):
        """
        Initialize RL Trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for fine-tuning
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.feedback_log = 'results/feedback_log.json'
        self.training_history = 'results/rl_training_history.json'
        self.model.to(device)
        
    def train_on_feedback(self, image_tensor, actual_label, prediction):
        """
        Train model on single feedback instance.
        
        Args:
            image_tensor: Preprocessed image tensor
            actual_label: True label (0 for real, 1 for fake)
            prediction: Model's original prediction
            
        Returns:
            dict: Training metrics
        """
        # Keep model in eval mode because batch norm doesn't work with batch size 1
        # Only the optimizer will update gradients for parameters
        self.model.eval()
        
        image_tensor = image_tensor.to(self.device)
        target = torch.tensor([actual_label], dtype=torch.long).to(self.device)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            logits = self.model(image_tensor)
            loss = self.criterion(logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.softmax(logits / 2.0, dim=1)  # Apply temperature
            predicted_label = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
        
        was_correct = (prediction == actual_label)
        is_now_correct = (predicted_label == actual_label)
        
        metrics = {
            'loss': loss.item(),
            'was_correct': was_correct,
            'is_now_correct': is_now_correct,
            'improvement': is_now_correct and not was_correct,
            'confidence': float(confidence),
            'actual_label': int(actual_label),
            'original_prediction': int(prediction),
            'new_prediction': int(predicted_label)
        }
        
        return metrics
    
    def get_feedback_count(self):
        """Get total number of feedback entries."""
        try:
            if os.path.exists(self.feedback_log):
                with open(self.feedback_log, 'r') as f:
                    feedback_list = json.load(f)
                    return len(feedback_list)
        except:
            pass
        return 0
    
    def get_training_stats(self):
        """Get RL training statistics."""
        try:
            if os.path.exists(self.training_history):
                with open(self.training_history, 'r') as f:
                    history = json.load(f)
                    if history:
                        total_trainings = len(history)
                        improvements = sum(1 for h in history if h.get('improvement', False))
                        avg_loss = sum(h.get('loss', 0) for h in history) / len(history) if history else 0
                        return {
                            'total_trainings': total_trainings,
                            'improvements': improvements,
                            'avg_loss': float(avg_loss)
                        }
        except:
            pass
        return {'total_trainings': 0, 'improvements': 0, 'avg_loss': 0.0}
    
    def save_training_record(self, metrics, file_name=''):
        """Save training record to history."""
        try:
            history = []
            os.makedirs('results', exist_ok=True)
            
            if os.path.exists(self.training_history):
                with open(self.training_history, 'r') as f:
                    history = json.load(f)
            
            record = {
                'timestamp': datetime.now().isoformat(),
                'file_name': file_name,
                **metrics
            }
            
            history.append(record)
            
            # Keep only last 1000 records
            history = history[-1000:]
            
            with open(self.training_history, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"✓ RL Training recorded: Loss={metrics['loss']:.4f}, "
                  f"Improvement={'Yes' if metrics['improvement'] else 'No'}")
            
            return True
        except Exception as e:
            print(f"⚠ Failed to save training record: {e}")
            return False
    
    def save_model_checkpoint(self, checkpoint_path='trained_models/rl_checkpoint.pth'):
        """Save model checkpoint."""
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"✓ RL Checkpoint saved: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            return False
    
    def should_retrain(self):
        """Check if model should be retrained (every N feedback instances)."""
        # Retrain every 5 feedback instances
        return self.get_feedback_count() % 5 == 0 and self.get_feedback_count() > 0
