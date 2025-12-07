"""Inference engine for model predictions."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class InferenceEngine:
    """Engine for running inference with detection models."""
    
    def __init__(self, model, device='cpu', temperature=2.0):
        """
        Initialize inference engine.
        
        Args:
            model: PyTorch model
            device (str): Device to run model on ('cpu' or 'cuda')
            temperature (float): Temperature for scaling logits (higher = less confident)
        """
        self.model = model
        self.device = device
        self.temperature = temperature  # Temperature scaling to reduce overconfidence
        self.model.to(device)
        self.model.eval()
        
    def detect_image(self, image_tensor):
        """
        Run detection on image.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            dict: Detection results
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_prediction_confidence'):
                # Ensemble model
                results = self.model.get_prediction_confidence(image_tensor)
                # Apply temperature scaling to reduce overconfidence
                real_prob = results['real_probability']
                fake_prob = results['fake_probability']
                total = real_prob + fake_prob
                # Normalize and apply temperature scaling
                real_prob_scaled = (real_prob / (total + 1e-8)) ** (1 / self.temperature)
                fake_prob_scaled = (fake_prob / (total + 1e-8)) ** (1 / self.temperature)
                # Re-normalize after temperature scaling
                total_scaled = real_prob_scaled + fake_prob_scaled
                results['real_probability'] = float(real_prob_scaled / (total_scaled + 1e-8))
                results['fake_probability'] = float(fake_prob_scaled / (total_scaled + 1e-8))
                # Update is_fake based on scaled probabilities
                results['is_fake'] = results['fake_probability'] > results['real_probability']
                results['confidence'] = max(results['fake_probability'], results['real_probability'])
            else:
                # Single model
                logits = self.model(image_tensor)
                # Apply temperature scaling
                scaled_logits = logits / self.temperature
                probs = F.softmax(scaled_logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = torch.max(probs, dim=1)[0].item()
                
                results = {
                    'prediction': pred,
                    'confidence': conf,
                    'is_fake': pred == 1,
                    'fake_probability': probs[0, 1].item(),
                    'real_probability': probs[0, 0].item()
                }
        
        return results
    
    def detect_video(self, video_tensor):
        """
        Run detection on video.
        
        Args:
            video_tensor (torch.Tensor): Preprocessed video tensor
            
        Returns:
            dict: Detection results
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        video_tensor = video_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(video_tensor)
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs, dim=1)[0].item()
        
        return {
            'prediction': pred,
            'confidence': conf,
            'is_fake': pred == 1,
            'fake_probability': probs[0, 1].item(),
            'real_probability': probs[0, 0].item()
        }
    
    def batch_detect(self, batch_tensor):
        """
        Run detection on batch of images.
        
        Args:
            batch_tensor (torch.Tensor): Batch of preprocessed images
            
        Returns:
            dict: Batch detection results
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch_tensor)
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1)[0]
        
        results = []
        for i in range(len(batch_tensor)):
            results.append({
                'prediction': preds[i].item(),
                'confidence': confs[i].item(),
                'is_fake': preds[i].item() == 1,
                'fake_probability': probs[i, 1].item(),
                'real_probability': probs[i, 0].item()
            })
        
        return results
    
    def save_model(self, save_path):
        """
        Save model to file.
        
        Args:
            save_path (str): Path to save model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
    
    def load_model(self, load_path):
        """
        Load model from file.
        
        Args:
            load_path (str): Path to load model from
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.eval()
