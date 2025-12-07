"""Training script for deepfake detection models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from models import CNNModel, ResNextModel, ViTModel, LSTMModel, EnsembleDetector
from utils import MetricsCalculator


class DeepfakeDataset(Dataset):
    """Dataset for loading deepfake images."""
    
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Initialize dataset.
        
        Args:
            image_dir (str): Directory containing images
            labels_file (str): File with image names and labels
            transform: Image transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []
        
        # Load labels
        with open(labels_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.samples.append((parts[0], int(parts[1])))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample by index."""
        from PIL import Image
        
        img_name, label = self.samples[idx]
        img_path = self.image_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class Trainer:
    """Trainer class for model training."""
    
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=1e-5):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device (str): Device to train on
            lr (float): Learning rate
            weight_decay (float): Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_accuracy = 0
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for validation
            
        Returns:
            Tuple[float, float]: Validation loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        metrics = MetricsCalculator()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                metrics.update(predicted, labels)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, save_path, metrics=None):
        """
        Save model checkpoint.
        
        Args:
            save_path (str): Path to save checkpoint
            metrics (dict): Optional metrics to save
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_accuracy': self.best_accuracy
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['cnn', 'resnext', 'vit', 'ensemble'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--save-dir', type=str, default='trained_models',
                       help='Directory to save models')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes')
    
    args = parser.parse_args()
    
    # Create model
    if args.model == 'cnn':
        model = CNNModel(num_classes=args.num_classes)
    elif args.model == 'resnext':
        model = ResNextModel(num_classes=args.num_classes)
    elif args.model == 'vit':
        model = ViTModel(num_classes=args.num_classes)
    else:
        model = EnsembleDetector(num_classes=args.num_classes)
    
    # Create trainer
    trainer = Trainer(model, device=args.device, lr=args.lr)
    
    print(f"Training {args.model} model on {args.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop (placeholder - replace with actual data loading)
    print("Ready for training. Replace with actual dataset and dataloader.")
    
    # Save model
    save_path = Path(args.save_dir) / f"{args.model}_model.pth"
    trainer.save_checkpoint(save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
