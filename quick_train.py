"""Quick training script for deepfake detection model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import EnsembleDetector
from utils import ImagePreprocessor


class QuickDeepfakeDataset(Dataset):
    """Quick dataset for deepfake training."""
    
    def __init__(self, data_dir, labels_file):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.preprocessor = ImagePreprocessor()
        
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
        img_path, label = self.samples[idx]
        full_path = self.data_dir / img_path
        
        try:
            image = Image.open(full_path).convert('RGB')
            tensor = self.preprocessor.transform(image)
            return tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return random tensor if error
            return torch.randn(3, 224, 224), torch.tensor(label, dtype=torch.long)


def train_model(model, train_loader, val_loader, epochs, device, lr=0.001):
    """Train the model."""
    # Use label smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss / train_total, 'acc': train_correct / train_total})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': val_loss / val_total, 'acc': val_correct / val_total})
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_correct/val_total:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--labels-file', type=str, default='data/labels.txt', help='Labels file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--save-dir', type=str, default='trained_models', help='Save directory')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = QuickDeepfakeDataset(args.data_dir, args.labels_file)
    print(f"Loaded {len(dataset)} images")
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating ensemble model...")
    model = EnsembleDetector(num_classes=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*60)
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs with learning rate {args.lr}...\n")
    train_model(model, train_loader, val_loader, args.epochs, device, lr=args.lr)
    
    # Save model
    save_path = Path(args.save_dir) / "ensemble_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")
    print(f"Model size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()
