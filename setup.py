"""
Setup script to initialize the deepfake detection project.
Run this script to download pre-trained models and set up the environment.
"""

import os
import sys
import urllib.request
from pathlib import Path
import json


def create_directories():
    """Create necessary project directories."""
    directories = [
        'trained_models',
        'data/train/fake',
        'data/train/real',
        'data/test/fake',
        'data/test/real',
        'web_app/uploads',
        'web_app/results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_config():
    """Create configuration file."""
    config = {
        'model_config': {
            'num_classes': 2,
            'input_size': 224,
            'sequence_length': 16,
            'fps': 30
        },
        'training_config': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'device': 'cuda'
        },
        'preprocessing_config': {
            'image_size': [224, 224],
            'normalize': True,
            'augmentation': True
        },
        'deployment_config': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'max_file_size': 500  # MB
        }
    }
    
    config_path = 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Created configuration file: {config_path}")


def create_env_file():
    """Create .env file with default settings."""
    env_content = """# Deepfake Detection System Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=trained_models/ensemble_model.pth
DEVICE=cuda

# Upload Configuration
UPLOAD_FOLDER=web_app/uploads
RESULTS_FOLDER=web_app/results
MAX_FILE_SIZE=524288000

# Database
DATABASE_URL=sqlite:///deepfake_detection.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security
ALLOWED_EXTENSIONS=jpg,jpeg,png,mp4,avi,mov,mkv
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✓ Created .env configuration file")


def download_sample_model():
    """Download sample pre-trained model (placeholder)."""
    print("\nNote: Pre-trained models need to be downloaded separately:")
    print("1. Visit: https://github.com/selimsef/dfdc_deepfake_challenge")
    print("2. Download trained weights")
    print("3. Place in: trained_models/ directory")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Download/prepare training data:")
    print("   - Place images in data/train/fake/ and data/train/real/")
    print("   - Organize test data similarly")
    print("\n3. Train the model (optional):")
    print("   python train.py --model ensemble --epochs 50")
    print("\n4. Start the web application:")
    print("   python web_app/app.py")
    print("\n5. Open browser and navigate to:")
    print("   http://localhost:5000")
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("Deepfake Detection System - Setup Script")
    print("="*60)
    
    try:
        print("\n1. Creating project directories...")
        create_directories()
        
        print("\n2. Creating configuration files...")
        create_config()
        
        print("\n3. Creating environment configuration...")
        create_env_file()
        
        print("\n4. Model download information...")
        download_sample_model()
        
        print_next_steps()
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
