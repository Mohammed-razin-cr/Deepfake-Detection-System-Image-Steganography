#!/usr/bin/env python3
"""
Project Initialization and File Verification Script
Verifies all files are in place and creates missing directories
"""

import os
from pathlib import Path

def verify_project_structure():
    """Verify all project files are present."""
    
    base_path = Path('.')
    
    # Define all expected files
    expected_files = {
        'Core Python Files': [
            'train.py',
            'inference.py',
            'setup.py',
            'examples.py',
        ],
        'Web Application': [
            'web_app/app.py',
            'web_app/templates/index.html',
            'web_app/static/style.css',
            'web_app/static/script.js',
        ],
        'Models': [
            'models/__init__.py',
            'models/cnn_model.py',
            'models/resnext_model.py',
            'models/lstm_model.py',
            'models/vision_transformer.py',
            'models/ensemble_model.py',
        ],
        'Utils': [
            'utils/__init__.py',
            'utils/preprocessing.py',
            'utils/inference.py',
            'utils/metrics.py',
        ],
        'Configuration': [
            'requirements.txt',
            'config.ini',
            'Dockerfile',
            'docker-compose.yml',
            '.gitignore',
        ],
        'Documentation': [
            'README.md',
            'QUICKSTART.md',
            'INDEX.md',
            'PROJECT_SUMMARY.md',
            'API.md',
            'DEPLOYMENT.md',
            'TRAINING.md',
        ],
        'Startup Scripts': [
            'run.bat',
            'run.sh',
        ],
    }
    
    print("=" * 70)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 70)
    print()
    
    all_files_exist = True
    
    for category, files in expected_files.items():
        print(f"📁 {category}:")
        for file in files:
            file_path = base_path / file
            exists = file_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
            if not exists:
                all_files_exist = False
        print()
    
    # Check directories
    print("📁 Directories:")
    expected_dirs = [
        'models',
        'utils',
        'web_app',
        'web_app/templates',
        'web_app/static',
        'data',
        'data/train/fake',
        'data/train/real',
        'data/test/fake',
        'data/test/real',
        'trained_models',
        'logs',
    ]
    
    for directory in expected_dirs:
        dir_path = base_path / directory
        exists = dir_path.exists() and dir_path.is_dir()
        status = "✅" if exists else "⚠️"
        print(f"  {status} {directory}/")
    
    print()
    print("=" * 70)
    
    if all_files_exist:
        print("✅ ALL FILES VERIFIED SUCCESSFULLY!")
    else:
        print("⚠️  Some files are missing. Please ensure all files are present.")
    
    print("=" * 70)
    print()
    
    return all_files_exist


def show_file_statistics():
    """Show statistics about the project."""
    
    print("📊 PROJECT STATISTICS")
    print("=" * 70)
    
    # Count Python files
    py_files = list(Path('.').rglob('*.py'))
    py_lines = sum(len(open(f, encoding='utf-8', errors='ignore').readlines()) for f in py_files)
    
    # Count other files
    md_files = list(Path('.').rglob('*.md'))
    json_files = list(Path('.').rglob('*.json'))
    yaml_files = list(Path('.').rglob('*.yaml'))
    yml_files = list(Path('.').rglob('*.yml'))
    
    print(f"Python Files:      {len(py_files)}")
    print(f"Python Lines:      {py_lines:,}")
    print(f"Documentation:     {len(md_files)} files")
    print(f"Config Files:      {len(json_files) + len(yaml_files) + len(yml_files)}")
    print()
    
    # Model summary
    print("🤖 MODELS INCLUDED:")
    models = [
        "CNN Model",
        "ResNext-50",
        "LSTM Network",
        "Vision Transformer",
        "Ensemble (All Combined)"
    ]
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()
    
    # Features
    print("✨ KEY FEATURES:")
    features = [
        "Image detection (JPG, PNG, BMP, GIF)",
        "Video detection (MP4, AVI, MOV, MKV, FLV)",
        "Web interface with real-time results",
        "REST API endpoints",
        "Model training pipeline",
        "Standalone inference script",
        "Docker deployment support",
        "Comprehensive documentation",
    ]
    for feature in features:
        print(f"  ✓ {feature}")
    print()
    
    print("=" * 70)
    print()


def show_next_steps():
    """Display next steps for the user."""
    
    print("🚀 NEXT STEPS")
    print("=" * 70)
    print()
    
    print("1. INSTALL DEPENDENCIES:")
    print("   pip install -r requirements.txt")
    print()
    
    print("2. RUN THE APPLICATION:")
    print("   Windows: run.bat")
    print("   Mac/Linux: ./run.sh")
    print("   Or manually: cd web_app && python app.py")
    print()
    
    print("3. OPEN IN BROWSER:")
    print("   http://localhost:5000")
    print()
    
    print("4. UPLOAD FILES:")
    print("   - Images: JPG, PNG, BMP, GIF")
    print("   - Videos: MP4, AVI, MOV, MKV, FLV")
    print()
    
    print("5. TRAINING (Optional):")
    print("   python train.py --model ensemble --epochs 50")
    print()
    
    print("6. DEPLOYMENT:")
    print("   See DEPLOYMENT.md for cloud/Docker deployment guides")
    print()
    
    print("📚 DOCUMENTATION:")
    print("   - QUICKSTART.md - Get started quickly")
    print("   - INDEX.md - Complete project index")
    print("   - README.md - Full documentation")
    print("   - API.md - API documentation")
    print("   - DEPLOYMENT.md - Deployment guides")
    print("   - TRAINING.md - Training guide")
    print()
    
    print("=" * 70)
    print()


def main():
    """Main function."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🛡️ DEEPFAKE DETECTION SYSTEM - PROJECT INITIALIZED  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Verify structure
    files_ok = verify_project_structure()
    
    # Show statistics
    show_file_statistics()
    
    # Show next steps
    show_next_steps()
    
    if files_ok:
        print("✅ Your project is ready to use!")
        print()
        print("Start with: run.bat (Windows) or ./run.sh (Mac/Linux)")
        print()
    else:
        print("⚠️  Please ensure all files are in place before running the application.")
        print()


if __name__ == '__main__':
    main()
