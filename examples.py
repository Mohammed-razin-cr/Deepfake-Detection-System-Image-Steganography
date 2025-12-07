"""
Example usage of the deepfake detection system.
"""

import torch
from pathlib import Path
from models import EnsembleDetector, ResNextModel, ViTModel
from utils import ImagePreprocessor, VideoProcessor, InferenceEngine


def example_image_detection():
    """Example: Detect deepfake in an image."""
    print("Example 1: Image Detection")
    print("-" * 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnsembleDetector(num_classes=2)
    model.to(device)
    
    # Create inference engine
    inference_engine = InferenceEngine(model, device=device)
    
    # Preprocess image
    preprocessor = ImagePreprocessor()
    image_path = "path/to/your/image.jpg"  # Replace with actual path
    
    if Path(image_path).exists():
        image_tensor = preprocessor.preprocess(image_path)
        
        # Run detection
        results = inference_engine.detect_image(image_tensor)
        
        print(f"Is Fake: {results['is_fake']}")
        print(f"Fake Probability: {results['fake_probability']:.4f}")
        print(f"Real Probability: {results['real_probability']:.4f}")
        print(f"Confidence: {results['confidence']:.4f}")
    else:
        print(f"Image not found: {image_path}")
    
    print()


def example_video_detection():
    """Example: Detect deepfake in a video."""
    print("Example 2: Video Detection")
    print("-" * 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnsembleDetector(num_classes=2)
    model.to(device)
    
    # Create inference engine
    inference_engine = InferenceEngine(model, device=device)
    
    # Process video
    processor = VideoProcessor()
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    if Path(video_path).exists():
        # Get video information
        video_info = processor.get_video_info(video_path)
        print(f"Video Duration: {video_info['duration']:.2f}s")
        print(f"Video FPS: {video_info['fps']:.2f}")
        print(f"Video Resolution: {video_info['width']}x{video_info['height']}")
        
        # Preprocess video
        video_tensor = processor.preprocess_video(video_path)
        
        # Run detection
        results = inference_engine.detect_video(video_tensor)
        
        print(f"\nIs Fake: {results['is_fake']}")
        print(f"Fake Probability: {results['fake_probability']:.4f}")
        print(f"Real Probability: {results['real_probability']:.4f}")
        print(f"Confidence: {results['confidence']:.4f}")
    else:
        print(f"Video not found: {video_path}")
    
    print()


def example_batch_detection():
    """Example: Detect deepfakes in batch of images."""
    print("Example 3: Batch Image Detection")
    print("-" * 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnsembleDetector(num_classes=2)
    model.to(device)
    
    # Create inference engine
    inference_engine = InferenceEngine(model, device=device)
    
    # Preprocess batch of images
    preprocessor = ImagePreprocessor()
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    # Filter existing paths
    existing_paths = [p for p in image_paths if Path(p).exists()]
    
    if existing_paths:
        batch_tensor = preprocessor.preprocess_batch(existing_paths)
        
        # Run batch detection
        results = inference_engine.batch_detect(batch_tensor)
        
        for i, (path, result) in enumerate(zip(existing_paths, results)):
            print(f"\nImage {i+1}: {path}")
            print(f"  Is Fake: {result['is_fake']}")
            print(f"  Confidence: {result['confidence']:.4f}")
    else:
        print("No existing image files found")
    
    print()


def example_model_comparison():
    """Example: Compare different models."""
    print("Example 4: Model Comparison")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize different models
    models = {
        'ResNext-50': ResNextModel(num_classes=2),
        'Vision Transformer': ViTModel(num_classes=2),
        'Ensemble': EnsembleDetector(num_classes=2)
    }
    
    print(f"Device: {device}")
    print(f"\nModel Sizes:")
    for name, model in models.items():
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name}:")
        print(f"    Total Parameters: {total_params:,}")
        print(f"    Trainable Parameters: {trainable_params:,}")
    
    print()


def example_feature_extraction():
    """Example: Extract features for analysis."""
    print("Example 5: Feature Extraction")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNextModel(num_classes=2)
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.get_features(dummy_input)
    
    print(f"Feature Shape: {features.shape}")
    print(f"Feature Dimension: {features.shape[1]}")
    print(f"Feature Range: [{features.min():.4f}, {features.max():.4f}]")
    print()


def main():
    """Run all examples."""
    print("="*60)
    print("Deepfake Detection System - Usage Examples")
    print("="*60)
    print()
    
    try:
        example_image_detection()
        example_video_detection()
        example_batch_detection()
        example_model_comparison()
        example_feature_extraction()
        
        print("="*60)
        print("Examples completed!")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
