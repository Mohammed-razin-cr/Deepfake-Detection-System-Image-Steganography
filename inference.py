"""Inference script for deepfake detection."""

import torch
import argparse
from pathlib import Path
import json

from models import EnsembleDetector, ResNextModel, ViTModel, CNNModel
from utils import ImagePreprocessor, VideoProcessor, InferenceEngine


def run_image_detection(image_path, model_path, device='cpu'):
    """
    Run detection on a single image.
    
    Args:
        image_path (str): Path to image
        model_path (str): Path to trained model
        device (str): Device to run inference on
        
    Returns:
        dict: Detection results
    """
    # Load model
    model = EnsembleDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Preprocess image
    preprocessor = ImagePreprocessor()
    image_tensor = preprocessor.preprocess(image_path)
    
    # Run inference
    engine = InferenceEngine(model, device=device)
    results = engine.detect_image(image_tensor)
    
    return results


def run_video_detection(video_path, model_path, device='cpu'):
    """
    Run detection on video.
    
    Args:
        video_path (str): Path to video
        model_path (str): Path to trained model
        device (str): Device to run inference on
        
    Returns:
        dict: Detection results
    """
    # Load model
    model = EnsembleDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Preprocess video
    processor = VideoProcessor()
    video_tensor = processor.preprocess_video(video_path)
    
    # Get video info
    video_info = processor.get_video_info(video_path)
    
    # Run inference
    engine = InferenceEngine(model, device=device)
    results = engine.detect_video(video_tensor)
    results['video_info'] = video_info
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run deepfake detection')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image or video file')
    parser.add_argument('--model', type=str, default='trained_models/ensemble_model.pth',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine if input is image or video
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    
    results = None
    
    if input_path.suffix.lower() in image_extensions:
        print(f"Running image detection on {input_path}")
        results = run_image_detection(str(input_path), args.model, args.device)
    elif input_path.suffix.lower() in video_extensions:
        print(f"Running video detection on {input_path}")
        results = run_video_detection(str(input_path), args.model, args.device)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        return
    
    # Display results
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION RESULTS")
    print("="*50)
    
    if results['is_fake']:
        print("⚠️  DETECTED: LIKELY DEEPFAKE")
    else:
        print("✓ DETECTED: LIKELY AUTHENTIC")
    
    print(f"\nFake Probability: {results['fake_probability']:.4f}")
    print(f"Real Probability: {results['real_probability']:.4f}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    if 'video_info' in results:
        print(f"\nVideo Information:")
        print(f"  Duration: {results['video_info']['duration']:.2f}s")
        print(f"  FPS: {results['video_info']['fps']:.2f}")
        print(f"  Resolution: {results['video_info']['width']}x{results['video_info']['height']}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {
                'is_fake': results['is_fake'],
                'fake_probability': float(results['fake_probability']),
                'real_probability': float(results['real_probability']),
                'confidence': float(results['confidence']),
                'prediction': int(results['prediction'])
            }
            json.dump(json_results, f, indent=4)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
