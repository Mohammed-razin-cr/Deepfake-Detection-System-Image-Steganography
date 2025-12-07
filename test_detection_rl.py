"""Test script to verify detection and RL training work correctly."""

import requests
import json
import os
import cv2
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:5000"

def create_test_image(size=(224, 224)):
    """Create a simple test image."""
    img_array = np.random.randint(100, 200, (size[0], size[1], 3), dtype=np.uint8)
    test_img_path = "test_image.jpg"
    cv2.imwrite(test_img_path, img_array)
    return test_img_path

def create_test_video(filename="test_video.mp4", duration=1, fps=10):
    """Create a simple test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (224, 224))
    
    # Create 10 frames
    for i in range(duration * fps):
        frame = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return filename

def test_image_detection():
    """Test image detection."""
    print("\n" + "="*60)
    print("TEST 1: IMAGE DETECTION")
    print("="*60)
    
    # Create test image
    print("Creating test image...")
    test_img = create_test_image()
    
    # Upload and detect
    print("Uploading image for detection...")
    with open(test_img, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/detect/image", files=files)
    
    if response.status_code != 200:
        print(f"FAIL: Detection failed: {response.text}")
        return False
    
    result = response.json()
    print(f"SUCCESS: Image detection completed")
    print(f"  Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
    print(f"  Fake Probability: {result['fake_probability']:.4f}")
    print(f"  Real Probability: {result['real_probability']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    # Clean up
    os.remove(test_img)
    return result

def test_video_detection():
    """Test video detection."""
    print("\n" + "="*60)
    print("TEST 2: VIDEO DETECTION")
    print("="*60)
    
    # Create test video
    print("Creating test video...")
    test_vid = create_test_video()
    
    # Upload and detect
    print("Uploading video for detection...")
    with open(test_vid, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/detect/video", files=files)
    
    if response.status_code != 200:
        print(f"FAIL: Video detection failed: {response.text}")
        return False
    
    result = response.json()
    print(f"SUCCESS: Video detection completed")
    print(f"  Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
    print(f"  Fake Probability: {result['fake_probability']:.4f}")
    print(f"  Real Probability: {result['real_probability']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    # Clean up
    os.remove(test_vid)
    return result

def test_feedback_and_rl():
    """Test feedback submission and RL training."""
    print("\n" + "="*60)
    print("TEST 3: FEEDBACK & RL TRAINING")
    print("="*60)
    
    # Create and detect image
    print("Step 1: Creating and detecting test image...")
    test_img = create_test_image()
    with open(test_img, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/detect/image", files=files)
    
    detection_result = response.json()
    print(f"  Detection complete: {detection_result['is_fake']} (confidence: {detection_result['confidence']:.4f})")
    
    # Submit feedback
    print("\nStep 2: Submitting feedback as 'incorrect'...")
    feedback_data = {
        'feedback_type': 'incorrect',
        'prediction': int(detection_result['prediction']),
        'fake_probability': detection_result['fake_probability'],
        'real_probability': detection_result['real_probability'],
        'uploaded_file': detection_result['uploaded_file'],
        'file_name': 'test_image.jpg',
        'file_type': 'image'
    }
    
    response = requests.post(f"{BASE_URL}/api/feedback", json=feedback_data)
    if response.status_code != 200:
        print(f"FAIL: Feedback submission failed: {response.text}")
        return False
    
    print(f"  Feedback submitted successfully")
    
    # Trigger RL training
    print("\nStep 3: Triggering RL training...")
    response = requests.post(f"{BASE_URL}/api/rl/retrain")
    if response.status_code != 200:
        print(f"FAIL: RL training failed: {response.text}")
        return False
    
    retrain_result = response.json()
    print(f"  RL Training completed")
    print(f"  Trained samples: {retrain_result.get('trained_samples', 0)}")
    print(f"  Improvements: {retrain_result.get('improvements', 0)}")
    print(f"  Stats: {json.dumps(retrain_result.get('rl_stats', {}), indent=4)}")
    
    # Check if training history was created
    print("\nStep 4: Verifying training history...")
    if os.path.exists("results/rl_training_history.json"):
        with open("results/rl_training_history.json", 'r') as f:
            history = json.load(f)
        print(f"  Training history entries: {len(history)}")
        if history:
            print(f"  Latest entry: {json.dumps(history[-1], indent=4)}")
        print("  SUCCESS: Training history exists and updated")
    else:
        print("  WARNING: Training history file not created")
    
    # Clean up
    os.remove(test_img)
    return retrain_result

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Image detection
        img_result = test_image_detection()
        if not img_result:
            print("\n❌ Image detection test FAILED")
            return False
        
        # Test 2: Video detection
        vid_result = test_video_detection()
        if not vid_result:
            print("\n❌ Video detection test FAILED")
            return False
        
        # Test 3: Feedback and RL
        rl_result = test_feedback_and_rl()
        if not rl_result:
            print("\n❌ Feedback & RL test FAILED")
            return False
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ Image detection: PASS")
        print("✓ Video detection: PASS")
        print("✓ Feedback & RL: PASS")
        print("\n✅ ALL TESTS PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
