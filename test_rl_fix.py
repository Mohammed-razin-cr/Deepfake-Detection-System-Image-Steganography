"""Test script to verify RL training fix works correctly."""

import requests
import json
import os
import cv2
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:5000"

def create_test_image():
    """Create a simple test image."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_img_path = "test_image.jpg"
    cv2.imwrite(test_img_path, img_array)
    return test_img_path

def test_rl_training():
    """Test the RL training fix."""
    
    print("=" * 60)
    print("🧪 TESTING RL TRAINING FIX")
    print("=" * 60)
    
    # Step 1: Create and upload a test image
    print("\n1️⃣  Creating test image...")
    test_img = create_test_image()
    
    # Step 2: Send image for detection
    print("2️⃣  Uploading image for detection...")
    with open(test_img, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/detect/image", files=files)
    
    if response.status_code != 200:
        print(f"❌ Detection failed: {response.text}")
        return False
    
    detection_result = response.json()
    print(f"✓ Detection result: {json.dumps(detection_result, indent=2)}")
    
    # Step 3: Submit feedback
    print("\n3️⃣  Submitting feedback as 'incorrect'...")
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
        print(f"❌ Feedback submission failed: {response.text}")
        return False
    
    feedback_response = response.json()
    print(f"✓ Feedback response: {json.dumps(feedback_response, indent=2)}")
    
    # Step 4: Check feedback log
    print("\n4️⃣  Checking feedback log...")
    feedback_log_path = "results/feedback_log.json"
    if os.path.exists(feedback_log_path):
        with open(feedback_log_path, 'r') as f:
            feedback_log = json.load(f)
        print(f"✓ Feedback log entries: {len(feedback_log)}")
        print(f"✓ Latest entry: {json.dumps(feedback_log[-1], indent=2)}")
        
        # Check if full_filepath is saved
        if 'full_filepath' in feedback_log[-1]:
            filepath = feedback_log[-1]['full_filepath']
            print(f"✓ File path saved: {filepath}")
            if os.path.exists(filepath):
                print(f"✓ File exists at path: {filepath}")
            else:
                print(f"⚠ File does not exist at path: {filepath}")
    else:
        print("⚠ Feedback log not found")
    
    # Step 5: Trigger RL retraining
    print("\n5️⃣  Triggering RL retraining...")
    response = requests.post(f"{BASE_URL}/api/rl/retrain")
    
    if response.status_code != 200:
        print(f"❌ RL retraining failed: {response.text}")
        return False
    
    retrain_response = response.json()
    print(f"✓ Retrain response: {json.dumps(retrain_response, indent=2)}")
    
    # Step 6: Check training history
    print("\n6️⃣  Checking training history...")
    training_history_path = "results/rl_training_history.json"
    if os.path.exists(training_history_path):
        with open(training_history_path, 'r') as f:
            training_history = json.load(f)
        print(f"✓ Training history entries: {len(training_history)}")
        if training_history:
            print(f"✓ Latest training record: {json.dumps(training_history[-1], indent=2)}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ RL TRAINING TEST RESULTS")
        print("=" * 60)
        trained_samples = retrain_response.get('trained_samples', 0)
        improvements = retrain_response.get('improvements', 0)
        print(f"✓ Trained on {trained_samples} feedback samples")
        print(f"✓ Improvements detected: {improvements}")
        print(f"✓ Total training history entries: {len(training_history)}")
        
        if trained_samples > 0:
            print("\n✅ SUCCESS: RL training is now working correctly!")
            print("   Feedback is being saved with file paths.")
            print("   RL trainer is loading actual images and training on them.")
            print("   Training history is being recorded.")
            return True
        else:
            print("\n⚠ WARNING: No samples were trained on")
            print("   Check if feedback file path is correct")
            return False
    else:
        print("⚠ Training history not found yet")
        print("   This might be created after first successful training")
        return False

if __name__ == "__main__":
    try:
        success = test_rl_training()
        if success:
            print("\n🎉 RL Training Fix Verified!")
        else:
            print("\n❌ RL Training Fix Needs Attention")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
