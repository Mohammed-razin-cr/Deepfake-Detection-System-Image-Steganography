"""Generate demo dataset for testing deepfake detection."""

import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def create_synthetic_images(output_dir='data', num_real=10, num_fake=10):
    """Create synthetic test images."""
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/real").mkdir(exist_ok=True)
    Path(f"{output_dir}/fake").mkdir(exist_ok=True)
    
    # Create real images (natural patterns)
    print("Generating real images...")
    for i in range(num_real):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add gradient backgrounds
        for j in range(224):
            img[j, :] = [int(j * 255/224), int((224-j) * 255/224), 128]
        
        # Add some noise
        noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
        
        # Add some shapes
        cv2.circle(img, (112 + i*5, 112 + i*3), 20 + i*2, (255, 200, 100), -1)
        cv2.rectangle(img, (50, 50), (150 + i*3, 150 + i*3), (100, 200, 255), 2)
        
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/real/real_{i:03d}.jpg")
    
    # Create fake images (artifacts and distortions)
    print("Generating fake images...")
    for i in range(num_fake):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add unnatural patterns - strong artifacts
        for j in range(224):
            img[j, :] = [255 - int(j * 255/224), int(j * 255/224), 0]
        
        # Add more aggressive noise
        noise = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
        
        # Add distortion patterns typical of deepfakes
        for x in range(50, 174, 20):
            for y in range(50, 174, 20):
                cv2.circle(img, (x + i*2, y + i*2), 10 + i, (50, 50, 50), -1)
        
        # Add checkerboard pattern (common in generated images)
        for x in range(0, 224, 16):
            for y in range(0, 224, 16):
                if (x // 16 + y // 16) % 2 == 0:
                    img[y:y+16, x:x+16] = [200, 50, 50]
        
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/fake/fake_{i:03d}.jpg")
    
    print(f"✅ Generated {num_real} real and {num_fake} fake images in {output_dir}/")

def create_labels_file(output_dir='data', labels_file='labels.txt'):
    """Create labels file."""
    with open(labels_file, 'w') as f:
        # Real images (label 0)
        real_dir = Path(f"{output_dir}/real")
        if real_dir.exists():
            for img_file in sorted(real_dir.glob("*.jpg")):
                f.write(f"real/{img_file.name} 0\n")
        
        # Fake images (label 1)
        fake_dir = Path(f"{output_dir}/fake")
        if fake_dir.exists():
            for img_file in sorted(fake_dir.glob("*.jpg")):
                f.write(f"fake/{img_file.name} 1\n")
    
    print(f"✅ Created labels file: {labels_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate demo dataset')
    parser.add_argument('--num-real', type=int, default=20, help='Number of real images')
    parser.add_argument('--num-fake', type=int, default=20, help='Number of fake images')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    create_synthetic_images(args.output_dir, args.num_real, args.num_fake)
    create_labels_file(args.output_dir, f'{args.output_dir}/labels.txt')
    
    print("\n📊 Demo dataset ready for training!")
    print(f"Run: python train.py --data-dir {args.output_dir} --labels-file {args.output_dir}/labels.txt")
