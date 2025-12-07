"""Image and video preprocessing utilities."""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os


class ImagePreprocessor:
    """Preprocessing pipeline for images."""
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialize image preprocessor.
        
        Args:
            target_size (tuple): Target image size (height, width)
            normalize (bool): Whether to normalize using ImageNet stats
        """
        self.target_size = target_size
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) if normalize else transforms.ToTensor()
        ])
        
    def preprocess(self, image_path):
        """
        Preprocess image from file path.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def preprocess_array(self, image_array):
        """
        Preprocess image from numpy array.
        
        Args:
            image_array (np.ndarray): Image array (H, W, C) with values 0-255
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        else:
            image = image_array
        
        return self.transform(image).unsqueeze(0)
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess batch of images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        tensors = [self.preprocess(path).squeeze(0) for path in image_paths]
        return torch.stack(tensors)


class VideoProcessor:
    """Processing pipeline for video files."""
    
    def __init__(self, target_size=(224, 224), fps=30, frame_count=16):
        """
        Initialize video processor.
        
        Args:
            target_size (tuple): Target frame size
            fps (int): Frames per second to extract
            frame_count (int): Number of frames to extract
        """
        self.target_size = target_size
        self.fps = fps
        self.frame_count = frame_count
        self.image_preprocessor = ImagePreprocessor(target_size)
        
    def extract_frames(self, video_path):
        """
        Extract frames from video.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            np.ndarray: Array of frames (num_frames, H, W, 3)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)
        
        frames = []
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in frame_indices:
                # Resize frame
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            current_frame += 1
        
        cap.release()
        
        return np.array(frames)
    
    def preprocess_video(self, video_path):
        """
        Preprocess video for model inference.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            torch.Tensor: Preprocessed frames (1, seq_length, 3, H, W)
        """
        frames = self.extract_frames(video_path)
        
        # Ensure we have exactly frame_count frames
        if len(frames) < self.frame_count:
            # Pad with last frame
            padding = self.frame_count - len(frames)
            frames = np.vstack([frames, np.tile(frames[-1:], (padding, 1, 1, 1))])
        elif len(frames) > self.frame_count:
            frames = frames[:self.frame_count]
        
        # Convert to tensor
        tensors = []
        for frame in frames:
            tensor = self.image_preprocessor.preprocess_array(frame).squeeze(0)
            tensors.append(tensor)
        
        video_tensor = torch.stack(tensors)
        return video_tensor.unsqueeze(0)  # Add batch dimension
    
    def get_video_info(self, video_path):
        """
        Get video information.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
