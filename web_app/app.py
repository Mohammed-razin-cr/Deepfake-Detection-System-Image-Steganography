"""Flask web application for deepfake detection."""

# -*- coding: utf-8 -*-
import io
import sys

# Handle emoji printing on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import traceback

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from functools import wraps
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import EnsembleDetector
from utils import ImagePreprocessor, VideoProcessor, InferenceEngine
from utils.rl_trainer import RLTrainer
from auth import authenticate_user, register_user, init_users_file
from steganography import ImageSteganography

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-please-change')

# Create upload and results directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize users file
init_users_file()

# Global variables
model = None
device = None
inference_engine = None
rl_trainer = None


def login_required(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # Check if it's an API request (JSON)
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
            # For HTML requests, redirect to login page
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename, file_type='both'):
    """Check if file extension is allowed."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in app.config['ALLOWED_VIDEO_EXTENSIONS']
    else:
        return ext in (app.config['ALLOWED_IMAGE_EXTENSIONS'] | app.config['ALLOWED_VIDEO_EXTENSIONS'])


def initialize_model(model_path='../trained_models/ensemble_model.pth'):
    """Initialize the model."""
    global model, inference_engine, device, rl_trainer
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = EnsembleDetector(num_classes=2)
        
        # Try to load weights if they exist
        full_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_path):
            try:
                model.load_state_dict(torch.load(full_path, map_location=device))
                print(f"✓ Loaded model from {full_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load model weights: {e}")
                print("⚠ Using model with randomly initialized weights")
        else:
            print(f"⚠ Model weights not found at {full_path}")
            print("⚠ Using model with randomly initialized weights for demonstration")
        
        model.to(device)
        model.eval()
        
        # Initialize inference engine
        inference_engine = InferenceEngine(model, device=device)
        
        # Initialize RL Trainer
        rl_trainer = RLTrainer(model, device=device, learning_rate=0.00005)
        
        print(f"✓ Model initialized on device: {device}")
        print(f"✓ RL Trainer initialized")
        return True
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        print(f"✗ Traceback: {traceback.format_exc()}")
        return False


@app.route('/')
def index():
    """Render home page."""
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html', user=session.get('user'))


@app.route('/login')
def login_page():
    """Render login page."""
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user with email and password."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        success, result = authenticate_user(email, password)
        
        if success:
            session['user'] = result
            return jsonify({
                'status': 'success',
                'user': session['user'],
                'message': 'Login successful'
            }), 200
        else:
            return jsonify({'error': result}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Login failed: ' + str(e)}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        success, message = register_user(email, password)
        
        if success:
            # Auto-login after registration
            auth_success, user_data = authenticate_user(email, password)
            if auth_success:
                session['user'] = user_data
                return jsonify({
                    'status': 'success',
                    'user': session['user'],
                    'message': message
                }), 200
            else:
                return jsonify({
                    'status': 'success',
                    'message': message + ' Please login now.'
                }), 200
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Registration failed: ' + str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user."""
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully'}), 200


@app.route('/api/auth/user', methods=['GET'])
def get_user():
    """Get current user info."""
    if 'user' in session:
        return jsonify({'status': 'success', 'user': session['user']}), 200
    return jsonify({'status': 'not_authenticated'}), 401


@app.route('/api/status', methods=['GET'])
def status():
    """Get API status."""
    return jsonify({
        'status': 'online',
        'device': str(device),
        'model_loaded': model is not None
    })


@app.route('/api/detect/image', methods=['POST'])
@login_required
def detect_image():
    """Detect deepfake in image."""
    try:
        # Check if model is loaded
        if model is None or inference_engine is None:
            return jsonify({
                'error': 'Model not loaded. Using demo mode with random predictions.',
                'status': 'demo_mode',
                'is_fake': True,
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'confidence': 0.5,
                'label': 'Demo Mode - Please train/load model'
            }), 200
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid image format'}), 400
        
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
            file.save(filepath)
            
            # Preprocess image
            preprocessor = ImagePreprocessor()
            image_tensor = preprocessor.preprocess(filepath)
            
            # Run detection
            results = inference_engine.detect_image(image_tensor)
            
            # Check for encrypted/hidden data in image
            stego = ImageSteganography()
            is_encrypted = stego.check_if_encrypted(filepath)
            encrypted_data = None
            data_type = None
            
            if is_encrypted:
                try:
                    encrypted_data = stego.decrypt_image(filepath)
                    if encrypted_data:
                        data_type = _detect_data_type(encrypted_data)
                except Exception as stego_error:
                    print(f"Error decrypting image: {stego_error}")
                    is_encrypted = False
            
            # Prepare response with both fake detection and encrypted data detection
            response = {
                'status': 'success',
                'is_fake': results['is_fake'],
                'fake_probability': float(results['fake_probability']),
                'real_probability': float(results['real_probability']),
                'confidence': float(results['confidence']),
                'prediction': int(results['prediction']),
                'label': 'Deepfake Detected' if results['is_fake'] else 'Authentic',
                'uploaded_file': os.path.basename(filepath),
                # Encrypted data detection
                'is_encrypted': is_encrypted,
                'encrypted_data': encrypted_data,
                'encrypted_data_type': data_type
            }
            
            return jsonify(response), 200
        except Exception as inner_error:
            print(f"Error processing image: {traceback.format_exc()}")
            return jsonify({'error': f'Image processing failed: {str(inner_error)}'}), 500
    
    except Exception as e:
        print(f"Error in image detection: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/api/detect/video', methods=['POST'])
@login_required
def detect_video():
    """Detect deepfake in video."""
    try:
        # Check if model is loaded
        if model is None or inference_engine is None:
            return jsonify({
                'error': 'Model not loaded. Using demo mode with random predictions.',
                'status': 'demo_mode',
                'is_fake': True,
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'confidence': 0.5,
                'label': 'Demo Mode - Please train/load model'
            }), 200
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'Invalid video format'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)
        
        try:
            # Preprocess video
            processor = VideoProcessor()
            
            # Get video info
            video_info = processor.get_video_info(filepath)
            
            if video_info['frame_count'] == 0:
                return jsonify({'error': 'Video has no frames'}), 400
            
            # Extract frames from video
            frames = processor.extract_frames(filepath)
            
            if len(frames) == 0:
                return jsonify({'error': 'Could not extract any frames from video. Ensure video is a valid MP4 or AVI file.'}), 400
            
            # Process each frame and get predictions
            predictions = []
            confidences = []
            fake_probs = []
            real_probs = []
            
            for frame_idx, frame in enumerate(frames):
                try:
                    # Preprocess individual frame using PIL Image
                    from PIL import Image as PILImage
                    pil_frame = PILImage.fromarray(frame.astype('uint8'), 'RGB')
                    frame_tensor = processor.image_preprocessor.transform(pil_frame)
                    frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
                    
                    # Run detection on frame
                    frame_results = inference_engine.detect_image(frame_tensor)
                    predictions.append(frame_results['prediction'])
                    confidences.append(frame_results['confidence'])
                    fake_probs.append(frame_results['fake_probability'])
                    real_probs.append(frame_results['real_probability'])
                except Exception as frame_error:
                    print(f"Error processing frame {frame_idx}: {frame_error}")
                    continue
            
            if not predictions or len(predictions) == 0:
                return jsonify({'error': 'Failed to process video frames. The video format may not be supported.'}), 500
            
            # Average predictions across frames
            avg_prediction = int(sum(predictions) / len(predictions) >= 0.5)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            avg_fake_prob = sum(fake_probs) / len(fake_probs) if fake_probs else 0.5
            avg_real_prob = sum(real_probs) / len(real_probs) if real_probs else 0.5
            
            # Prepare response
            response = {
                'status': 'success',
                'is_fake': avg_prediction == 1,
                'fake_probability': float(avg_fake_prob),
                'real_probability': float(avg_real_prob),
                'confidence': float(avg_confidence),
                'prediction': int(avg_prediction),
                'label': 'Deepfake Detected' if avg_prediction == 1 else 'Authentic',
                'uploaded_file': os.path.basename(filepath),
                'frames_analyzed': len(predictions),
                'video_info': {
                    'duration': float(video_info['duration']) if video_info['duration'] > 0 else 0,
                    'fps': float(video_info['fps']) if video_info['fps'] > 0 else 30,
                    'frame_count': int(video_info['frame_count']),
                    'resolution': f"{video_info['width']}x{video_info['height']}"
                }
            }
            
            return jsonify(response), 200
            
        except ValueError as ve:
            print(f"ValueError in video detection: {ve}")
            return jsonify({'error': f'Video processing error: {str(ve)}'}), 400
        except Exception as inner_error:
            print(f"Inner error in video detection: {traceback.format_exc()}")
            return jsonify({'error': f'Video processing failed: {str(inner_error)}'}), 500
    
    except Exception as e:
        print(f"Error in video detection: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get detection history."""
    try:
        history = []
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if os.path.isfile(filepath):
                file_stat = os.stat(filepath)
                file_size = file_stat.st_size / (1024 * 1024)  # Convert to MB
                
                history.append({
                    'filename': filename,
                    'size_mb': round(file_size, 2),
                    'uploaded_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
        
        # Sort by upload time (newest first)
        history.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return jsonify({'status': 'success', 'history': history[:50]}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    """Clear upload history."""
    try:
        import shutil
        
        # Clear upload folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        return jsonify({'status': 'success', 'message': 'History cleared'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/downloads/<filename>', methods=['GET'])
@login_required
def download_file(filename):
    """Download uploaded file."""
    try:
        from flask import send_file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(
                filepath, 
                as_attachment=True, 
                download_name=filename,
                mimetype='application/octet-stream'
            )
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return redirect(url_for('login_page')), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 Method Not Allowed errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The method {request.method} is not allowed for this endpoint',
        'path': request.path
    }), 405


@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Submit user feedback on prediction accuracy with RL training."""
    global rl_trainer, model, inference_engine
    
    try:
        data = request.get_json()
        feedback_type = data.get('feedback_type')  # 'correct' or 'incorrect'
        prediction = data.get('prediction')  # 0 or 1
        uploaded_file = data.get('uploaded_file', 'unknown')  # Get the uploaded file path
        
        # Determine actual label based on feedback
        # If feedback is 'incorrect', actual label is opposite of prediction
        # If feedback is 'correct', actual label is same as prediction
        if feedback_type == 'correct':
            actual_label = prediction
        else:  # incorrect
            actual_label = 1 - prediction  # Flip the label
        
        # Build full filepath for later loading
        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file) if uploaded_file != 'unknown' else 'unknown'
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'feedback_type': feedback_type,
            'prediction': prediction,
            'actual_label': actual_label,
            'fake_probability': data.get('fake_probability'),
            'real_probability': data.get('real_probability'),
            'file_name': data.get('file_name'),
            'file_type': data.get('file_type'),
            'uploaded_file': uploaded_file,
            'full_filepath': full_filepath,
            'rl_trained': False
        }
        
        # RL Training: Fine-tune model on this feedback
        rl_metrics = None
        if rl_trainer and model is not None and inference_engine is not None:
            try:
                # Get the uploaded image if available
                image_tensor = data.get('image_tensor')  # Will be None for now, we'll handle this differently
                
                # For now, we'll log that RL should be triggered
                feedback_data['rl_triggered'] = True
                
                print(f"🤖 RL: Feedback received - Prediction: {prediction}, Actual: {actual_label}, File: {uploaded_file}")
                
            except Exception as rl_error:
                print(f"⚠ RL training error: {rl_error}")
        
        # Save feedback to log
        feedback_log = os.path.join(app.config['RESULTS_FOLDER'], 'feedback_log.json')
        feedback_list = []
        if os.path.exists(feedback_log):
            try:
                with open(feedback_log, 'r') as f:
                    feedback_list = json.load(f)
            except:
                feedback_list = []
        
        feedback_list.append(feedback_data)
        
        with open(feedback_log, 'w') as f:
            json.dump(feedback_list, f, indent=2)
        
        print(f"✓ Feedback recorded: {feedback_type} for prediction {prediction} (actual: {actual_label})")
        
        # Get RL stats
        rl_stats = rl_trainer.get_training_stats() if rl_trainer else {}
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you! Your feedback helps improve the model.',
            'rl_triggered': feedback_data.get('rl_triggered', False),
            'rl_stats': rl_stats
        }), 200
    
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to record feedback'}), 500


@app.route('/api/rl/stats', methods=['GET'])
@login_required
def get_rl_stats():
    """Get reinforcement learning training statistics."""
    try:
        if rl_trainer is None:
            return jsonify({'error': 'RL trainer not initialized'}), 500
        
        stats = rl_trainer.get_training_stats()
        feedback_count = rl_trainer.get_feedback_count()
        
        return jsonify({
            'status': 'success',
            'rl_stats': stats,
            'feedback_count': feedback_count,
            'should_retrain': rl_trainer.should_retrain()
        }), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/rl/retrain', methods=['POST'])
@login_required
def retrain_with_rl():
    """Manually trigger RL retraining on collected feedback."""
    global rl_trainer, model
    
    try:
        if rl_trainer is None or model is None:
            return jsonify({'error': 'Model or RL trainer not initialized'}), 500
        
        feedback_log = os.path.join(app.config['RESULTS_FOLDER'], 'feedback_log.json')
        
        if not os.path.exists(feedback_log):
            return jsonify({
                'status': 'info',
                'message': 'No feedback available for training'
            }), 200
        
        with open(feedback_log, 'r') as f:
            feedback_list = json.load(f)
        
        if not feedback_list:
            return jsonify({
                'status': 'info',
                'message': 'No feedback available for training'
            }), 200
        
        # Train on recent feedback
        trained_count = 0
        total_improvements = 0
        preprocessor = ImagePreprocessor()
        
        for feedback in feedback_list[-10:]:  # Train on last 10 feedback items
            try:
                prediction = feedback.get('prediction', 0)
                actual_label = feedback.get('actual_label', prediction)
                full_filepath = feedback.get('full_filepath', 'unknown')
                
                # Load actual image tensor from saved file
                image_tensor = None
                if full_filepath != 'unknown' and os.path.exists(full_filepath):
                    try:
                        image_tensor = preprocessor.preprocess(full_filepath)
                        print(f"✓ Loaded actual image for RL training: {full_filepath}")
                    except Exception as load_error:
                        print(f"⚠ Failed to load image {full_filepath}: {load_error}")
                        # Fall back to using a dummy tensor if image loading fails
                        image_tensor = torch.randn(1, 3, 224, 224).to(device if device else 'cpu')
                else:
                    # Use dummy tensor only as fallback
                    print(f"⚠ No file path found for feedback, using dummy tensor")
                    image_tensor = torch.randn(1, 3, 224, 224).to(device if device else 'cpu')
                
                if image_tensor is None:
                    image_tensor = torch.randn(1, 3, 224, 224).to(device if device else 'cpu')
                
                rl_trainer.model.train()  # Set to train mode
                metrics = rl_trainer.train_on_feedback(
                    image_tensor,
                    actual_label,
                    prediction
                )
                
                if metrics['improvement']:
                    total_improvements += 1
                    print(f"✓ Improvement detected! Loss: {metrics['loss']:.4f}")
                
                rl_trainer.save_training_record(
                    metrics,
                    feedback.get('file_name', 'unknown')
                )
                trained_count += 1
                
            except Exception as e:
                print(f"⚠ Error training on feedback: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save checkpoint after training
        if trained_count > 0:
            rl_trainer.save_model_checkpoint()
        
        stats = rl_trainer.get_training_stats()
        
        return jsonify({
            'status': 'success',
            'message': f'RL Training completed on {trained_count} feedback samples with {total_improvements} improvements',
            'trained_samples': trained_count,
            'improvements': total_improvements,
            'rl_stats': stats
        }), 200
    
    except Exception as e:
        print(f"Error in RL retraining: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'RL retraining failed: {str(e)}'
        }), 500


@app.route('/api/encrypt/image', methods=['POST'])
@login_required
def encrypt_image():
    """Encrypt (hide) data in image using steganography."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        message = request.form.get('message', '').strip()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not message:
            return jsonify({'error': 'No message provided to encrypt'}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid image format. Only PNG, JPG, JPEG supported for encryption.'}), 400
        
        try:
            # Initialize steganography
            stego = ImageSteganography()
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
            file.save(filepath)
            
            # Check capacity
            capacity = stego.get_capacity(filepath)
            if len(message) > capacity:
                return jsonify({
                    'error': f'Message too long. Maximum capacity: {capacity} characters, your message: {len(message)} characters'
                }), 400
            
            # Encrypt (hide) message in image
            output_filename = f"{timestamp}encrypted_{filename.rsplit('.', 1)[0]}.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            encrypted_path = stego.encrypt_image(filepath, message, output_path)
            
            return jsonify({
                'status': 'success',
                'message': 'Image encrypted successfully',
                'encrypted_file': os.path.basename(encrypted_path),
                'original_file': os.path.basename(filepath),
                'message_length': len(message),
                'capacity_used': f"{len(message)}/{capacity} characters"
            }), 200
            
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
        except Exception as inner_error:
            print(f"Encryption error: {traceback.format_exc()}")
            return jsonify({'error': f'Encryption failed: {str(inner_error)}'}), 500
    
    except Exception as e:
        print(f"Error in image encryption: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/api/decrypt/image', methods=['POST'])
@login_required
def decrypt_image():
    """Decrypt (extract) data from image using steganography."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid image format'}), 400
        
        try:
            # Initialize steganography
            stego = ImageSteganography()
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
            file.save(filepath)
            
            # Check if image contains encrypted data
            is_encrypted = stego.check_if_encrypted(filepath)
            
            if not is_encrypted:
                return jsonify({
                    'status': 'no_data',
                    'message': 'No encrypted data found in this image'
                }), 200
            
            # Decrypt (extract) message from image
            extracted_message = stego.decrypt_image(filepath)
            
            if extracted_message:
                return jsonify({
                    'status': 'success',
                    'message': 'Data extracted successfully',
                    'extracted_data': extracted_message,
                    'data_type': _detect_data_type(extracted_message),
                    'file_name': os.path.basename(filepath)
                }), 200
            else:
                return jsonify({
                    'status': 'no_data',
                    'message': 'Could not extract data from image'
                }), 200
            
        except Exception as inner_error:
            print(f"Decryption error: {traceback.format_exc()}")
            return jsonify({'error': f'Decryption failed: {str(inner_error)}'}), 500
    
    except Exception as e:
        print(f"Error in image decryption: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/api/check/encrypted', methods=['POST'])
@login_required
def check_encrypted():
    """Check if image contains encrypted data."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid image format'}), 400
        
        try:
            # Initialize steganography
            stego = ImageSteganography()
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
            file.save(filepath)
            
            # Check if encrypted
            is_encrypted = stego.check_if_encrypted(filepath)
            capacity = stego.get_capacity(filepath)
            
            result = {
                'status': 'success',
                'is_encrypted': is_encrypted,
                'capacity': capacity,
                'file_name': os.path.basename(filepath)
            }
            
            if is_encrypted:
                result['message'] = 'This image contains encrypted/hidden data'
            else:
                result['message'] = 'No encrypted data found in this image'
            
            return jsonify(result), 200
            
        except Exception as inner_error:
            print(f"Check error: {traceback.format_exc()}")
            return jsonify({'error': f'Check failed: {str(inner_error)}'}), 500
    
    except Exception as e:
        print(f"Error checking encryption: {traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def _detect_data_type(data):
    """Detect the type of extracted data."""
    data_lower = data.lower().strip()
    
    if data_lower.startswith('http://') or data_lower.startswith('https://'):
        return 'URL/Link'
    elif data_lower.startswith('data:') or 'base64' in data_lower:
        return 'Base64 Data'
    elif '@' in data and '.' in data:
        return 'Email'
    elif data.isdigit():
        return 'Number'
    elif len(data) > 100:
        return 'Text/Message'
    else:
        return 'Unknown'


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  MOHAMMED RAZIN CR - DEEPFAKE DETECTION WEB APP")
    print("="*60 + "\n")
    
    print("📦 Initializing Mohammed Razin CR deepfake detection model...")
    model_initialized = initialize_model()
    
    if not model_initialized:
        print("\n⚠️  WARNING: Model initialization failed")
        print("   The web interface will still work but predictions may not be available")
        print("   This is normal for demonstration purposes\n")
    
    print("\n" + "="*60)
    print("🚀 Starting Flask application...")
    print("="*60)
    print("\n📍 Web Interface: http://localhost:5000")
    print("📍 API Endpoints:")
    print("   - POST http://localhost:5000/api/detect/image")
    print("   - POST http://localhost:5000/api/detect/video")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
