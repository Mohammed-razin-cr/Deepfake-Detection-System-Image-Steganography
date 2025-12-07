# API Documentation

## Overview

The Deepfake Detection API provides endpoints for analyzing images and videos for deepfake content.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, the API doesn't require authentication. For production, implement:
- API key authentication
- JWT tokens
- OAuth 2.0

## Endpoints

### 1. Image Detection

**Endpoint:** `POST /api/detect/image`

**Description:** Detect deepfake in an image file

**Request:**
```bash
curl -X POST http://localhost:5000/api/detect/image \
  -F "file=@image.jpg"
```

**Request Parameters:**
- `file` (required): Image file (JPG, PNG, BMP, GIF)
- `max_size`: 500MB

**Response (200):**
```json
{
  "status": "success",
  "is_fake": false,
  "fake_probability": 0.15,
  "real_probability": 0.85,
  "confidence": 0.92,
  "prediction": 0,
  "label": "Authentic",
  "uploaded_file": "timestamp_image.jpg"
}
```

**Response (400 - Bad Request):**
```json
{
  "error": "Invalid image format"
}
```

### 2. Video Detection

**Endpoint:** `POST /api/detect/video`

**Description:** Detect deepfake in a video file

**Request:**
```bash
curl -X POST http://localhost:5000/api/detect/video \
  -F "file=@video.mp4"
```

**Request Parameters:**
- `file` (required): Video file (MP4, AVI, MOV, MKV, FLV)
- `max_size`: 500MB

**Response (200):**
```json
{
  "status": "success",
  "is_fake": true,
  "fake_probability": 0.87,
  "real_probability": 0.13,
  "confidence": 0.91,
  "prediction": 1,
  "label": "Deepfake Detected",
  "uploaded_file": "timestamp_video.mp4",
  "video_info": {
    "duration": 15.5,
    "fps": 30.0,
    "frame_count": 465,
    "resolution": "1920x1080"
  }
}
```

### 3. System Status

**Endpoint:** `GET /api/status`

**Description:** Get API and model status

**Response:**
```json
{
  "status": "online",
  "device": "cuda",
  "model_loaded": true
}
```

### 4. Detection History

**Endpoint:** `GET /api/history`

**Description:** Get recent detection history

**Response:**
```json
{
  "status": "success",
  "history": [
    {
      "filename": "20240101_120000_image.jpg",
      "size_mb": 2.5,
      "uploaded_at": "2024-01-01T12:00:00"
    },
    {
      "filename": "20240101_110000_video.mp4",
      "size_mb": 45.3,
      "uploaded_at": "2024-01-01T11:00:00"
    }
  ]
}
```

### 5. Clear History

**Endpoint:** `POST /api/clear-history`

**Description:** Delete all uploaded files

**Response:**
```json
{
  "status": "success",
  "message": "History cleared"
}
```

### 6. Download File

**Endpoint:** `GET /downloads/<filename>`

**Description:** Download uploaded file

**Response:** File download

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 413 | File Too Large |
| 404 | Not Found |
| 500 | Internal Server Error |

## Error Handling

All errors return JSON with error message:

```json
{
  "error": "Error description"
}
```

Common errors:
- `No file provided` - Missing file in request
- `No file selected` - Empty file selected
- `Invalid image format` - Unsupported file type
- `Invalid video format` - Unsupported video format
- `File too large` - Exceeds 500MB limit

## Rate Limiting

For production deployment, implement rate limiting:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/detect/image', methods=['POST'])
@limiter.limit("5 per minute")
def detect_image():
    # Implementation
    pass
```

## Caching

Implement caching for identical detections:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_detection_result(file_hash):
    # Return cached result if exists
    pass
```

## Webhooks

Implement async webhooks for long-running tasks:

```python
@app.route('/api/detect/video/async', methods=['POST'])
def detect_video_async():
    # Queue task
    task_id = queue_detection_task(file)
    
    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'callback_url': request.json.get('callback_url')
    })

@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    # Return task status
    pass
```

## SDK Examples

### Python

```python
import requests

def detect_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:5000/api/detect/image',
            files=files
        )
    return response.json()

# Usage
result = detect_image('image.jpg')
print(f"Is Fake: {result['is_fake']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript

```javascript
async function detectImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:5000/api/detect/image', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Usage
const file = document.getElementById('file-input').files[0];
const result = await detectImage(file);
console.log(`Is Fake: ${result.is_fake}`);
```

### cURL

```bash
# Detect image
curl -X POST http://localhost:5000/api/detect/image \
  -F "file=@image.jpg"

# Detect video
curl -X POST http://localhost:5000/api/detect/video \
  -F "file=@video.mp4"

# Get status
curl http://localhost:5000/api/status

# Get history
curl http://localhost:5000/api/history
```

## Performance

- Image detection: ~1-3 seconds
- Video detection: ~5-15 seconds
- Batch processing: Can handle 100+ images/hour

## Versioning

API version: v1

Future versions planned:
- v2: Batch processing API
- v3: Streaming/WebSocket support
- v4: Advanced filtering options

## Support

For API issues or feature requests, create an issue on GitHub or contact support.
