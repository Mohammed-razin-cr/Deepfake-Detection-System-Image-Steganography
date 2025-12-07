# Deepfake Detection - Deployment Guide

## Quick Start

### Option 1: Local Deployment (Development)

#### 1. Prerequisites
- Python 3.10+
- pip package manager
- 8GB+ RAM
- CUDA 11.8+ (optional, for GPU support)

#### 2. Setup

```bash
# Clone or navigate to the project
cd deepfake_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py

# Start web application
cd web_app
python app.py
```

Access the application at `http://localhost:5000`

### Option 2: Docker Deployment (Recommended)

#### 1. Build Docker Image

```bash
# Build the image
docker build -t deepfake-detection:latest .

# OR use docker-compose
docker-compose build
```

#### 2. Run Container

```bash
# Run with docker
docker run -p 5000:5000 \
  -v $(pwd)/trained_models:/app/trained_models \
  -v $(pwd)/web_app/uploads:/app/web_app/uploads \
  deepfake-detection:latest

# OR use docker-compose
docker-compose up
```

#### 3. Access Application

Open browser and navigate to `http://localhost:5000`

### Option 3: Cloud Deployment

#### AWS Deployment (EC2 + Docker)

```bash
# 1. Launch EC2 instance (Ubuntu 20.04)
# 2. Install Docker
sudo apt-get update
sudo apt-get install docker.io -y

# 3. Clone repository
git clone <repo-url>
cd deepfake_detection

# 4. Build and run
docker-compose up -d

# 5. Configure security group to allow port 5000
```

#### AWS Lambda (API-only)

```bash
# Package model for serverless
pip install zappa

# Deploy
zappa init
zappa deploy production
```

#### Google Cloud Run

```bash
# Build and push to GCP Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/deepfake-detection

# Deploy to Cloud Run
gcloud run deploy deepfake-detection \
  --image gcr.io/PROJECT-ID/deepfake-detection \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 120
```

#### Azure Container Instances

```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name deepfake-detection \
  --image deepfake-detection:latest \
  --ports 5000 \
  --environment-variables FLASK_ENV=production
```

### Option 4: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepfake-detection
  template:
    metadata:
      labels:
        app: deepfake-detection
    spec:
      containers:
      - name: app
        image: deepfake-detection:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: models
          mountPath: /app/trained_models
        - name: uploads
          mountPath: /app/web_app/uploads
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: uploads
        persistentVolumeClaim:
          claimName: uploads-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: deepfake-detection-service
spec:
  selector:
    app: deepfake-detection
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

Deploy with:
```bash
kubectl apply -f deployment.yaml
```

## Production Configuration

### Environment Variables

```bash
# Flask
FLASK_ENV=production
SECRET_KEY=your-secure-random-key

# Model
MODEL_PATH=/app/trained_models/ensemble_model.pth
DEVICE=cuda

# Server
WORKERS=4
TIMEOUT=120

# Uploads
MAX_FILE_SIZE=524288000  # 500MB
UPLOAD_FOLDER=/app/web_app/uploads
```

### Performance Optimization

1. **GPU Support**
   ```bash
   # Install CUDA runtime
   docker build -f Dockerfile.cuda -t deepfake-detection:cuda .
   
   # Run with GPU
   docker run --gpus all -p 5000:5000 deepfake-detection:cuda
   ```

2. **Load Balancing**
   - Use Nginx for reverse proxy
   - Configure multiple worker processes
   - Implement request queuing

3. **Caching**
   - Enable Redis for session caching
   - Cache model predictions
   - Use CDN for static assets

### Monitoring & Logging

```bash
# Docker logs
docker logs -f deepfake-detection

# Prometheus metrics
pip install prometheus-client

# ELK Stack integration
# Configure in flask app
```

### Security Best Practices

1. **API Security**
   - Use HTTPS/TLS
   - Implement rate limiting
   - Add authentication/authorization
   - Validate all inputs

2. **File Security**
   - Scan uploaded files for malware
   - Sanitize filenames
   - Set proper permissions
   - Implement file cleanup

3. **Model Security**
   - Verify model integrity
   - Encrypt model weights
   - Monitor for attacks

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or input size
   # Use CPU mode
   export DEVICE=cpu
   ```

2. **Slow Detection**
   ```bash
   # Enable GPU
   export DEVICE=cuda
   
   # Optimize model
   # Use quantization
   ```

3. **File Upload Issues**
   ```bash
   # Check permissions
   chmod 755 web_app/uploads
   
   # Increase file size limit
   export MAX_FILE_SIZE=1000000000
   ```

4. **Port Already in Use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   
   # Kill process or use different port
   export API_PORT=8000
   ```

## Maintenance

### Regular Tasks

- Monitor disk space for uploads
- Clean up old uploaded files
- Update dependencies
- Backup models and data
- Monitor API performance
- Review security logs

### Update Procedure

```bash
# Pull latest code
git pull origin main

# Install new dependencies
pip install -r requirements.txt

# Rebuild Docker image
docker build -t deepfake-detection:latest .

# Restart services
docker-compose restart
```

## Support

For deployment issues:
1. Check logs: `docker logs <container-id>`
2. Verify configuration
3. Test with sample files
4. Check system resources
5. Review documentation

## Additional Resources

- [Docker Documentation](https://docs.docker.com)
- [Flask Deployment](https://flask.palletsprojects.com/en/2.3.x/deployment/)
- [PyTorch Serving](https://pytorch.org/serve/)
- [Kubernetes Basics](https://kubernetes.io/docs/setup/)
