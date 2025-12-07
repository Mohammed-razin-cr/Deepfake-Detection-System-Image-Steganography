# Model Training Guide

## Dataset Preparation

### Expected Data Format

```
data/
├── train/
│   ├── fake/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── real/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── fake/
    │   └── ...
    └── real/
        └── ...
```

### Popular Deepfake Datasets

1. **DFDC (Deepfake Detection Challenge)**
   - 100,000+ videos
   - Download: https://www.kaggle.com/c/deepfake-detection-challenge

2. **FaceForensics++**
   - 1000 original videos
   - Multiple manipulation techniques
   - Download: https://github.com/ondyari/FaceForensics

3. **Celeb-DF**
   - High-quality deepfakes
   - Download: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics/

4. **TIMIT**
   - Audio-visual deepfakes
   - Download: https://github.com/hasacat/TIMIT

## Training Models

### Single Model Training

#### CNN Model
```bash
python train.py --model cnn --epochs 50 --batch-size 32 --device cuda
```

#### ResNext Model
```bash
python train.py --model resnext --epochs 50 --batch-size 32 --device cuda
```

#### Vision Transformer
```bash
python train.py --model vit --epochs 50 --batch-size 16 --device cuda
```

#### Ensemble Model
```bash
python train.py --model ensemble --epochs 50 --batch-size 32 --device cuda
```

### Training Hyperparameters

```bash
python train.py \
  --model ensemble \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --device cuda \
  --save-dir trained_models
```

### Advanced Training Options

```bash
# Resume training from checkpoint
python train.py \
  --model ensemble \
  --checkpoint trained_models/checkpoint.pth \
  --epochs 100

# Distributed training on multiple GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train.py \
  --model ensemble \
  --epochs 100

# Mixed precision training (faster, lower memory)
python train.py \
  --model ensemble \
  --mixed-precision \
  --epochs 100
```

## Model Evaluation

### Evaluation Metrics

```bash
python evaluate.py \
  --model trained_models/ensemble_model.pth \
  --test-dir data/test \
  --device cuda
```

Output:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Cross-Validation

```bash
python cross_validate.py \
  --model ensemble \
  --folds 5 \
  --epochs 50 \
  --data-dir data
```

## Transfer Learning

### Fine-tune Pre-trained Models

```python
from models import ResNextModel

# Load pre-trained model
model = ResNextModel(pretrained=True)

# Freeze backbone
model.freeze_backbone()

# Train classifier head
# ... training code ...

# Unfreeze for full training
model.unfreeze_backbone()

# Continue training
# ... training code ...
```

## Data Augmentation

Enable augmentation during training:

```bash
python train.py \
  --model ensemble \
  --augmentation \
  --augmentation-prob 0.7 \
  --epochs 100
```

Augmentation techniques applied:
- Random horizontal flip
- Random rotation (±10°)
- Random brightness/contrast
- Random Gaussian blur
- Random JPEG compression

## Hyperparameter Tuning

### Grid Search

```bash
python hyperparameter_search.py \
  --learning-rates 1e-4 1e-3 1e-2 \
  --batch-sizes 16 32 64 \
  --optimizers adam sgd \
  --schedulers cosine step
```

### Bayesian Optimization

```bash
pip install optuna

python bayesian_optimization.py \
  --n-trials 100 \
  --epochs 20
```

## Model Optimization

### Quantization

Reduce model size for deployment:

```bash
python quantize_model.py \
  --model trained_models/ensemble_model.pth \
  --output trained_models/ensemble_model_quantized.pth \
  --quantization-type int8
```

### Pruning

Remove non-essential weights:

```bash
python prune_model.py \
  --model trained_models/ensemble_model.pth \
  --pruning-ratio 0.3 \
  --output trained_models/ensemble_model_pruned.pth
```

### Distillation

Create smaller student model:

```bash
python distill_model.py \
  --teacher trained_models/ensemble_model.pth \
  --student-model cnn \
  --temperature 4 \
  --alpha 0.7
```

## Distributed Training

### Data Parallel Training

```bash
python train.py \
  --model ensemble \
  --data-parallel \
  --epochs 100
```

### Distributed Data Parallel

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  train.py \
  --model ensemble \
  --epochs 100
```

## Monitoring Training

### TensorBoard

```bash
pip install tensorboard

# Start training with logging
python train.py \
  --model ensemble \
  --log-dir ./logs

# View in TensorBoard
tensorboard --logdir=./logs
```

### Weights & Biases

```bash
pip install wandb

# Login
wandb login

# Train with W&B logging
python train.py \
  --model ensemble \
  --wandb \
  --epochs 100
```

## Debugging

### Gradient Checking

```python
# Check for gradient explosion/vanishing
from torch.autograd import grad_check

inputs = torch.randn(1, 3, 224, 224, requires_grad=True)
loss = model(inputs).sum()
loss.backward()

# Inspect gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: grad_norm={grad_norm:.4f}")
```

### Layer-wise Analysis

```python
# Analyze activation statistics
from torch.utils.hooks import register_forward_hook

def hook_fn(module, input, output):
    print(f"{module}: mean={output.mean():.4f}, std={output.std():.4f}")

for module in model.modules():
    module.register_forward_hook(hook_fn)
```

## Best Practices

1. **Data Split**: 70% train, 15% validation, 15% test
2. **Balanced Data**: Equal samples of fake and real
3. **Batch Normalization**: Use for stability
4. **Learning Rate Scheduler**: Reduce over time
5. **Early Stopping**: Monitor validation loss
6. **Model Checkpointing**: Save best models
7. **Data Augmentation**: Improve generalization
8. **Class Weighting**: Handle imbalanced data

## Performance Tips

- Use GPU for training (4-10x faster)
- Batch size 32-64 for optimal speed
- Use mixed precision for speed/memory
- Pre-compute and cache features
- Use distributed training for large datasets
- Monitor memory usage with `nvidia-smi`

## Troubleshooting

### Issue: NaN Loss
- Reduce learning rate
- Check for faulty data
- Use gradient clipping

### Issue: Overfitting
- Increase augmentation
- Add dropout/regularization
- Use more training data
- Reduce model complexity

### Issue: Slow Training
- Use GPU
- Increase batch size
- Reduce image resolution
- Use distributed training

### Issue: Out of Memory
- Reduce batch size
- Reduce image resolution
- Use gradient checkpointing
- Use mixed precision

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git push -u origin main
