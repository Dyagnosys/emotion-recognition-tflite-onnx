# Multi-Framework Emotion Recognition

## Overview
A comprehensive emotion recognition pipeline supporting multiple deep learning frameworks (PyTorch, TensorFlow Lite) with optimized inference for various hardware platforms.

## Features
- Multiple framework support
- Real-time emotion detection
- Hardware-optimized inference
- Video processing capabilities
- Intel CPU optimization (XNNPACK)

## Supported Emotions
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Prerequisites
- Python 3.8+
- Docker (optional)
- Supported frameworks:
  * PyTorch
  * TensorFlow
  * ONNX Runtime

## Installation

### Clone Repository
```bash
git clone https://github.com/dyagnosys/emotion-recognition-toolkit.git
cd emotion-recognition-toolkit
```

### Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### PyTorch Inference
```bash
python main_pytorch.py
```

### TensorFlow Lite Inference
```bash
python main_tflite.py
```

## Docker Support
```bash
# Build Docker image
docker build -t emotion-recognition .

# Run Docker container
docker run --rm emotion-recognition
```

## Model Conversion
```bash
python convert_model.py
```

## Performance Optimization
- XNNPACK acceleration
- Multi-threading
- Intel CPU optimization

## Benchmarks
- Framework performance comparison
- Hardware-specific optimizations

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License
MIT License

## Acknowledgments
- ResNet50 Architecture
- AffectNet Dataset
- Open-source deep learning frameworks