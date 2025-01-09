import os
import time
import cv2
import numpy as np
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import onnxruntime as ort
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PYTORCH_MODEL_PATH = 'assets/models/FER_static_ResNet50_AffectNet.pt'
ONNX_MODEL_PATH = 'assets/models/FER_static_ResNet50_AffectNet.onnx'
TFLITE_MODEL_PATH = 'assets/models/FER_static_ResNet50_AffectNet.tflite'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = (224, 224)

class ResNet50(nn.Module):
    def __init__(self, num_classes=7, channels=3):
        super(ResNet50, self).__init__()
        # Initial layers
        self.conv_layer_s2_same = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Load pre-trained ResNet50 model
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Extract layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Fully connected layers
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layer_s2_same(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ModelComparison:
    def __init__(self):
        # PyTorch Model
        self.pytorch_model = self._load_pytorch_model()
        
        # ONNX Model
        self.onnx_session = self._load_onnx_model()
        
        # TensorFlow Lite Model
        self.tflite_interpreter = self._load_tflite_model()

    def _load_pytorch_model(self):
        model = ResNet50(num_classes=7, channels=3)
        
        # Load the state dict
        state_dict = torch.load(PYTORCH_MODEL_PATH, map_location='cpu')
        
        # Create a mapping to handle different key names
        state_dict_mapping = {}
        for k, v in state_dict.items():
            # Replace batch_norm with bn, i_downsample with downsample
            k = k.replace('batch_norm', 'bn')
            k = k.replace('i_downsample', 'downsample')
            
            # Modify layer names to match standard ResNet architecture
            k = k.replace('layer1.0.', 'layer1.0.')
            k = k.replace('layer1.1.', 'layer1.1.')
            # Continue for other layers as needed
            
            state_dict_mapping[k] = v
        
        # Load with a more lenient approach
        model.load_state_dict(state_dict_mapping, strict=False)
        model.eval()
        return model

    def _load_onnx_model(self):
        return ort.InferenceSession(ONNX_MODEL_PATH)

    def _load_tflite_model(self):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter

    def preprocess_frame(self, frame):
        # Resize and normalize
        resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        input_data = resized.astype(np.float32) / 255.0
        input_data = (input_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return input_data

    def pytorch_inference(self, frame):
        input_data = self.preprocess_frame(frame)
        input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            output = self.pytorch_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            emotion_idx = torch.argmax(probabilities).item()
        return EMOTION_LABELS[emotion_idx], probabilities[0][emotion_idx].item()

    def onnx_inference(self, frame):
        input_data = self.preprocess_frame(frame)
        input_tensor = input_data.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        ort_inputs = {self.onnx_session.get_inputs()[0].name: input_tensor}
        output = self.onnx_session.run(None, ort_inputs)[0]
        
        emotion_idx = np.argmax(output)
        return EMOTION_LABELS[emotion_idx], output[0][emotion_idx]

    def tflite_inference(self, frame):
        input_data = self.preprocess_frame(frame)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()
        
        self.tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        self.tflite_interpreter.invoke()
        output = self.tflite_interpreter.get_tensor(output_details[0]['index'])
        
        emotion_idx = np.argmax(output)
        return EMOTION_LABELS[emotion_idx], output[0][emotion_idx]

def benchmark_models(model_comparison, input_video):
    cap = cv2.VideoCapture(input_video)
    
    frameworks = {
        'PyTorch': model_comparison.pytorch_inference,
        'ONNX Runtime': model_comparison.onnx_inference,
        'TensorFlow Lite': model_comparison.tflite_inference
    }
    
    results = {framework: {'frames': 0, 'total_time': 0} for framework in frameworks}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        for framework, inference_func in frameworks.items():
            start_time = time.time()
            emotion, confidence = inference_func(frame)
            inference_time = time.time() - start_time
            
            results[framework]['frames'] += 1
            results[framework]['total_time'] += inference_time
    
    cap.release()
    
    # Print comparison results
    logger.info("\nModel Performance Comparison:")
    for framework, data in results.items():
        fps = data['frames'] / data['total_time'] if data['total_time'] > 0 else 0
        logger.info(f"{framework}:")
        logger.info(f"  Total Frames: {data['frames']}")
        logger.info(f"  Total Inference Time: {data['total_time']:.4f}s")
        logger.info(f"  Average FPS: {fps:.2f}")

def main():
    input_video = 'assets/videos/input.mp4'
    model_comparison = ModelComparison()
    benchmark_models(model_comparison, input_video)

if __name__ == "__main__":
    main()