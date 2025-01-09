import torch
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class TFResNet50(tf.keras.Model):
    def __init__(self, num_classes=7, channels=3):
        super(TFResNet50, self).__init__()
        # Initial layers matching PyTorch model
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', input_shape=(224, 224, channels))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(3, strides=2, padding='same')

        # Use pre-trained ResNet50 as base
        base_model = models.ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, channels)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Custom top layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = base_model(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def convert_pytorch_to_tflite(pth_path, tflite_path):
    # Load PyTorch model state dict for reference
    pth_state_dict = torch.load(pth_path, map_location='cpu')
    
    # Create TensorFlow model
    tf_model = TFResNet50(num_classes=7, channels=3)
    
    # Dummy input for tracing
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # TensorFlow Lite Converter with optimizations
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
        tf.lite.Optimize.OPTIMIZE_FOR_SIZE
    ]
    
    # XNNPACK and quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert and save model
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Converted TensorFlow Lite model saved to {tflite_path}")

if __name__ == "__main__":
    convert_pytorch_to_tflite(
        'assets/models/FER_static_ResNet50_AffectNet.pt',
        'assets/models/FER_static_ResNet50_AffectNet.tflite'
    )