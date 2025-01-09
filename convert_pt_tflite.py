import os
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import preprocess_input

class TFResNet50(Model):
    def __init__(self, num_classes=7, channels=3):
        super(TFResNet50, self).__init__()
        # Input layer
        inputs = layers.Input(shape=(224, 224, channels))
        
        # Preprocessing
        x = preprocess_input(inputs)
        
        # Base ResNet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_tensor=inputs
        )
        base_model.trainable = False
        
        # Custom top layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)

    def call(self, x):
        return self.model(x)

def convert_pytorch_to_tflite(pth_path, tflite_path):
    # Create TensorFlow model
    tf_model = TFResNet50(num_classes=7, channels=3)
    
    # Dummy input for tracing
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Trace the model
    concrete_func = tf.function(lambda x: tf_model(x))
    concrete_func.get_concrete_function(
        tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)
    )
    
    # TensorFlow Lite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model.model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert and save model
    tflite_model = converter.convert()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    
    # Save model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Converted TensorFlow Lite model saved to {tflite_path}")

if __name__ == "__main__":
    convert_pytorch_to_tflite(
        'assets/models/FER_static_ResNet50_AffectNet.pt',
        'assets/models/FER_static_ResNet50_AffectNet.tflite'
    )