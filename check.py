import sys
import platform
import torch
import tensorflow as tf
import numpy as np

def system_diagnostics():
    print("System Diagnostics:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    print("\nLibrary Versions:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    print("\nHardware Information:")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch MPS Available: {torch.backends.mps.is_available()}")
    
    try:
        model_path = 'assets/models/FER_static_ResNet50_AffectNet.pt'
        model = torch.load(model_path, map_location='cpu')
        print("\nModel Loading:")
        print("Model keys:")
        for key in model.state_dict().keys():
            print(key)
    except Exception as e:
        print(f"\nModel Loading Error: {e}")

if __name__ == "__main__":
    system_diagnostics()