import os
import time
import cv2
import numpy as np
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = 'assets/models/FER_static_ResNet50_AffectNet.tflite'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = (224, 224)

class EmotionRecognizer:
   def __init__(self, model_path):
       try:
           # Try XNNPACK delegation
           try:
               delegates = [
                   tf.lite.experimental.XNNPACKDelegate(
                       options={
                           'num_threads': 2,
                           'performance_preference': 0.5
                       }
                   )
               ]
               self.interpreter = tf.lite.Interpreter(
                   model_path=model_path, 
                   experimental_delegates=delegates
               )
           except Exception as delegate_error:
               logger.warning(f"XNNPACK delegation failed: {delegate_error}")
               # Fallback to standard interpreter
               self.interpreter = tf.lite.Interpreter(model_path=model_path)
           
           self.interpreter.allocate_tensors()
           
           # Get input and output tensor details
           self.input_details = self.interpreter.get_input_details()
           self.output_details = self.interpreter.get_output_details()
           
           # Detailed model input/output logging
           logger.info("Model Input Details:")
           for detail in self.input_details:
               logger.info(f"  Shape: {detail['shape']}")
               logger.info(f"  Type: {detail['dtype']}")
           
           logger.info("Model Output Details:")
           for detail in self.output_details:
               logger.info(f"  Shape: {detail['shape']}")
               logger.info(f"  Type: {detail['dtype']}")
           
           logger.info("TensorFlow Lite model loaded successfully")
       except Exception as e:
           logger.error(f"Comprehensive model loading error: {e}")
           raise

   def preprocess_frame(self, frame):
    # Resize to exactly match model input
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize and explicitly cast to float32
    input_data = resized.astype(np.float32) / 255.0
    input_data = (input_data - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    
    # Ensure correct shape: (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data
  
   def recognize_emotion(self, frame):
       # Preprocess and run inference
       input_data = self.preprocess_frame(frame)
       self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
       self.interpreter.invoke()
       
       # Extract emotion
       output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
       emotion_idx = np.argmax(output_data)
       confidence = output_data[0][emotion_idx]
       
       return EMOTION_LABELS[emotion_idx], float(confidence)

def process_video(input_path, output_path, emotion_recognizer):
   cap = cv2.VideoCapture(input_path)
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
   out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
   
   frame_count = 0
   start_time = time.time()
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       frame_count += 1
       
       # Emotion recognition
       emotion, confidence = emotion_recognizer.recognize_emotion(frame)
       
       # Annotate frame
       text = f"{emotion} ({confidence:.2f})"
       cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       out.write(frame)
       
       # Progress logging
       if frame_count % 10 == 0:
           logger.info(f"Processing frame {frame_count}")
   
   cap.release()
   out.release()
   
   duration = time.time() - start_time
   logger.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")
   
   return frame_count

def main():
   input_video = 'assets/videos/input.mp4'
   output_video = 'output.mp4'
   
   # Initialize emotion recognizer
   emotion_recognizer = EmotionRecognizer(MODEL_PATH)
   
   # Process video
   process_video(input_video, output_video, emotion_recognizer)

if __name__ == "__main__":
   main()