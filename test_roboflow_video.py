from roboflow import Roboflow
import cv2
import numpy as np

# RoboFlow API configuration
API_KEY = "XgThuNgb3yfLih6r9k7K"

print("Initializing RoboFlow v20 model...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
model = project.version(20).model
print("Model loaded successfully!")

# Test with first 5 frames of video
video_path = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"
cap = cv2.VideoCapture(video_path)

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"\nProcessing frame {i+1}...")
    cv2.imwrite("temp_frame.jpg", frame)
    
    result = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()
    predictions = result.get('predictions', [])
    
    print(f"Detected {len(predictions)} objects")
    
    # Count by class
    class_counts = {}
    for pred in predictions:
        cls = pred['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"Class breakdown: {class_counts}")

# Clean up
import os
if os.path.exists("temp_frame.jpg"):
    os.remove("temp_frame.jpg")

cap.release()
print("\nTest complete!")
