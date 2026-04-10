from roboflow import Roboflow
import cv2

# RoboFlow API configuration
API_KEY = "XgThuNgb3yfLih6r9k7K"

# Test with a single frame using roboflow package
video_path = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit(1)

# Save frame temporarily
cv2.imwrite("test_frame.jpg", frame)

print("Testing RoboFlow package with v20 model...")
try:
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    model = project.version(20).model
    
    result = model.predict("test_frame.jpg", confidence=40, overlap=30).json()
    print(f"Success! Detected {len(result.get('predictions', []))} objects")
    print(f"Sample prediction: {result.get('predictions', [])[:1]}")
except Exception as e:
    print(f"Error: {e}")

# Clean up
import os
if os.path.exists("test_frame.jpg"):
    os.remove("test_frame.jpg")

cap.release()
