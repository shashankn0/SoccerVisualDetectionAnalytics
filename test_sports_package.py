import sports
from sports import PlayerDetectionModel
import cv2
import sys

# Initialize sports player detection model
print("Loading sports player detection model...")
model = PlayerDetectionModel()

# Test on video
video_path = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    sys.exit(1)

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run sports player detection
    try:
        result = model.predict(frame)
        print(f"Frame {frame_count}/{total_frames}: Detected {len(result)} players")
    except Exception as e:
        print(f"Error on frame {frame_count}: {e}")
    
    # Stop after 10 frames for testing
    if frame_count >= 10:
        break

cap.release()
print("\nTest complete!")
