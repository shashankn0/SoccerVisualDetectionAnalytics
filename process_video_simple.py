from roboflow import Roboflow
import cv2
import numpy as np
from datetime import datetime
import os

# Video configuration
VIDEO_PATH = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"

def process_video_simple(input_path, output_detection_path=None):
    """Process video with RoboFlow v20 detection (minimal version)."""
    
    print(f"Processing video: {input_path}")
    print("Initializing RoboFlow v20 model (mAP 83.0%)...")
    
    # Initialize RoboFlow v20 model (same as test script)
    rf = Roboflow(api_key="XgThuNgb3yfLih6r9k7K")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    model = project.version(20).model
    print("RoboFlow v20 model loaded successfully!")
    
    # Create output directory
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if output_detection_path is None:
        output_detection_path = os.path.join(output_dir, f"{base_name}_simple_{timestamp}.mp4")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_w}x{frame_h} @ {fps} FPS, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_det = cv2.VideoWriter(output_detection_path, fourcc, fps, (frame_w, frame_h))
    
    print("Running RoboFlow v20 detection across video...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Use RoboFlow v20 for detection (same as test script)
        cv2.imwrite("temp_frame.jpg", frame)
        result = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()
        predictions = result.get('predictions', [])
        
        # Clean up temp file
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
        
        # Draw detections (simple version)
        det_frame = frame.copy()
        for pred in predictions:
            x, y = pred['x'], pred['y']
            w, h = pred['width'], pred['height']
            cls = pred['class']
            cls_id = pred['class_id']
            
            # Color by class
            if cls_id == 0:  # ball
                color = (0, 255, 255)  # Yellow
            elif cls_id == 1:  # goalkeeper
                color = (0, 255, 0)  # Green
            elif cls_id == 2:  # player
                color = (0, 80, 255)  # Orange
            elif cls_id == 3:  # referee
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            cv2.rectangle(det_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            cv2.putText(det_frame, cls, (int(x), int(y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out_det.write(det_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - {len(predictions)} detections")
    
    cap.release()
    out_det.release()
    
    print(f"\nProcessing complete!")
    print(f"Detection output: {output_detection_path}")
    
    return output_detection_path

if __name__ == "__main__":
    process_video_simple(VIDEO_PATH)
