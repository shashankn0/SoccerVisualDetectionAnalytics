from ultralytics import YOLO
import cv2
import pandas as pd

video_path = "uploaded_videos/videoplaybacktest.mp4"
model = YOLO('models/player.pt')  # Change path as needed

# OpenCV to read frame-by-frame
cap = cv2.VideoCapture(video_path)
frame_idx = 0
results_data = []  # Collects all detection info

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run YOLO detection
    result = model.predict(frame, conf=0.4, verbose=False)[0]
    # Loop through results: boxes.xyxy gives bounding boxes, boxes.cls is class index
    for bbox, class_id, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = bbox
        results_data.append({
            'frame': frame_idx,
            'class_id': int(class_id),
            'confidence': float(conf),
            'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)
        })
    frame_idx += 1
cap.release()

# Turn results into DataFrame for easy analysis
results_df = pd.DataFrame(results_data)
results_df.to_csv('detection_results.csv', index=False)
print(f'Saved detections for {frame_idx} frames.')
