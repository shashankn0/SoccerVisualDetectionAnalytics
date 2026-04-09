# Unified video processing pipeline
# Usage: python process_video.py [video_path]
# 
# CONFIGURATION: Paste your video path below
# If VIDEO_PATH is set, you can just run: python process_video.py
# Otherwise, provide path as argument: python process_video.py "path\to\video.mp4"

VIDEO_PATH = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"  # PASTE YOUR VIDEO PATH HERE (leave empty to use command line argument)

import sys
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Pitch configuration
PITCH_WIDTH = 105   # meters
PITCH_HEIGHT = 68   # meters
VIZ_SCALE = 8       # pixels per meter in the 2D view

# Global team color centers - fit once on first frame, reuse forever
team_color_centers = None

def get_player_team(frame, bbox):
    """Assign team based on jersey color using KMeans."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop to upper body only (jersey area, avoid shorts/grass)
    player_crop = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
    
    if player_crop.size == 0:
        return np.array([0, 0, 0])
    
    # Reshape for KMeans
    pixels = player_crop.reshape(-1, 3).astype(np.float32)
    
    # Separate player from background using top 2 colors
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(pixels)
    
    # Corner pixels are likely background - identify which cluster is background
    corners = np.array([
        player_crop[0,0], player_crop[0,-1],
        player_crop[-1,0], player_crop[-1,-1]
    ], dtype=np.float32)
    
    corner_labels = kmeans.predict(corners)
    bg_cluster = np.bincount(corner_labels).argmax()
    player_cluster = 1 - bg_cluster
    
    # Return the dominant player color for team assignment
    return kmeans.cluster_centers_[player_cluster]

def classify_referee(frame, bbox):
    """
    Detects if a detection is a referee based on yellow jersey color.
    Returns: True if referee, False otherwise
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop to upper body (jersey area)
    crop = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
    if crop.size == 0:
        return False
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Yellow HSV range (referee kit)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_ratio = np.sum(yellow_mask > 0) / (crop.shape[0] * crop.shape[1] + 1e-5)
    
    # 15%+ yellow pixels = referee
    return yellow_ratio > 0.15

def detect_ball(frame, model):
    """
    Detect ball using the same YOLO model.
    Ball is typically small and round - filter by size and position.
    Returns: (x, y) center of ball or None
    """
    # Run detection with lower confidence for small objects
    results = model(frame, conf=0.15, imgsz=640, verbose=False)[0]
    
    if results.boxes is None or len(results.boxes) == 0:
        return None
    
    # Filter for small objects (ball is typically smaller than players)
    ball_candidates = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # Ball is typically small (less than 50x50 pixels)
        if width < 50 and height < 50:
            # Ball is typically round (width/height ratio close to 1)
            ratio = width / height if height > 0 else 0
            if 0.5 < ratio < 2.0:
                ball_candidates.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(box.conf[0].cpu().numpy()),
                    'size': width * height
                })
    
    # Return highest confidence small object
    if ball_candidates:
        best = max(ball_candidates, key=lambda x: x['conf'])
        x1, y1, x2, y2 = best['box']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return (cx, cy)
    
    return None

def detect_field_lines(frame):
    """
    Detects white field lines using color masking + Hough transform.
    Returns line segments for perspective mapping.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Isolate green field area first
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Isolate white lines within green area
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Only keep white pixels on the green field
    field_lines = cv2.bitwise_and(white_mask, green_mask)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        field_lines, rho=1, theta=np.pi/180,
        threshold=50, minLineLength=60, maxLineGap=20
    )
    
    return lines

def assign_teams_stable(frame, all_player_bboxes):
    """Run KMeans across all players to split into 2 teams using stable cluster centers."""
    global team_color_centers
    
    player_colors = []
    for bbox in all_player_bboxes:
        color = get_player_team(frame, bbox)
        player_colors.append(color)
    
    # Filter out invalid colors
    player_colors = np.array([c for c in player_colors if c is not None], dtype=np.float32)
    
    if len(player_colors) < 2:
        return [0] * len(all_player_bboxes)
    
    if team_color_centers is None:
        # Fit once on first frame with enough players
        if len(player_colors) >= 6:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(player_colors)
            team_color_centers = kmeans.cluster_centers_
            print(f"Team color centers established on frame with {len(player_colors)} players")
        else:
            return [0] * len(all_player_bboxes)
    
    # Always predict using the SAME centers from first frame
    distances = np.array([
        [np.linalg.norm(c - center) for center in team_color_centers]
        for c in player_colors
    ])
    return np.argmin(distances, axis=1).tolist()

def pixel_to_pitch(pixel_x, pixel_y, frame_w, frame_h):
    """Simple perspective mapping from frame coords to pitch coords."""
    # Normalize pixel position
    norm_x = pixel_x / frame_w
    norm_y = pixel_y / frame_h
    
    # Map to pitch dimensions (rough linear mapping)
    # For broadcast angle: bottom of frame = near side, top = far side
    pitch_x = norm_x * PITCH_WIDTH
    pitch_y = norm_y * PITCH_HEIGHT
    
    return int(pitch_x * VIZ_SCALE), int(pitch_y * VIZ_SCALE)

def draw_pitch_frame(player_positions, team_assignments, ball_pos=None):
    """Draw one frame of the 2D pitch view."""
    pitch_img = np.zeros((
        PITCH_HEIGHT * VIZ_SCALE, 
        PITCH_WIDTH * VIZ_SCALE, 3
    ), dtype=np.uint8)
    
    # Draw green pitch
    pitch_img[:] = (34, 139, 34)
    
    # Draw field lines
    cv2.rectangle(pitch_img, (0,0), 
                  (PITCH_WIDTH*VIZ_SCALE-1, PITCH_HEIGHT*VIZ_SCALE-1), 
                  (255,255,255), 2)
    # Center line
    cv2.line(pitch_img, 
             (PITCH_WIDTH*VIZ_SCALE//2, 0),
             (PITCH_WIDTH*VIZ_SCALE//2, PITCH_HEIGHT*VIZ_SCALE),
             (255,255,255), 2)
    # Center circle
    cv2.circle(pitch_img, 
               (PITCH_WIDTH*VIZ_SCALE//2, PITCH_HEIGHT*VIZ_SCALE//2),
               int(9.15*VIZ_SCALE), (255,255,255), 2)
    
    # Draw players
    team_colors = [(255, 50, 50), (50, 50, 255)]  # Red vs Blue
    for (px, py), team in zip(player_positions, team_assignments):
        color = team_colors[int(team) % 2]
        cv2.circle(pitch_img, (px, py), 8, color, -1)
        cv2.circle(pitch_img, (px, py), 8, (255,255,255), 1)
    
    # Draw ball
    if ball_pos:
        cv2.circle(pitch_img, ball_pos, 5, (0, 255, 255), -1)
    
    return pitch_img

def process_video(input_path, output_detection_path=None, output_2d_path=None):
    """Process video with ByteTrack tracking, team assignment, and 2D visualization."""
    
    if not os.path.exists(input_path):
        print(f"Error: Video file not found: {input_path}")
        return None
    
    print(f"Processing video: {input_path}")
    print("Loading yolov8x.pt model (auto-downloads on first run)...")
    
    model = YOLO("yolov8x.pt")
    
    # Create output directory
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if output_detection_path is None:
        output_detection_path = os.path.join(output_dir, f"{base_name}_detected_{timestamp}.mp4")
    if output_2d_path is None:
        output_2d_path = os.path.join(output_dir, f"{base_name}_2d_{timestamp}.mp4")
    
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
    out_2d = cv2.VideoWriter(output_2d_path, fourcc, fps, 
                               (PITCH_WIDTH*VIZ_SCALE, PITCH_HEIGHT*VIZ_SCALE))
    
    print("Running ByteTrack tracking across video...")
    
    # Run tracking across entire video
    results = model.track(
        source=input_path,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.45,
        imgsz=1280,
        classes=[0],      # person only
        persist=True,
        stream=True       # memory-efficient
    )
    
    cap.release()
    cap = cv2.VideoCapture(input_path)  # reopen for drawing
    
    frame_count = 0
    for result in results:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            out_det.write(frame)
            out_2d.write(draw_pitch_frame([], []))
            continue
        
        bboxes = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else \
                    np.arange(len(bboxes))
        
        # Assign teams by jersey color (using stable clustering)
        teams = assign_teams_stable(frame, bboxes)
        
        # Detect ball
        ball_pos = detect_ball(frame, model)
        
        # Draw detection frame with referee detection
        det_frame = frame.copy()
        label_colors = {
            'team_a': (0, 80, 255),    # Orange/Red
            'team_b': (255, 80, 0),    # Blue
            'referee': (0, 255, 255),  # Yellow
        }
        
        for bbox, tid, team in zip(bboxes, track_ids, teams):
            x1,y1,x2,y2 = map(int, bbox)
            
            # Check if this is a referee
            is_referee_flag = classify_referee(frame, bbox)
            
            if is_referee_flag:
                role = 'referee'
                color = label_colors['referee']
                label = "REF"
            else:
                role = 'team_a' if team == 0 else 'team_b'
                color = label_colors[role]
                label = f"#{int(tid)}"
            
            cv2.rectangle(det_frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(det_frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        # Draw ball on detection frame
        if ball_pos:
            cx, cy = ball_pos
            cv2.circle(det_frame, (cx, cy), 8, (0, 255, 255), -1)   # Yellow dot
            cv2.circle(det_frame, (cx, cy), 8, (0, 0, 0), 2)        # Black outline
            cv2.putText(det_frame, "BALL", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        
        out_det.write(det_frame)
        
        # Build 2D positions
        player_positions = []
        for bbox in bboxes:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int(bbox[3])  # feet position
            px, py = pixel_to_pitch(cx, cy, frame_w, frame_h)
            player_positions.append((px, py))
        
        # Map ball to 2D pitch
        ball_2d_pos = None
        if ball_pos:
            bx, by = ball_pos
            ball_2d_pos = pixel_to_pitch(bx, by, frame_w, frame_h)
        
        pitch_frame = draw_pitch_frame(player_positions, teams, ball_2d_pos)
        out_2d.write(pitch_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    out_det.release()
    out_2d.release()
    
    print(f"\nProcessing complete!")
    print(f"Detection output: {output_detection_path}")
    print(f"2D visualization: {output_2d_path}")
    
    return output_detection_path, output_2d_path

def main():
    # Use VIDEO_PATH variable if set, otherwise use command line argument
    if VIDEO_PATH:
        video_path = VIDEO_PATH
        print(f"Using configured video path: {video_path}")
    elif len(sys.argv) >= 2:
        video_path = sys.argv[1]
        print(f"Using command line argument: {video_path}")
    else:
        print("No video path provided.")
        print("\nOptions:")
        print("1. Set VIDEO_PATH variable at top of this file")
        print("2. Or provide path as argument: python process_video.py \"path\\to\\video.mp4\"")
        print("\nExample: python process_video.py C:\\Users\\snell\\Downloads\\samples\\game.mp4")
        return
    
    # Convert relative path to absolute if needed
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    det_path, viz_path = process_video(video_path)
    
    if det_path and viz_path:
        print(f"\nTo view the processed videos:")
        print(f"  Detection: {det_path}")
        print(f"  2D View: {viz_path}")
        print(f"\nNext steps:")
        print(f"  1. Upload the original video to the web interface at http://localhost:8082")
        print(f"  2. The processed videos will be displayed in the 3-panel view")

if __name__ == "__main__":
    main()
