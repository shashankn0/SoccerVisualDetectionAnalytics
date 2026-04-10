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

# Track team assignments per player ID for temporal consistency
player_team_history = {}  # {track_id: team_assignment}

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

def classify_referee(frame, bbox, frame_w, frame_h):
    """
    Detects if a detection is a referee based on yellow jersey color and position.
    Excludes spectators in the stands.
    Returns: True if referee, False otherwise
    """
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Position filtering: exclude top 20% of frame where stands typically are
    if cy < frame_h * 0.2:
        return False
    
    # Minimum player height to exclude distant spectators
    player_height = y2 - y1
    if player_height < frame_h * 0.05:  # Less than 5% of frame height
        return False
    
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

def classify_goalie(frame, bbox, frame_w, frame_h, team_color_centers):
    """
    Detects if a player is a goalie based on position, different jersey color, and gloves.
    Returns: True if goalie, False otherwise
    """
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Goalies are typically near the left or right edges of the frame (goal areas)
    # Use 15% of frame width from each edge as goal area
    goal_area_threshold = frame_w * 0.15
    
    # Check if player is in left or right goal area
    in_goal_area = cx < goal_area_threshold or cx > (frame_w - goal_area_threshold)
    
    # Goalies are also typically in the upper half of the field (further away)
    in_upper_half = cy < frame_h * 0.6
    
    if not (in_goal_area and in_upper_half):
        return False
    
    # Check if jersey color is different from both team colors
    if team_color_centers is not None:
        player_color = get_player_team(frame, bbox)
        distances = [np.linalg.norm(player_color - center) for center in team_color_centers]
        # If player color is far from both team colors, likely a goalie
        if min(distances) > 50:  # Threshold for "different" color
            return True
    
    # Check for gloves (skin color in hand area)
    # Crop to lower body area where hands might be visible
    hand_crop = frame[y1+int((y2-y1)*0.5):y2, x1:x2]
    if hand_crop.size > 0:
        hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
        # Skin color range (approximate)
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_ratio = np.sum(skin_mask > 0) / (hand_crop.shape[0] * hand_crop.shape[1] + 1e-5)
        if skin_ratio > 0.05:  # 5%+ skin pixels = likely gloves/hands
            return True
    
    return in_goal_area and in_upper_half

def assign_teams_stable(frame, all_player_bboxes, track_ids):
    """Run KMeans across all players to split into 2 teams using stable cluster centers and temporal consistency."""
    global team_color_centers, player_team_history
    
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
    new_assignments = np.argmin(distances, axis=1).tolist()
    
    # Apply temporal consistency - use previous assignments when available
    final_assignments = []
    for i, tid in enumerate(track_ids):
        tid_int = int(tid)
        if tid_int in player_team_history:
            # Use previous assignment to prevent color switching
            final_assignments.append(player_team_history[tid_int])
        else:
            # New player or no history - use current prediction
            final_assignments.append(new_assignments[i])
            player_team_history[tid_int] = new_assignments[i]
    
    # Update history with current assignments
    for i, tid in enumerate(track_ids):
        tid_int = int(tid)
        player_team_history[tid_int] = final_assignments[i]
    
    return final_assignments

def process_video(input_path, output_detection_path=None):
    """Process video with ByteTrack tracking and team assignment."""
    
    if not os.path.exists(input_path):
        print(f"Error: Video file not found: {input_path}")
        return None
    
    print(f"Processing video: {input_path}")
    print("Loading yolov8s.pt model with ByteTrack tracking...")
    
    model = YOLO("yolov8s.pt")
    
    # Create output directory
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if output_detection_path is None:
        output_detection_path = os.path.join(output_dir, f"{base_name}_detected_{timestamp}.mp4")
    
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
    
    print("Running ByteTrack tracking across video...")
    
    # Run tracking across entire video
    results = model.track(
        source=input_path,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        classes=[0],      # person only
        persist=True
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
            continue
        
        bboxes = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else \
                    np.arange(len(bboxes))
        
        # Assign teams by jersey color (using stable clustering with temporal consistency)
        teams = assign_teams_stable(frame, bboxes, track_ids)
        
        # Draw detection frame with referee and goalie detection
        det_frame = frame.copy()
        label_colors = {
            'team_a': (0, 80, 255),    # Orange/Red
            'team_b': (255, 80, 0),    # Blue
            'referee': (0, 255, 255),  # Yellow
        }
        
        for bbox, tid, team in zip(bboxes, track_ids, teams):
            x1,y1,x2,y2 = map(int, bbox)
            
            # Check if this is a referee
            is_referee_flag = classify_referee(frame, bbox, frame_w, frame_h)
            
            # Check if this is a goalie (only if not a referee)
            is_goalie_flag = False
            if not is_referee_flag:
                is_goalie_flag = classify_goalie(frame, bbox, frame_w, frame_h, team_color_centers)
            
            if is_referee_flag:
                role = 'referee'
                color = label_colors['referee']
                label = "REF"
            else:
                role = 'team_a' if team == 0 else 'team_b'
                color = label_colors[role]
                if is_goalie_flag:
                    label = f"GK#{int(tid)}"
                else:
                    label = f"#{int(tid)}"
            
            # Draw thicker box for goalies to distinguish them
            box_thickness = 4 if is_goalie_flag else 2
            cv2.rectangle(det_frame, (x1,y1), (x2,y2), color, box_thickness)
            cv2.putText(det_frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        out_det.write(det_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    out_det.release()
    
    print(f"\nProcessing complete!")
    print(f"Detection output: {output_detection_path}")
    
    return output_detection_path

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
    
    det_path = process_video(video_path)
    
    if det_path:
        print(f"\nTo view the processed video:")
        print(f"  Detection: {det_path}")

if __name__ == "__main__":
    main()
