from roboflow import Roboflow
import cv2
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

# Video configuration
VIDEO_PATH = r"C:\Users\snell\Downloads\samples\videoplaybacktest.mp4"

# Global team color centers
team_color_centers = None

# Track team assignments per player ID for temporal consistency
player_team_history = {}

def get_player_team(frame, bbox):
    """Assign team based on jersey color using KMeans."""
    x1, y1, x2, y2 = map(int, bbox)
    
    player_crop = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
    
    if player_crop.size == 0:
        return np.array([0, 0, 0])
    
    # Convert to RGB for KMeans
    player_crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
    pixels = player_crop_rgb.reshape(-1, 3)
    
    # Use 2 clusters for 2 teams
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    return kmeans.cluster_centers_[kmeans.labels_[0]]

def assign_teams_stable(frame, bboxes, track_ids):
    """Assign teams using stable clustering with temporal consistency."""
    global team_color_centers, player_team_history
    
    # First frame: establish team color centers
    if team_color_centers is None:
        colors = []
        for bbox in bboxes:
            color = get_player_team(frame, bbox)
            colors.append(color)
        
        colors = np.array(colors)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(colors)
        team_color_centers = kmeans.cluster_centers_
    
    # Assign each player to nearest team center
    teams = []
    for tid, bbox in zip(track_ids, bboxes):
        color = get_player_team(frame, bbox)
        
        # Check if this player has a history
        if tid in player_team_history:
            teams.append(player_team_history[tid])
        else:
            # Assign to nearest team center
            dists = np.linalg.norm(team_color_centers - color, axis=1)
            team = np.argmin(dists)
            teams.append(team)
            player_team_history[tid] = team
    
    return np.array(teams)

def process_video_roboflow(input_path, output_detection_path=None):
    """Process video with RoboFlow v20 detection."""
    
    print(f"Processing video: {input_path}")
    print("Initializing RoboFlow v20 model (mAP 83.0%)...")
    
    # Initialize RoboFlow v20 model (same as test script)
    rf = Roboflow(api_key="XgThuNgb3yfLih6r9k7K")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    model = project.version(20).model
    print("RoboFlow v20 model loaded successfully!")
    
    # Create output directory
    output_dir = "processed_videos"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if output_detection_path is None:
        output_detection_path = os.path.join(output_dir, f"{base_name}_roboflow_{timestamp}.mp4")
    
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
        
        if len(predictions) == 0:
            out_det.write(frame)
            continue
        
        # Convert RoboFlow detections to bounding boxes
        bboxes = []
        classes = []
        for pred in predictions:
            x, y = pred['x'], pred['y']
            w, h = pred['width'], pred['height']
            bboxes.append([x, y, x + w, y + h])
            classes.append(pred['class_id'])  # 0=ball, 1=goalkeeper, 2=player, 3=referee
        
        bboxes = np.array(bboxes)
        classes = np.array(classes)
        
        # Assign teams by jersey color
        track_ids = np.arange(len(bboxes))
        teams = assign_teams_stable(frame, bboxes, track_ids)
        
        # Draw detection frame
        det_frame = frame.copy()
        label_colors = {
            'team_a': (0, 80, 255),
            'team_b': (255, 80, 0),
            'referee': (0, 255, 255),
        }
        
        for bbox, tid, team, cls in zip(bboxes, track_ids, teams, classes):
            x1, y1, x2, y2 = map(int, bbox)
            
            if cls == 0:  # ball
                continue
            elif cls == 1:  # goalkeeper
                role = 'team_a' if team == 0 else 'team_b'
                color = label_colors[role]
                label = f"GK#{int(tid)}"
                box_thickness = 4
            elif cls == 3:  # referee
                role = 'referee'
                color = label_colors['referee']
                label = "REF"
                box_thickness = 2
            else:  # player
                role = 'team_a' if team == 0 else 'team_b'
                color = label_colors[role]
                label = f"#{int(tid)}"
                box_thickness = 2
            
            cv2.rectangle(det_frame, (x1, y1), (x2, y2), color, box_thickness)
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

if __name__ == "__main__":
    process_video_roboflow(VIDEO_PATH)
