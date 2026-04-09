# enhanced video detection system with comprehensive object detection and tracking
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import math

class EnhancedSoccerDetector:
    """comprehensive soccer detection system with tracking and field transformation."""
    
    def __init__(self):
        # detection models
        self.models = {}
        self.confidence_threshold = 0.4
        
        # object classes and colors
        self.classes = {
            'player': {'color': (0, 255, 0), 'label': 'Player'},
            'goalkeeper': {'color': (255, 0, 0), 'label': 'Goalkeeper'},
            'referee': {'color': (0, 0, 255), 'label': 'Referee'},
            'ball': {'color': (255, 255, 0), 'label': 'Ball'},
            'field': {'color': (255, 165, 0), 'label': 'Field'},
            'goal': {'color': (128, 0, 128), 'label': 'Goal'}
        }
        
        # tracking system
        self.trackers = {}
        self.next_id = 0
        self.tracks = defaultdict(lambda: {'positions': deque(maxlen=30), 'class': None, 'last_seen': 0})
        
        # field transformation matrices
        self.field_homography = None
        self.field_corners = None
        self.pitch_dimensions = (105, 68)  # standard soccer pitch in meters
        
        # load models
        self.load_models()
    
    def load_models(self):
        """load detection models for different objects."""
        model_paths = {
            'players': 'models/player.pt',
            'ball': 'models/ball.pt',
            'field': 'models/field.pt'
        }
        
        for name, path in model_paths.items():
            try:
                if path and os.path.exists(path):
                    self.models[name] = YOLO(path)
                    print(f"+ loaded {name} model")
                else:
                    print(f"- model file not found: {path}")
            except Exception as e:
                print(f"- failed to load {name} model: {e}")
    
    def detect_objects(self, frame):
        """detect all objects in frame using available models."""
        detections = {
            'players': [],
            'ball': [],
            'field': [],
            'goalkeepers': [],
            'referees': []
        }
        
        # detect players and separate goalkeepers
        if 'players' in self.models:
            results = self.models['players'].predict(frame, conf=self.confidence_threshold, verbose=False)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # determine if goalkeeper based on position or class
                    obj_class = self.classify_player_role(x1, y1, x2, y2, frame.shape)
                    
                    detections[obj_class].append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': cls
                    })
        
        # detect ball
        if 'ball' in self.models:
            results = self.models['ball'].predict(frame, conf=self.confidence_threshold, verbose=False)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections['ball'].append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        # detect field lines for homography
        if 'field' in self.models:
            results = self.models['field'].predict(frame, conf=self.confidence_threshold, verbose=False)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections['field'].append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        return detections
    
    def classify_player_role(self, x1, y1, x2, y2, frame_shape):
        """classify player as goalkeeper, referee, or regular player based on position."""
        h, w = frame_shape[:2]
        
        # goalkeepers are typically near the goals (top/bottom 20% of frame)
        goal_area_threshold = 0.2
        center_y = (y1 + y2) / 2
        
        if center_y < h * goal_area_threshold or center_y > h * (1 - goal_area_threshold):
            return 'goalkeepers'
        
        # referees can be identified by different movement patterns or colors
        # for now, classify based on position away from main player clusters
        return 'players'
    
    def update_tracks(self, detections, frame_num):
        """update object tracks using simple tracking algorithm."""
        current_tracks = {}
        
        for class_name, objects in detections.items():
            if class_name == 'field':  # don't track field lines
                continue
                
            for obj in objects:
                bbox = obj['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # find best match from existing tracks
                best_id = None
                best_distance = float('inf')
                
                for track_id, track_data in self.tracks.items():
                    if track_data['class'] != class_name:
                        continue
                    
                    if track_data['positions']:
                        last_pos = track_data['positions'][-1]
                        distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                        
                        if distance < best_distance and distance < 100:  # max tracking distance
                            best_distance = distance
                            best_id = track_id
                
                if best_id is None:
                    # create new track
                    best_id = self.next_id
                    self.next_id += 1
                    self.tracks[best_id]['class'] = class_name
                
                # update track
                self.tracks[best_id]['positions'].append(center)
                self.tracks[best_id]['last_seen'] = frame_num
                current_tracks[best_id] = {
                    'bbox': bbox,
                    'center': center,
                    'class': class_name,
                    'confidence': obj.get('confidence', 0.0)
                }
        
        # remove old tracks
        current_frame = frame_num
        old_tracks = [tid for tid, data in self.tracks.items() 
                     if current_frame - data['last_seen'] > 30]
        for tid in old_tracks:
            del self.tracks[tid]
        
        return current_tracks
    
    def detect_field_lines(self, frame):
        """detect field lines and calculate homography matrix."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect lines using hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # find field corners from line intersections
            corners = self.find_field_corners(lines, frame.shape)
            if len(corners) >= 4:
                self.field_corners = corners[:4]
                self.field_homography = self.calculate_homography(corners[:4], frame.shape)
                return True
        
        return False
    
    def find_field_corners(self, lines, frame_shape):
        """find field corners from detected lines."""
        h, w = frame_shape[:2]
        corners = []
        
        # look for intersections that could be field corners
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # calculate intersection point
                intersection = self.line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                
                if intersection:
                    x, y = intersection
                    # check if intersection is within reasonable bounds
                    if 0 < x < w and 0 < y < h:
                        corners.append((x, y))
        
        # filter and sort corners
        if corners:
            corners = self.filter_corners(corners, frame_shape)
        
        return corners
    
    def line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """calculate intersection point of two lines."""
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        if 0 <= t <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return (x, y)
        
        return None
    
    def filter_corners(self, corners, frame_shape):
        """filter and sort field corners."""
        h, w = frame_shape[:2]
        
        # keep only corners that are likely field corners
        filtered = []
        for corner in corners:
            x, y = corner
            # corners should be away from center and towards edges
            center_dist = math.sqrt((x - w/2)**2 + (y - h/2)**2)
            edge_dist = min(x, y, w-x, h-y)
            
            if center_dist > min(w, h) * 0.3 and edge_dist < min(w, h) * 0.2:
                filtered.append(corner)
        
        # sort corners: top-left, top-right, bottom-right, bottom-left
        if len(filtered) >= 4:
            filtered.sort(key=lambda p: (p[1], p[0]))
            top_corners = sorted(filtered[:2], key=lambda p: p[0])
            bottom_corners = sorted(filtered[2:4], key=lambda p: p[0])
            return top_corners + bottom_corners
        
        return filtered
    
    def calculate_homography(self, corners, frame_shape):
        """calculate homography matrix for field transformation."""
        h, w = frame_shape[:2]
        
        # destination points on standard pitch
        pitch_corners = np.array([
            [0, 0],           # top-left
            [self.pitch_dimensions[0], 0],  # top-right
            [self.pitch_dimensions[0], self.pitch_dimensions[1]],  # bottom-right
            [0, self.pitch_dimensions[1]]   # bottom-left
        ], dtype=np.float32)
        
        # source points from image
        src_corners = np.array(corners, dtype=np.float32)
        
        # calculate homography
        h_matrix, _ = cv2.findHomography(src_corners, pitch_corners, cv2.RANSAC, 5.0)
        
        return h_matrix
    
    def transform_to_2d(self, point):
        """transform image point to 2d pitch coordinates."""
        if self.field_homography is None:
            return None
        
        # convert to homogeneous coordinates
        point_homo = np.array([point[0], point[1], 1.0])
        
        # apply homography
        pitch_point = self.field_homography @ point_homo
        
        # convert back to 2d
        if pitch_point[2] != 0:
            x_2d = pitch_point[0] / pitch_point[2]
            y_2d = pitch_point[1] / pitch_point[2]
            return (x_2d, y_2d)
        
        return None
    
    def draw_detections(self, frame, tracks, frame_num):
        """draw all detections and tracking information on frame."""
        output_frame = frame.copy()
        
        # draw field transformation info if available
        if self.field_corners and len(self.field_corners) >= 4:
            # draw field corners
            for corner in self.field_corners:
                cv2.circle(output_frame, (int(corner[0]), int(corner[1])), 5, (255, 255, 255), -1)
            
            # draw field outline
            pts = np.array(self.field_corners, np.int32)
            cv2.polylines(output_frame, [pts], True, (255, 255, 255), 2)
        
        # draw tracked objects
        for track_id, track_data in tracks.items():
            bbox = track_data['bbox']
            class_name = track_data['class']
            confidence = track_data['confidence']
            
            # get color and label for this class
            class_info = self.classes.get(class_name.split('s')[0], 
                                         {'color': (128, 128, 128), 'label': 'Unknown'})
            color = class_info['color']
            label = class_info['label']
            
            # draw bounding box
            cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # draw label with track id
            label_text = f"{label} #{track_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # label background
            cv2.rectangle(output_frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), 
                         color, -1)
            
            # label text
            cv2.putText(output_frame, label_text, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # draw trajectory
            if track_id in self.tracks and len(self.tracks[track_id]['positions']) > 1:
                positions = list(self.tracks[track_id]['positions'])
                for i in range(len(positions) - 1):
                    cv2.line(output_frame, 
                           (int(positions[i][0]), int(positions[i][1])),
                           (int(positions[i+1][0]), int(positions[i+1][1])),
                           color, 1)
            
            # draw 2d position if homography available
            center = track_data['center']
            pos_2d = self.transform_to_2d(center)
            if pos_2d:
                x_2d, y_2d = pos_2d
                pos_text = f"({x_2d:.1f}m, {y_2d:.1f}m)"
                cv2.putText(output_frame, pos_text,
                           (bbox[0], bbox[3] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # add frame info
        info_text = f"frame: {frame_num} | tracks: {len(tracks)} | field: {'calibrated' if self.field_homography else 'not calibrated'}"
        cv2.putText(output_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame
    
    def process_frame(self, frame, frame_num):
        """process single frame with full detection and tracking pipeline."""
        # detect objects
        detections = self.detect_objects(frame)
        
        # detect field lines for calibration
        if self.field_homography is None and frame_num % 30 == 0:  # try calibration every 30 frames
            self.detect_field_lines(frame)
        
        # update tracks
        tracks = self.update_tracks(detections, frame_num)
        
        # draw results
        output_frame = self.draw_detections(frame, tracks, frame_num)
        
        return output_frame, tracks
    
    def get_spatial_data(self, tracks):
        """extract spatial data from current tracks."""
        spatial_data = []
        
        for track_id, track_data in tracks.items():
            center = track_data['center']
            pos_2d = self.transform_to_2d(center)
            
            spatial_data.append({
                'track_id': track_id,
                'class': track_data['class'],
                'position_2d': pos_2d,
                'position_image': center,
                'confidence': track_data['confidence']
            })
        
        return spatial_data

# import os for model loading
import os
