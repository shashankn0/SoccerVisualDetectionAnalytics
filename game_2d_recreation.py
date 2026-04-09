# complete 2d game recreation system with trained models
import os
import cv2
import numpy as np
import pandas as pd
from video_detector import EnhancedSoccerDetector
from jersey_number_detector import JerseyNumberDetector
from pitch_2d_visualizer import Pitch2DVisualizer
import torch
from ultralytics import YOLO

class Game2DRecreator:
    """complete system for 2d soccer game recreation from video."""
    
    def __init__(self):
        # initialize components
        self.detector = EnhancedSoccerDetector()
        self.jersey_detector = JerseyNumberDetector()
        self.visualizer = Pitch2DVisualizer()
        
        # game data storage
        self.game_frames = {}
        self.current_frame_data = {}
        
        # processing settings
        self.output_dir = '2d_game_recreation'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("+ initialized 2d game recreation system")
    
    def process_video_complete(self, video_path):
        """process video and create complete 2d recreation."""
        print(f"\nprocessing video: {os.path.basename(video_path)}")
        print("=" * 60)
        
        # open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        
        # get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"video info: {width}x{height}, {fps}fps, {total_frames} frames")
        
        frame_count = 0
        processing_interval = 1  # process every frame
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # process frame at intervals
                if frame_count % processing_interval == 0:
                    self.process_single_frame(frame, frame_count, fps)
                
                frame_count += 1
                
                # progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
        
        print(f"\nprocessed {frame_count} frames")
        print(f"extracted {len(self.game_frames)} frames for recreation")
        
        return self.game_frames
    
    def process_single_frame(self, frame, frame_num, fps):
        """process single frame for 2d recreation."""
        # run enhanced detection
        output_frame, tracks = self.detector.process_frame(frame, frame_num)
        
        # extract player bboxes for jersey detection
        player_bboxes = []
        for track_id, track_data in tracks.items():
            if 'players' in track_data['class'] or 'goalkeepers' in track_data['class']:
                player_bboxes.append(track_data['bbox'])
        
        # detect jersey numbers
        jersey_numbers = {}
        if player_bboxes:
            jersey_numbers = self.jersey_detector.batch_detect_numbers(frame, player_bboxes)
        
        # compile frame data
        frame_data = {
            'tracks': tracks,
            'jersey_numbers': jersey_numbers,
            'timestamp': frame_num / fps,
            'field_homography': self.detector.field_homography
        }
        
        self.game_frames[frame_num] = frame_data
        
        # print frame summary
        if frame_num % 50 == 0:
            print(f"  frame {frame_num}: {len(tracks)} objects detected")
    
    def create_enhanced_2d_data(self):
        """create enhanced 2d data with jersey numbers and teams."""
        print("\ncreating enhanced 2d game data...")
        
        enhanced_data = []
        
        for frame_num, frame_data in self.game_frames.items():
            tracks = frame_data['tracks']
            jersey_numbers = frame_data['jersey_numbers']
            timestamp = frame_data['timestamp']
            
            # process each track
            for track_id, track_data in tracks.items():
                class_name = track_data['class']
                bbox = track_data['bbox']
                center = track_data['center']
                confidence = track_data['confidence']
                
                # transform to 2d coordinates
                pos_2d = self.detector.transform_to_2d(center)
                
                if pos_2d:
                    # get jersey number
                    jersey_num = jersey_numbers.get(track_id, None)
                    
                    # determine team based on position
                    team = self.determine_team_from_position(pos_2d, track_id)
                    
                    enhanced_data.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'class': class_name,
                        'jersey_number': jersey_num,
                        'team': team,
                        'x_2d': pos_2d[0],
                        'y_2d': pos_2d[1],
                        'x_image': center[0],
                        'y_image': center[1],
                        'confidence': confidence,
                        'bbox_x1': bbox[0],
                        'bbox_y1': bbox[1],
                        'bbox_x2': bbox[2],
                        'bbox_y2': bbox[3]
                    })
        
        print(f"+ created {len(enhanced_data)} enhanced 2d records")
        return enhanced_data
    
    def determine_team_from_position(self, pos_2d, track_id):
        """determine team assignment based on position and tracking."""
        # use position clustering for team assignment
        if not hasattr(self, 'team_assignments'):
            self.team_assignments = {}
        
        # if track already has team assignment, use it
        if track_id in self.team_assignments:
            return self.team_assignments[track_id]
        
        # simple team assignment based on x-position
        # split pitch in half
        if pos_2d[0] < 52.5:  # half of 105m
            team = 'team_a'
        else:
            team = 'team_b'
        
        # store assignment
        self.team_assignments[track_id] = team
        
        return team
    
    def create_2d_visualizations(self, enhanced_data):
        """create comprehensive 2d visualizations."""
        print("\ncreating 2d pitch visualizations...")
        
        # convert to format expected by visualizer
        viz_data = {}
        
        for record in enhanced_data:
            frame_num = record['frame']
            
            if frame_num not in viz_data:
                viz_data[frame_num] = {
                    'players': [],
                    'ball': None
                }
            
            if record['class'] in ['players', 'goalkeepers']:
                player_info = {
                    'position': (record['x_2d'], record['y_2d']),
                    'jersey_number': record['jersey_number'] or '?',
                    'confidence': record['confidence']
                }
                viz_data[frame_num]['players'].append(player_info)
            
            elif record['class'] == 'ball':
                viz_data[frame_num]['ball'] = {
                    'position': (record['x_2d'], record['y_2d']),
                    'confidence': record['confidence']
                }
        
        # load data into visualizer
        self.visualizer.player_data = viz_data
        
        # create static visualizations for key frames
        key_frames = [0, 100, 200, 300, 400, 500]
        for frame_num in key_frames:
            if frame_num in viz_data:
                output_path = os.path.join(self.output_dir, f'enhanced_frame_{frame_num:04d}.png')
                self.visualizer.create_static_visualization(frame_num, output_path)
        
        # create animation
        anim_output = os.path.join(self.output_dir, 'enhanced_game_animation.gif')
        self.visualizer.create_animation(anim_output, fps=10)
        
        print(f"+ created 2d visualizations in {self.output_dir}")
    
    def export_comprehensive_data(self, enhanced_data):
        """export comprehensive game data for analysis."""
        print("\nexporting comprehensive game data...")
        
        # create dataframe
        df = pd.DataFrame(enhanced_data)
        
        # save main data
        main_output = os.path.join(self.output_dir, 'comprehensive_game_data.csv')
        df.to_csv(main_output, index=False)
        print(f"+ main data exported to: {main_output}")
        
        # create team-specific data
        for team in ['team_a', 'team_b']:
            team_data = df[df['team'] == team]
            if not team_data.empty:
                team_output = os.path.join(self.output_dir, f'{team}_data.csv')
                team_data.to_csv(team_output, index=False)
                print(f"+ {team} data exported to: {team_output}")
        
        # create summary statistics
        self.create_game_summary(df)
        
        return main_output
    
    def create_game_summary(self, df):
        """create comprehensive game summary statistics."""
        summary_path = os.path.join(self.output_dir, 'game_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("soccer game analysis summary\n")
            f.write("=" * 50 + "\n\n")
            
            # basic stats
            total_frames = df['frame'].nunique()
            total_players = df[df['class'].isin(['players', 'goalkeepers'])]['track_id'].nunique()
            
            f.write(f"total frames analyzed: {total_frames}\n")
            f.write(f"unique players tracked: {total_players}\n")
            
            # team statistics
            team_stats = df.groupby('team').agg({
                'track_id': 'nunique',
                'confidence': 'mean'
            }).round(3)
            
            f.write("\nteam statistics:\n")
            f.write("-" * 20 + "\n")
            
            for team, stats in team_stats.iterrows():
                f.write(f"{team}:\n")
                f.write(f"  players: {stats['track_id']}\n")
                f.write(f"  avg confidence: {stats['confidence']}\n")
            
            # position analysis
            f.write("\nposition analysis:\n")
            f.write("-" * 20 + "\n")
            
            for axis in ['x_2d', 'y_2d']:
                pos_stats = df[axis].describe()
                f.write(f"{axis} statistics:\n")
                f.write(f"  mean: {pos_stats['mean']:.2f}m\n")
                f.write(f"  std: {pos_stats['std']:.2f}m\n")
                f.write(f"  min: {pos_stats['min']:.2f}m\n")
                f.write(f"  max: {pos_stats['max']:.2f}m\n")
            
            # jersey number detection rate
            detected_numbers = df[df['jersey_number'].notna() & (df['jersey_number'] != '?')]
            detection_rate = (len(detected_numbers) / total_players) * 100 if total_players > 0 else 0
            
            f.write(f"\njersey number detection rate: {detection_rate:.1f}%\n")
            f.write(f"players with detected numbers: {len(detected_numbers)}\n")
        
        print(f"+ game summary saved to: {summary_path}")
    
    def create_training_dataset(self, enhanced_data):
        """create training dataset from processed game data."""
        print("\ncreating training dataset...")
        
        # create yolo format dataset
        dataset_dir = os.path.join(self.output_dir, 'training_dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # group by frame
        frame_groups = enhanced_data.groupby('frame')
        
        for frame_num, frame_data in frame_groups:
            # create yolo labels
            label_lines = []
            
            for _, record in frame_data.iterrows():
                if record['class'] in ['players', 'goalkeepers']:
                    # convert to yolo format
                    x_center = record['x_image']
                    y_center = record['y_image']
                    width = record['bbox_x2'] - record['bbox_x1']
                    height = record['bbox_y2'] - record['bbox_y1']
                    
                    # normalize to 0-1
                    img_h = 480  # assume standard height
                    img_w = 640  # assume standard width
                    
                    x_norm = x_center / img_w
                    y_norm = y_center / img_h
                    w_norm = width / img_w
                    h_norm = height / img_h
                    
                    # class id (0 for players)
                    class_id = 0
                    
                    label_line = f"{class_id} {x_norm} {y_norm} {w_norm} {h_norm}"
                    label_lines.append(label_line)
            
            # save label file
            if label_lines:
                label_path = os.path.join(labels_dir, f'frame_{frame_num:06d}.txt')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
        
        print(f"+ training dataset created in: {dataset_dir}")
        return dataset_dir

def main():
    """main function for complete 2d game recreation."""
    print("2d soccer game recreation system")
    print("=" * 60)
    
    # check for video file
    video_path = "uploaded_videos/videoplaybacktest.mp4"
    
    if not os.path.exists(video_path):
        print(f"- video file not found: {video_path}")
        print("please place a soccer video in uploaded_videos/ directory")
        return
    
    # initialize recreator
    recreator = Game2DRecreator()
    
    try:
        # process video
        game_frames = recreator.process_video_complete(video_path)
        
        if game_frames:
            # create enhanced 2d data
            enhanced_data = recreator.create_enhanced_2d_data()
            
            # create visualizations
            recreator.create_2d_visualizations(enhanced_data)
            
            # export data
            recreator.export_comprehensive_data(enhanced_data)
            
            # create training dataset
            recreator.create_training_dataset(enhanced_data)
            
            print(f"\n{'='*60}")
            print("2d game recreation completed successfully!")
            print(f"{'='*60}")
            print(f"outputs saved to: {recreator.output_dir}")
            print("\noutputs created:")
            print("- enhanced 2d pitch visualizations")
            print("- animated game recreation")
            print("- comprehensive game data csv")
            print("- team-specific data files")
            print("- game summary statistics")
            print("- training dataset for future model training")
        
    except Exception as e:
        print(f"- recreation failed: {e}")
        raise

if __name__ == "__main__":
    main()
