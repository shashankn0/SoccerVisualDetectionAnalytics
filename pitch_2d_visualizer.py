# 2d pitch visualization with team-colored bubbles and jersey numbers
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
from collections import defaultdict
import math

class Pitch2DVisualizer:
    """comprehensive 2d pitch visualization system for soccer games."""
    
    def __init__(self):
        # pitch dimensions (standard soccer pitch in meters)
        self.pitch_length = 105
        self.pitch_width = 68
        
        # team colors
        self.team_colors = {
            'team_a': '#FF6B6B',  # red
            'team_b': '#4169E1',  # blue
            'referee': '#FFD700',   # gold
            'goalkeeper': '#FF8C00'  # orange
        }
        
        # player tracking data
        self.player_data = defaultdict(list)
        self.ball_data = []
        self.current_frame = 0
        
        # visualization settings
        self.bubble_size = 1.5  # meters
        self.jersey_size = 0.8
        self.trail_length = 20  # frames
        
        # setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.setup_pitch()
    
    def setup_pitch(self):
        """setup the soccer pitch with proper markings."""
        self.ax.clear()
        
        # set pitch boundaries
        self.ax.set_xlim(-5, self.pitch_length + 5)
        self.ax.set_ylim(-5, self.pitch_width + 5)
        self.ax.set_aspect('equal')
        
        # draw grass
        grass = patches.Rectangle((0, 0), self.pitch_length, self.pitch_width, 
                              linewidth=0, facecolor='#2E7D32', alpha=0.3)
        self.ax.add_patch(grass)
        
        # draw pitch lines
        # outer boundary
        outer_line = patches.Rectangle((0, 0), self.pitch_length, self.pitch_width, 
                                linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(outer_line)
        
        # center line
        self.ax.plot([self.pitch_length/2, self.pitch_length/2], [0, self.pitch_width], 
                    'white', linewidth=2)
        
        # center circle
        center_circle = patches.Circle((self.pitch_length/2, self.pitch_width/2), 9.15, 
                                   linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(center_circle)
        
        # penalty areas
        penalty_width = 16.5
        penalty_length = 40.3
        
        # left penalty area
        left_penalty = patches.Rectangle((0, (self.pitch_width - penalty_width)/2), 
                                    penalty_length, penalty_width, 
                                    linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(left_penalty)
        
        # right penalty area
        right_penalty = patches.Rectangle((self.pitch_length - penalty_length, (self.pitch_width - penalty_width)/2), 
                                     penalty_length, penalty_width, 
                                     linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(right_penalty)
        
        # goals
        goal_width = 7.32
        goal_depth = 2.44
        
        # left goal
        left_goal = patches.Rectangle((-goal_depth, (self.pitch_width - goal_width)/2), 
                                 goal_depth, goal_width, 
                                 linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(left_goal)
        
        # right goal
        right_goal = patches.Rectangle((self.pitch_length, (self.pitch_width - goal_width)/2), 
                                  goal_depth, goal_width, 
                                  linewidth=2, facecolor='none', edgecolor='white')
        self.ax.add_patch(right_goal)
        
        # labels
        self.ax.set_xlabel('Distance (meters)', fontsize=12)
        self.ax.set_ylabel('Width (meters)', fontsize=12)
        self.ax.set_title('2D Soccer Pitch Visualization', fontsize=16, fontweight='bold')
        
        # add grid
        self.add_pitch_grid()
    
    def add_pitch_grid(self):
        """add reference grid to pitch."""
        grid_spacing = 10  # meters
        
        # vertical lines
        for x in range(0, self.pitch_length + 1, grid_spacing):
            self.ax.plot([x, x], [0, self.pitch_width], 'gray', 
                        linewidth=0.5, alpha=0.3)
        
        # horizontal lines
        for y in range(0, self.pitch_width + 1, grid_spacing):
            self.ax.plot([0, self.pitch_length], [y, y], 'gray', 
                        linewidth=0.5, alpha=0.3)
    
    def detect_team_assignment(self, player_positions):
        """assign players to teams based on position clustering."""
        if len(player_positions) < 2:
            return {}
        
        # simple clustering: split by x-position median
        x_positions = [pos[0] for pos in player_positions]
        median_x = np.median(x_positions)
        
        team_assignments = {}
        for i, pos in enumerate(player_positions):
            team = 'team_a' if pos[0] < median_x else 'team_b'
            team_assignments[i] = team
        
        return team_assignments
    
    def extract_jersey_number(self, bbox, frame):
        """extract jersey number from player bounding box."""
        # this is a simplified implementation
        # in practice, you'd use ocr or a dedicated number detection model
        
        x1, y1, x2, y2 = bbox
        
        # extract jersey area (upper part of bbox)
        jersey_y1 = y1
        jersey_y2 = y1 + int((y2 - y1) * 0.4)  # top 40%
        jersey_x1 = x1
        jersey_x2 = x2
        
        # for now, return random numbers for demonstration
        # in real implementation, you'd use ocr here
        import random
        jersey_number = random.randint(1, 99)
        
        return jersey_number
    
    def load_detection_data(self, csv_path):
        """load detection data from csv file."""
        if not os.path.exists(csv_path):
            print(f"- detection data file not found: {csv_path}")
            return False
        
        try:
            df = pd.read_csv(csv_path)
            print(f"+ loaded {len(df)} detection records")
            
            # group by frame
            for frame_num in df['frame'].unique():
                frame_data = df[df['frame'] == frame_num]
                
                # extract players and ball
                players = []
                ball = None
                
                for _, row in frame_data.iterrows():
                    if row['class_id'] == 0:  # assuming class 0 is player
                        center_x = (row['x1'] + row['x2']) / 2
                        center_y = (row['y1'] + row['y2']) / 2
                        
                        # extract jersey number (simplified)
                        jersey_num = self.extract_jersey_number(
                            [row['x1'], row['y1'], row['x2'], row['y2']], 
                            None
                        )
                        
                        players.append({
                            'position': (center_x, center_y),
                            'jersey_number': jersey_num,
                            'confidence': row['confidence']
                        })
                    
                    elif row['class_id'] == 1:  # assuming class 1 is ball
                        ball_x = (row['x1'] + row['x2']) / 2
                        ball_y = (row['y1'] + row['y2']) / 2
                        ball = {'position': (ball_x, ball_y), 'confidence': row['confidence']}
                
                self.player_data[frame_num] = players
                if ball:
                    self.ball_data.append({'frame': frame_num, 'position': ball['position']})
            
            print(f"+ processed {len(self.player_data)} frames")
            return True
            
        except Exception as e:
            print(f"- failed to load detection data: {e}")
            return False
    
    def transform_to_2d_coordinates(self, image_position, image_shape):
        """transform image coordinates to 2d pitch coordinates."""
        # simplified transformation - assumes full pitch view
        # in practice, you'd use homography from the enhanced detector
        
        h, w = image_shape[:2]
        x, y = image_position
        
        # normalize to [0, 1] range
        x_norm = x / w
        y_norm = y / h
        
        # transform to pitch coordinates
        x_2d = x_norm * self.pitch_length
        y_2d = y_norm * self.pitch_width
        
        return (x_2d, y_2d)
    
    def draw_frame(self, frame_num):
        """draw single frame of 2d visualization."""
        self.setup_pitch()
        
        if frame_num not in self.player_data:
            return
        
        players = self.player_data[frame_num]
        
        # assign teams
        player_positions = [p['position'] for p in players]
        team_assignments = self.detect_team_assignment(player_positions)
        
        # draw player trails
        for i, player in enumerate(players):
            if i in team_assignments:
                team = team_assignments[i]
                color = self.team_colors[team]
                
                # draw trail
                trail_positions = []
                for past_frame in range(max(0, frame_num - self.trail_length), frame_num):
                    if past_frame in self.player_data:
                        for j, past_player in enumerate(self.player_data[past_frame]):
                            if j == i:  # same player
                                trail_positions.append(past_player['position'])
                
                if len(trail_positions) > 1:
                    trail_x = [pos[0] for pos in trail_positions]
                    trail_y = [pos[1] for pos in trail_positions]
                    self.ax.plot(trail_x, trail_y, color=color, alpha=0.3, linewidth=1)
        
        # draw players as colored bubbles
        for i, player in enumerate(players):
            pos = player['position']
            jersey_num = player['jersey_number']
            
            if i in team_assignments:
                team = team_assignments[i]
                color = self.team_colors[team]
                
                # draw player bubble
                bubble = patches.Circle(pos, self.bubble_size, 
                                   facecolor=color, alpha=0.7, edgecolor='white', linewidth=2)
                self.ax.add_patch(bubble)
                
                # draw jersey number
                self.ax.text(pos[0], pos[1], str(jersey_num), 
                            fontsize=10, fontweight='bold', 
                            ha='center', va='center', color='white')
                
                # draw confidence
                conf_text = f"{player['confidence']:.2f}"
                self.ax.text(pos[0], pos[1] - self.bubble_size - 0.5, conf_text, 
                            fontsize=8, ha='center', va='top')
        
        # draw ball
        ball_frame_data = [b for b in self.ball_data if b['frame'] == frame_num]
        if ball_frame_data:
            ball_pos = ball_frame_data[0]['position']
            ball = patches.Circle(ball_pos, 0.5, 
                               facecolor='white', edgecolor='black', linewidth=2)
            self.ax.add_patch(ball)
        
        # add frame info
        self.ax.text(2, self.pitch_width - 2, f'Frame: {frame_num}', 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # add legend
        self.add_legend(team_assignments)
    
    def add_legend(self, team_assignments):
        """add legend showing team colors."""
        used_teams = set(team_assignments.values())
        
        legend_elements = []
        for team in used_teams:
            if team in self.team_colors:
                legend_elements.append(
                    patches.Patch(facecolor=self.team_colors[team], 
                                 edgecolor='white', label=team.replace('_', ' ').title())
                )
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', 
                        bbox_to_anchor=(1.15, 1))
    
    def create_static_visualization(self, frame_num, output_path):
        """create static visualization for specific frame."""
        self.draw_frame(frame_num)
        
        # save figure
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"+ saved frame {frame_num} visualization to: {output_path}")
    
    def create_animation(self, output_path, fps=10):
        """create animated visualization from detection data."""
        frames = sorted(self.player_data.keys())
        
        if not frames:
            print("- no frame data available for animation")
            return
        
        print(f"+ creating animation with {len(frames)} frames")
        
        def animate(frame_idx):
            if frame_idx < len(frames):
                frame_num = frames[frame_idx]
                self.draw_frame(frame_num)
                return self.ax.patches + self.ax.lines + self.ax.texts
            return []
        
        # create animation
        anim = FuncAnimation(self.fig, animate, frames=len(frames), 
                           interval=1000/fps, blit=False)
        
        # save animation
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=fps)
        else:
            # default to gif
            anim.save(output_path + '.gif', writer='pillow', fps=fps)
        
        print(f"+ animation saved to: {output_path}")
    
    def export_2d_coordinates(self, output_path):
        """export 2d coordinates to csv."""
        export_data = []
        
        for frame_num in sorted(self.player_data.keys()):
            players = self.player_data[frame_num]
            
            for i, player in enumerate(players):
                pos = player['position']
                export_data.append({
                    'frame': frame_num,
                    'player_id': i,
                    'jersey_number': player['jersey_number'],
                    'x_2d': pos[0],
                    'y_2d': pos[1],
                    'confidence': player['confidence']
                })
        
        # add ball data
        for ball_data in self.ball_data:
            export_data.append({
                'frame': ball_data['frame'],
                'player_id': -1,  # indicate ball
                'jersey_number': 'BALL',
                'x_2d': ball_data['position'][0],
                'y_2d': ball_data['position'][1],
                'confidence': 1.0
            })
        
        # save to csv
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        print(f"+ exported 2d coordinates to: {output_path}")

def main():
    """main function for 2d visualization."""
    visualizer = Pitch2DVisualizer()
    
    # example usage with detection data
    detection_csv = 'detection_results.csv'
    
    if os.path.exists(detection_csv):
        print("loading detection data...")
        if visualizer.load_detection_data(detection_csv):
            
            # create visualizations for sample frames
            output_dir = '2d_visualizations'
            os.makedirs(output_dir, exist_ok=True)
            
            # static visualizations
            frames_to_visualize = [0, 50, 100, 150]  # sample frames
            for frame_num in frames_to_visualize:
                if frame_num in visualizer.player_data:
                    output_path = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
                    visualizer.create_static_visualization(frame_num, output_path)
            
            # export 2d coordinates
            coord_output = os.path.join(output_dir, '2d_coordinates.csv')
            visualizer.export_2d_coordinates(coord_output)
            
            # create animation
            anim_output = os.path.join(output_dir, 'game_animation.gif')
            visualizer.create_animation(anim_output, fps=5)
            
            print(f"\n+ 2d visualization complete!")
            print(f"+ outputs saved to: {output_dir}")
        else:
            print("- failed to load detection data")
    else:
        print(f"- detection data file not found: {detection_csv}")
        print("run video detection first to generate detection data")

if __name__ == "__main__":
    main()
