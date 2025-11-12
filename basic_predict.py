from ultralytics import YOLO
import cv2
import os
import gc
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# Configure safe globals for model loading
add_safe_globals([DetectionModel])

# Configuration
VIDEO_PATH = "uploaded_videos/videoplaybacktest.mp4"
MODEL_PATHS = {
    "players": "models/player.pt",
    "ball": "models/ball.pt",
    "field": "models/field.pt"
}
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = "runs/detect/combined_predict"
SKIP_SECONDS = 0  # Set to skip initial seconds if needed

# Color scheme for different detections (BGR format for OpenCV)
COLORS = {
    "players": (0, 255, 0),    # Green for players
    "ball": (0, 0, 255),       # Red for ball
    "field": (255, 0, 0)       # Blue for field
}

# Label names
LABELS = {
    "players": "Player",
    "ball": "Ball",
    "field": "Field"
}


def load_models():
    """Load all YOLO models at once."""
    print("Loading models...")
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            models[name] = YOLO(path)
            print(f"✓ Loaded {name} model")
        except Exception as e:
            raise RuntimeError(f"Failed to load {name} model: {e}")
    return models


def draw_detections(frame, results, model_name):
    """Draw detections from a single model on the frame."""
    if results.boxes is None or len(results.boxes) == 0:
        return frame
    
    color = COLORS[model_name]
    label = LABELS[model_name]
    
    # Get boxes, confidences, and classes
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_text = f"{label} {conf:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return frame


def process_video(video_path, models, output_dir):
    """Process video once with all models and create combined visualization."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Skip initial frames if specified
    skip_frames = int(fps * SKIP_SECONDS)
    if skip_frames > 0:
        print(f"Skipping first {skip_frames} frames (~{SKIP_SECONDS} seconds)...")
        for _ in range(skip_frames):
            cap.read()
        total_frames -= skip_frames
    
    # Setup video writer
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Output: {output_path}\n")
    
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run all models on the same frame
            frame_with_detections = frame.copy()
            
            for model_name, model in models.items():
                # Run prediction
                results = model.predict(
                    frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device='cpu'  # Use CPU to avoid GPU memory issues
                )[0]
                
                # Draw detections
                frame_with_detections = draw_detections(
                    frame_with_detections,
                    results,
                    model_name
                )
            
            # Write frame to output video
            out.write(frame_with_detections)
            frame_count += 1
            processed_count += 1
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (processed_count / total_frames) * 100
                print(f"Progress: {processed_count}/{total_frames} frames ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        raise
    finally:
        # Cleanup
        cap.release()
        out.release()
        print(f"\n✓ Processing complete!")
        print(f"✓ Processed {processed_count} frames")
        print(f"✓ Output saved to: {output_path}")


def main():
    """Main function to run the optimized prediction pipeline."""
    try:
        # Validate video path
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
        
        # Load all models
        models = load_models()
        
        # Process video with all models
        process_video(VIDEO_PATH, models, OUTPUT_DIR)
        
        # Cleanup models
        print("\nCleaning up models...")
        for model_name in models:
            del models[model_name]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
