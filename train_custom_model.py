from ultralytics import YOLO
import os

# Path to the downloaded RoboFlow dataset
dataset_path = "football-players-detection-3"
data_yaml = os.path.join(dataset_path, "data.yaml")

# Load pre-trained YOLOv8s model
model = YOLO("yolov8s.pt")

# Fine-tune on the soccer dataset
print(f"Training on dataset: {data_yaml}")
print("This may take 30-60 minutes depending on your hardware...")

results = model.train(
    data=data_yaml,
    epochs=50,  # Adjust based on your needs
    imgsz=640,
    batch=16,   # Adjust based on your GPU/CPU
    device='cpu',  # Change to 'cuda' if you have GPU
    name='soccer_custom',
    patience=10,  # Early stopping if no improvement
    plots=True,
    save=True
)

print("\nTraining complete!")
print("Custom model saved to: runs/detect/soccer_custom/weights/best.pt")
print("Use this model in process_video.py by changing:")
print("  model = YOLO('yolov8s.pt')")
print("to:")
print("  model = YOLO('runs/detect/soccer_custom/weights/best.pt')")
