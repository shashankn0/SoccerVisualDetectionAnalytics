from roboflow import Roboflow

# Use your private API key
rf = Roboflow(api_key="XgThuNgb3yfLih6r9k7K")

# Download RoboFlow soccer dataset for training custom YOLO model

# Download the football-players-detection dataset
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")

# Download dataset in YOLO format
dataset = project.version(3).download("yolov8")

print("Dataset downloaded successfully!")
print(f"Dataset location: {dataset.location}")
