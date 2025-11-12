from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(
    # Change to your project name
    "shashank-p3ytm").project("football-field-detection-f07vi-e8dgd")

project.version("1").deploy(
    model_type="yolov8", model_path=f"runs\\pose\\train")
