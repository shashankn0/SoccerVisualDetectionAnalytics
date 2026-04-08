# simple integration test for the unified soccer system
import os
import sys

def test_imports():
    """test if all required modules can be imported."""
    print("testing imports...")
    
    try:
        # test basic imports
        import cv2
        print("+ opencv imported")
        
        import pandas as pd
        print("+ pandas imported")
        
        # test ml imports
        from ultralytics import YOLO
        print("+ ultralytics imported")
        
        import torch
        print("+ pytorch imported")
        
        # test web framework imports
        try:
            from fastapi import FastAPI
            print("+ fastapi imported")
        except ImportError:
            print("- fastapi not found - install with: pip install fastapi")
        
        # test database imports
        try:
            import sqlalchemy
            print("+ sqlalchemy imported")
        except ImportError:
            print("- sqlalchemy not found - install with: pip install sqlalchemy")
        
        # test our modules
        try:
            import classifier
            print("+ classifier module imported")
        except ImportError as e:
            print(f"- classifier import failed: {e}")
        
        try:
            import database
            print("+ database module imported")
        except ImportError as e:
            print(f"- database import failed: {e}")
        
        try:
            import api_client
            print("+ api_client module imported")
        except ImportError as e:
            print(f"- api_client import failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"- import failed: {e}")
        return False

def test_file_structure():
    """test if required files and directories exist."""
    print("\ntesting file structure...")
    
    required_dirs = [
        "uploaded_videos",
        "models"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"+ {dir_name}/ directory exists")
        else:
            print(f"- {dir_name}/ directory missing - creating it")
            os.makedirs(dir_name, exist_ok=True)
    
    # check for model files
    model_files = [
        "models/player.pt",
        "models/ball.pt", 
        "models/field.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"+ {model_file} exists")
        else:
            print(f"- {model_file} missing - you'll need to add your model files")
    
    # check for config files
    if os.path.exists(".env.example"):
        print("+ .env.example exists")
    else:
        print("- .env.example missing - you may need to create it")
    
    if os.path.exists("requirements.txt"):
        print("+ requirements.txt exists")
    else:
        print("- requirements.txt missing")

def test_basic_functionality():
    """test basic functionality of individual components."""
    print("\ntesting basic functionality...")
    
    try:
        # test classifier
        from classifier import EventClassifier
        classifier = EventClassifier()
        
        # test event classification
        test_event = {
            "type": "goal",
            "player": {"name": "test player"},
            "team": {"name": "test team"},
            "minute": 45
        }
        
        event_type = classifier.classify(test_event)
        if event_type:
            print(f"+ event classifier works: {event_type}")
        else:
            print("- event classifier failed")
        
        # test api client
        from api_client import SoccerAPIClient
        client = SoccerAPIClient()
        print("+ api client initialized")
        
        return True
        
    except Exception as e:
        print(f"- functionality test failed: {e}")
        return False

def main():
    """run all integration tests."""
    print("unified soccer system - integration test")
    print("=" * 50)
    
    # run tests
    imports_ok = test_imports()
    test_file_structure()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("test summary:")
    
    if imports_ok and functionality_ok:
        print("+ core system is ready")
        print("next steps:")
        print("1. install missing dependencies: pip install -r requirements.txt")
        print("2. setup mysql database")
        print("3. configure .env file")
        print("4. add your model files to models/ directory")
        print("5. run: python unified_soccer_api.py")
    else:
        print("- some issues found - please fix before running the system")
    
    print("\nfor full functionality, ensure you have:")
    print("- mysql database running")
    print("- model files in models/ directory")
    print("- .env file configured with database and api settings")

if __name__ == "__main__":
    main()
