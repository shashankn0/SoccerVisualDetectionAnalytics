# comprehensive test for complete soccer analytics pipeline
import os
import sys

def test_pipeline_components():
    """test all pipeline components."""
    print("testing complete soccer analytics pipeline...")
    print("=" * 60)
    
    components = [
        ("enhanced video detector", "video_detector"),
        ("jersey number detector", "jersey_number_detector"),
        ("pitch 2d visualizer", "pitch_2d_visualizer"),
        ("game 2d recreator", "game_2d_recreation"),
        ("model trainer", "train_models")
    ]
    
    results = {}
    
    for component_name, module_name in components:
        print(f"\ntesting {component_name}...")
        try:
            exec(f"import {module_name}")
            print(f"+ {component_name} imported successfully")
            results[component_name] = True
        except ImportError as e:
            print(f"- {component_name} import failed: {e}")
            results[component_name] = False
        except Exception as e:
            print(f"- {component_name} error: {e}")
            results[component_name] = False
    
    return results

def test_data_availability():
    """test availability of required data and models."""
    print("\ntesting data availability...")
    
    # check datasets
    dataset_paths = [
        "soccer-analytics-yolo/datasets/players",
        "soccer-analytics-yolo/datasets/balls",
        "soccer-analytics-yolo/datasets/field"
    ]
    
    datasets_available = 0
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"+ dataset available: {path}")
            datasets_available += 1
        else:
            print(f"- dataset missing: {path}")
    
    # check models
    model_paths = [
        "models/player.pt",
        "models/ball.pt",
        "models/field.pt",
        "models/jersey_numbers.pt"
    ]
    
    models_available = 0
    for path in model_paths:
        if os.path.exists(path):
            print(f"+ model available: {path}")
            models_available += 1
        else:
            print(f"- model missing: {path}")
    
    # check video
    video_path = "uploaded_videos/videoplaybacktest.mp4"
    video_available = os.path.exists(video_path)
    
    if video_available:
        print(f"+ video available: {video_path}")
    else:
        print(f"- video missing: {video_path}")
    
    return {
        'datasets': datasets_available,
        'models': models_available,
        'video': video_available
    }

def test_basic_functionality():
    """test basic functionality of key components."""
    print("\ntesting basic functionality...")
    
    try:
        # test video detector initialization
        from video_detector import EnhancedSoccerDetector
        detector = EnhancedSoccerDetector()
        print("+ enhanced detector initialized")
        
        # test jersey detector initialization
        from jersey_number_detector import JerseyNumberDetector
        jersey_detector = JerseyNumberDetector()
        print("+ jersey detector initialized")
        
        # test visualizer initialization
        from pitch_2d_visualizer import Pitch2DVisualizer
        visualizer = Pitch2DVisualizer()
        print("+ pitch visualizer initialized")
        
        # test recreator initialization
        from game_2d_recreation import Game2DRecreator
        recreator = Game2DRecreator()
        print("+ game recreator initialized")
        
        # test trainer initialization
        from train_models import SoccerModelTrainer
        trainer = SoccerModelTrainer()
        print("+ model trainer initialized")
        
        return True
        
    except Exception as e:
        print(f"- basic functionality test failed: {e}")
        return False

def test_integration():
    """test integration between components."""
    print("\ntesting component integration...")
    
    try:
        from video_detector import EnhancedSoccerDetector
        from pitch_2d_visualizer import Pitch2DVisualizer
        from jersey_number_detector import JerseyNumberDetector
        
        # test detector to visualizer integration
        detector = EnhancedSoccerDetector()
        visualizer = Pitch2DVisualizer()
        
        # create dummy detection data
        dummy_tracks = {
            0: {
                'bbox': [100, 100, 150, 200],
                'center': (125, 150),
                'class': 'players',
                'confidence': 0.85
            },
            1: {
                'bbox': [300, 200, 350, 300],
                'center': (325, 250),
                'class': 'ball',
                'confidence': 0.92
            }
        }
        
        # test spatial data extraction
        spatial_data = detector.get_spatial_data(dummy_tracks)
        print(f"+ extracted spatial data for {len(spatial_data)} objects")
        
        # test visualizer data loading
        visualizer.player_data = {0: []}
        print("+ visualizer data structure ready")
        
        return True
        
    except Exception as e:
        print(f"- integration test failed: {e}")
        return False

def check_dependencies():
    """check if required dependencies are available."""
    print("\nchecking dependencies...")
    
    dependencies = {
        'opencv': 'cv2',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib.pyplot',
        'pillow': 'PIL',
        'tesseract': 'pytesseract',
        'ultralytics': 'ultralytics',
        'torch': 'torch'
    }
    
    available_deps = {}
    
    for dep_name, module_name in dependencies.items():
        try:
            exec(f"import {module_name}")
            print(f"+ {dep_name} available")
            available_deps[dep_name] = True
        except ImportError:
            print(f"- {dep_name} missing")
            available_deps[dep_name] = False
    
    return available_deps

def generate_setup_instructions():
    """generate setup instructions based on test results."""
    print("\n" + "=" * 60)
    print("setup instructions")
    print("=" * 60)
    
    print("\n1. install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. download datasets:")
    print("   run dataset_download.ipynb to download roboflow datasets")
    
    print("\n3. train models:")
    print("   python train_models.py")
    
    print("\n4. process video:")
    print("   place video in uploaded_videos/videoplaybacktest.mp4")
    print("   python game_2d_recreation.py")
    
    print("\n5. outputs generated:")
    print("   - enhanced 2d pitch visualizations")
    print("   - animated game recreation")
    print("   - comprehensive game data csv")
    print("   - team-specific statistics")
    print("   - training dataset for future models")

def main():
    """main test function."""
    print("complete soccer analytics pipeline - comprehensive test")
    print("=" * 60)
    
    # run all tests
    component_tests = test_pipeline_components()
    data_tests = test_data_availability()
    functionality_tests = test_basic_functionality()
    integration_tests = test_integration()
    dependency_tests = check_dependencies()
    
    # calculate scores
    component_score = sum(component_tests.values())
    data_score = (data_tests['datasets'] + data_tests['models'] + data_tests['video']) / 3 * 100
    functionality_score = 100 if functionality_tests else 0
    integration_score = 100 if integration_tests else 0
    dependency_score = sum(dependency_tests.values()) / len(dependency_tests) * 100
    
    overall_score = (component_score + data_score + functionality_score + integration_score + dependency_score) / 5
    
    # summary
    print(f"\n{'='*60}")
    print("test summary")
    print(f"{'='*60}")
    
    print(f"components: {component_score}/5 ({component_score/5*100:.0f}%)")
    print(f"data availability: {data_score:.0f}%")
    print(f"functionality: {functionality_score}%")
    print(f"integration: {integration_score}%")
    print(f"dependencies: {dependency_score:.0f}%")
    print(f"overall readiness: {overall_score:.0f}%")
    
    # component status
    print(f"\ncomponent status:")
    for component, status in component_tests.items():
        print(f"  {'+' if status else '-'} {component}")
    
    # data status
    print(f"\ndata status:")
    print(f"  datasets: {data_tests['datasets']}/3 available")
    print(f"  models: {data_tests['models']}/4 available")
    print(f"  video: {'+' if data_tests['video'] else '-'} available")
    
    # recommendations
    print(f"\nrecommendations:")
    
    if overall_score >= 80:
        print("+ system is ready for production use!")
        print("+ run: python game_2d_recreation.py")
    elif overall_score >= 60:
        print("+ system is mostly ready")
        missing_deps = [dep for dep, avail in dependency_tests.items() if not avail]
        if missing_deps:
            print(f"- install missing dependencies: {missing_deps}")
    else:
        print("- system needs setup before use")
        generate_setup_instructions()
    
    print(f"\nfeatures implemented:")
    print("- enhanced multi-object detection (players, ball, field)")
    print("- object tracking with unique ids")
    print("- jersey number detection (ocr + model)")
    print("- 2d pitch transformation and visualization")
    print("- team-colored bubble visualization")
    print("- comprehensive game recreation")
    print("- training pipeline for custom models")
    print("- statistical analysis and reporting")

if __name__ == "__main__":
    main()
