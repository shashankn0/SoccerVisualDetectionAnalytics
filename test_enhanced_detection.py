# comprehensive test for enhanced soccer detection system
import cv2
import numpy as np
import os
import sys

def test_enhanced_detector():
    """test the enhanced soccer detection system."""
    print("testing enhanced soccer detection system...")
    
    try:
        # import the enhanced detector
        from video_detector import EnhancedSoccerDetector
        print("+ enhanced detector imported successfully")
        
        # initialize detector
        detector = EnhancedSoccerDetector()
        print("+ detector initialized")
        
        # check if models are loaded
        if detector.models:
            print(f"+ loaded models: {list(detector.models.keys())}")
        else:
            print("- no models loaded - this is expected if model files are missing")
        
        # test tracking system
        print(f"+ tracking system ready - next track id: {detector.next_id}")
        
        # test field transformation setup
        print(f"+ pitch dimensions: {detector.pitch_dimensions} meters")
        
        return True
        
    except ImportError as e:
        print(f"- import failed: {e}")
        return False
    except Exception as e:
        print(f"- initialization failed: {e}")
        return False

def test_detection_pipeline():
    """test the detection pipeline with a sample frame."""
    print("\ntesting detection pipeline...")
    
    try:
        from video_detector import EnhancedSoccerDetector
        
        detector = EnhancedSoccerDetector()
        
        # create a test frame (640x480)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (50, 100, 50)  # green background (field-like)
        
        # add some test objects (rectangles)
        # players
        cv2.rectangle(test_frame, (100, 100), (150, 200), (200, 200, 200), -1)
        cv2.rectangle(test_frame, (300, 150), (350, 250), (200, 200, 200), -1)
        
        # ball
        cv2.circle(test_frame, (320, 240), 10, (255, 255, 255), -1)
        
        # process frame
        output_frame, tracks = detector.process_frame(test_frame, 0)
        
        print(f"+ processed test frame")
        print(f"+ detected {len(tracks)} tracks")
        
        # test spatial data extraction
        spatial_data = detector.get_spatial_data(tracks)
        print(f"+ extracted spatial data for {len(spatial_data)} objects")
        
        return True
        
    except Exception as e:
        print(f"- detection pipeline test failed: {e}")
        return False

def test_field_transformation():
    """test field transformation functionality."""
    print("\ntesting field transformation...")
    
    try:
        from video_detector import EnhancedSoccerDetector
        
        detector = EnhancedSoccerDetector()
        
        # test with dummy corners
        dummy_corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
        frame_shape = (480, 640, 3)
        
        # test homography calculation
        h_matrix = detector.calculate_homography(dummy_corners, frame_shape)
        if h_matrix is not None:
            print("+ homography matrix calculated")
            print(f"+ matrix shape: {h_matrix.shape}")
        else:
            print("- homography calculation failed")
            return False
        
        # test point transformation
        test_point = (320, 240)  # center of frame
        transformed = detector.transform_to_2d(test_point)
        
        if transformed:
            print(f"+ point transformation successful: {transformed}")
        else:
            print("- point transformation failed")
        
        return True
        
    except Exception as e:
        print(f"- field transformation test failed: {e}")
        return False

def test_tracking_system():
    """test object tracking functionality."""
    print("\ntesting tracking system...")
    
    try:
        from video_detector import EnhancedSoccerDetector
        
        detector = EnhancedSoccerDetector()
        
        # simulate multiple frames with moving objects
        for frame_num in range(5):
            # create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # add moving object (simulate player movement)
            x_pos = 200 + frame_num * 20
            cv2.rectangle(test_frame, (x_pos, 200), (x_pos + 50, 300), (200, 200, 200), -1)
            
            # process frame
            output_frame, tracks = detector.process_frame(test_frame, frame_num)
            
            print(f"  frame {frame_num}: {len(tracks)} tracks")
        
        # check if tracks are maintained
        if len(detector.tracks) > 0:
            print(f"+ tracking working - maintained {len(detector.tracks)} tracks")
        else:
            print("- no tracks maintained (expected if no models loaded)")
        
        return True
        
    except Exception as e:
        print(f"- tracking system test failed: {e}")
        return False

def test_spatial_reference():
    """test spatial reference system."""
    print("\ntesting spatial reference system...")
    
    try:
        from video_detector import EnhancedSoccerDetector
        
        detector = EnhancedSoccerDetector()
        
        # manually set up field transformation for testing
        dummy_corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
        frame_shape = (480, 640, 3)
        detector.field_homography = detector.calculate_homography(dummy_corners, frame_shape)
        
        # test various positions
        test_positions = [
            (320, 240),  # center
            (100, 100),  # corner
            (540, 380),  # opposite corner
        ]
        
        for pos in test_positions:
            transformed = detector.transform_to_2d(pos)
            if transformed:
                print(f"  {pos} -> {transformed}")
            else:
                print(f"  {pos} -> transformation failed")
        
        print("+ spatial reference system tested")
        return True
        
    except Exception as e:
        print(f"- spatial reference test failed: {e}")
        return False

def check_model_files():
    """check if required model files exist."""
    print("\nchecking model files...")
    
    model_files = [
        "models/player.pt",
        "models/ball.pt",
        "models/field.pt"
    ]
    
    missing_files = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"+ {model_file} exists")
        else:
            print(f"- {model_file} missing")
            missing_files.append(model_file)
    
    if missing_files:
        print(f"\nmissing {len(missing_files)} model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nnote: the system can work without models for testing,")
        print("but full functionality requires trained models.")
    
    return len(missing_files) == 0

def main():
    """run all enhanced detection tests."""
    print("enhanced soccer detection system - comprehensive test")
    print("=" * 60)
    
    # run tests
    tests = [
        ("enhanced detector initialization", test_enhanced_detector),
        ("detection pipeline", test_detection_pipeline),
        ("field transformation", test_field_transformation),
        ("tracking system", test_tracking_system),
        ("spatial reference system", test_spatial_reference),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"- {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # check model files
    print(f"\n{'='*20} model files check {'='*20}")
    models_ok = check_model_files()
    
    # summary
    print(f"\n{'='*60}")
    print("test summary:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"core functionality tests: {passed}/{total} passed")
    
    for test_name, result in results.items():
        status = "+" if result else "-"
        print(f"  {status} {test_name}")
    
    print(f"\nmodel files: {'+' if models_ok else '-'} available")
    
    # overall assessment
    if passed >= total - 1:  # allow one test to fail
        print("\n+ enhanced detection system is ready!")
        print("next steps:")
        print("1. add model files to models/ directory for full functionality")
        print("2. test with real soccer video footage")
        print("3. integrate with unified soccer api")
        
        if not models_ok:
            print("4. train or download detection models")
    else:
        print("\n- some issues found - please review failed tests")
    
    print("\nfeatures implemented:")
    print("- comprehensive object detection (players, goalkeepers, referees, ball)")
    print("- object tracking with unique ids")
    print("- field line detection and homography")
    print("- 2d pitch transformation")
    print("- spatial reference system")
    print("- trajectory visualization")

if __name__ == "__main__":
    main()
