# jersey number detection using ocr and specialized models
import cv2
import numpy as np
import pytesseract
import torch
from ultralytics import YOLO
import os
from PIL import Image
import re

class JerseyNumberDetector:
    """detect and read jersey numbers from player detections."""
    
    def __init__(self):
        # ocr configuration
        self.tesseract_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        
        # jersey number model (if available)
        self.number_model = None
        self.load_number_model()
        
        # preprocessing parameters
        self.min_digit_size = 10
        self.max_digit_size = 50
        self.confidence_threshold = 0.6
    
    def load_number_model(self):
        """load specialized jersey number detection model."""
        model_path = 'models/jersey_numbers.pt'
        if os.path.exists(model_path):
            try:
                self.number_model = YOLO(model_path)
                print("+ loaded jersey number detection model")
                return True
            except Exception as e:
                print(f"- failed to load jersey number model: {e}")
                return False
        else:
            print("- jersey number model not found, using ocr fallback")
            return False
    
    def preprocess_jersey_area(self, frame, bbox):
        """preprocess jersey area for better ocr accuracy."""
        x1, y1, x2, y2 = bbox
        
        # extract jersey area (upper portion of player bbox)
        jersey_height = int((y2 - y1) * 0.4)  # top 40%
        jersey_y1 = y1
        jersey_y2 = y1 + jersey_height
        jersey_x1 = x1
        jersey_x2 = x2
        
        # crop jersey area
        jersey_area = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        if jersey_area.size == 0:
            return None
        
        # enhance contrast and prepare for ocr
        # convert to grayscale
        gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)
        
        # increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=8)
        enhanced = clahe.apply(gray)
        
        # threshold to get binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # remove noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_numbers_with_model(self, frame, bbox):
        """detect jersey numbers using specialized model."""
        if self.number_model is None:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # extract jersey area
        jersey_area = frame[y1:y2, x1:x2]
        
        # run number detection
        results = self.number_model.predict(jersey_area, conf=0.5, verbose=False)[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            # get the most confident detection
            best_box = results.boxes[results.conf.argmax().cpu().numpy()]
            
            # extract number using ocr on detected region
            number_bbox = best_box.xyxy[0].cpu().numpy().astype(int)
            number_region = jersey_area[number_bbox[1]:number_bbox[3], number_bbox[0]:number_bbox[2]]
            
            if number_region.size[0] > 0 and number_region.size[1] > 0:
                number = self.extract_number_with_ocr(number_region)
                return number
        
        return None
    
    def extract_number_with_ocr(self, jersey_image):
        """extract jersey number using ocr."""
        try:
            # resize for better ocr accuracy
            if jersey_image.size[0] > 0 and jersey_image.size[1] > 0:
                scale_factor = max(100 / jersey_image.size[0], 100 / jersey_image.size[1])
                new_width = int(jersey_image.size[0] * scale_factor)
                new_height = int(jersey_image.size[1] * scale_factor)
                resized = cv2.resize(jersey_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                resized = jersey_image
            
            # apply additional preprocessing
            # gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(resized, (3,3), 0)
            
            # adaptive threshold
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
            
            # convert to PIL for tesseract
            pil_image = Image.fromarray(thresh)
            
            # ocr with custom configuration
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            # clean and validate the result
            cleaned_text = self.clean_ocr_text(text)
            
            if cleaned_text and cleaned_text.isdigit() and len(cleaned_text) <= 2:
                return int(cleaned_text)
            
        except Exception as e:
            print(f"- ocr failed: {e}")
        
        return None
    
    def clean_ocr_text(self, text):
        """clean and validate ocr text."""
        if not text:
            return None
        
        # remove common ocr errors
        cleaned = re.sub(r'[^0-9]', '', text.strip())
        
        # validate reasonable jersey numbers
        if cleaned and cleaned.isdigit():
            num = int(cleaned)
            if 1 <= num <= 99:  # reasonable jersey number range
                return cleaned
        
        return None
    
    def detect_jersey_number(self, frame, bbox, use_model=True):
        """main method to detect jersey number from player bbox."""
        # try specialized model first
        if use_model and self.number_model:
            number = self.detect_numbers_with_model(frame, bbox)
            if number is not None:
                return number
        
        # fallback to ocr
        processed_jersey = self.preprocess_jersey_area(frame, bbox)
        if processed_jersey is not None:
            number = self.extract_number_with_ocr(processed_jersey)
            return number
        
        return None
    
    def batch_detect_numbers(self, frame, player_bboxes):
        """detect jersey numbers for multiple players."""
        results = {}
        
        for i, bbox in enumerate(player_bboxes):
            number = self.detect_jersey_number(frame, bbox)
            results[i] = number
        
        return results
    
    def visualize_number_detection(self, frame, bbox, number, confidence=None):
        """visualize jersey number detection on frame."""
        if number is None:
            return frame
        
        x1, y1, x2, y2 = bbox
        
        # highlight jersey area
        jersey_height = int((y2 - y1) * 0.4)
        jersey_y1 = y1
        jersey_y2 = y1 + jersey_height
        
        # draw jersey area highlight
        cv2.rectangle(frame, (x1, jersey_y1), (x2, jersey_y2), (0, 255, 255), 1)
        
        # draw detected number
        text = f"#{number}"
        if confidence is not None:
            text += f" ({confidence:.2f})"
        
        # text background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, jersey_y1 - text_size[1] - 5), 
                     (x1 + text_size[0], jersey_y1), (0, 255, 0), -1)
        
        # text
        cv2.putText(frame, text, (x1, jersey_y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

def test_jersey_detection():
    """test jersey number detection system."""
    print("testing jersey number detection...")
    
    detector = JerseyNumberDetector()
    
    # create test frame with simulated jersey numbers
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 100, 50)  # green background
    
    # add simulated players with jersey numbers
    test_players = [
        {'bbox': (100, 100, 200, 250), 'number': '10'},
        {'bbox': (300, 150, 400, 350), 'number': '23'},
        {'bbox': (450, 200, 550, 400), 'number': '7'}
    ]
    
    for player in test_players:
        x1, y1, x2, y2 = player['bbox']
        # draw player
        cv2.rectangle(test_frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
        
        # add jersey number text (simulating printed number)
        jersey_y = y1 + int((y2 - y1) * 0.2)
        jersey_x = (x1 + x2) // 2
        cv2.putText(test_frame, player['number'], (jersey_x, jersey_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # test detection
        detected_number = detector.detect_jersey_number(test_frame, player['bbox'])
        print(f"  expected: {player['number']}, detected: {detected_number}")
        
        # visualize detection
        test_frame = detector.visualize_number_detection(test_frame, player['bbox'], detected_number)
    
    # save test result
    cv2.imwrite('jersey_detection_test.jpg', test_frame)
    print("+ jersey detection test completed: jersey_detection_test.jpg")

if __name__ == "__main__":
    test_jersey_detection()
