"""
vision/simple_ai_detector.py
==========================
Simple AI fallback detector using rule-based approach.
No model downloads required - uses computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Dict
import time

class SimpleAIDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.min_interval = 15.0  # Increased from 5.0 to reduce frequency
        self.enabled = True
        
        # Simple object detection using contour analysis
        self.contour_threshold = 50
        self.min_area = 1000
        self.max_area = 50000
    
    def is_ready(self) -> bool:
        return self.enabled
    
    def should_trigger_fallback(self, yolo_detections: List[Dict]) -> bool:
        if not self.is_ready():
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time
        
        # Only check timing logic, no debug spam
        if time_since_last < self.min_interval:
            return False
        
        # Only trigger if YOLO found absolutely NO objects
        if len(yolo_detections) == 0:
            return True
        
        return False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Simple object detection using computer vision techniques"""
        if not self.is_ready():
            return []
        
        try:
            objects = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Analyze shape characteristics
                    aspect_ratio = w / h if h > 0 else 1
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2) if cv2.arcLength(contour, True) > 0 else 0
                    
                    # Classify based on shape
                    obj_type = self._classify_shape(aspect_ratio, circularity, area)
                    
                    if obj_type:
                        objects.append({
                            'class': obj_type,
                            'confidence': 0.6,  # Fixed confidence for simple detector
                            'bbox': (x, y, x + w, y + h),
                            'source': 'ai'
                        })
            
            # Limit to top 5 objects
            objects = objects[:5]
            
            self.last_detection_time = time.time()
            return objects
            
        except Exception as e:
            print(f"[ERROR] Simple AI detection failed: {e}")
            return []
    
    def _classify_shape(self, aspect_ratio: float, circularity: float, area: float) -> str:
        """Classify object based on shape characteristics"""
        
        # Person-like (tall, not too circular)
        if aspect_ratio > 0.3 and aspect_ratio < 0.7 and circularity < 0.5:
            return "person"
        
        # Chair/table (medium aspect ratio, moderate circularity)
        elif aspect_ratio > 0.7 and aspect_ratio < 1.5 and circularity < 0.7:
            if area > 10000:
                return "table"
            else:
                return "chair"
        
        # Circular objects
        elif circularity > 0.7:
            if area > 5000:
                return "plate"
            else:
                return "cup"
        
        # Large rectangular objects
        elif aspect_ratio > 0.8 and aspect_ratio < 1.2 and area > 15000:
            return "screen"
        
        # Small objects
        elif area < 3000:
            return "object"
        
        return None
    
    def describe_scene(self, frame: np.ndarray) -> str:
        """Simple scene description"""
        objects = self.detect_objects(frame)
        
        if not objects:
            return "I don't see any distinct objects in this scene."
        
        # Count object types
        object_counts = {}
        for obj in objects:
            obj_type = obj['class']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Create description
        descriptions = []
        for obj_type, count in object_counts.items():
            if count == 1:
                descriptions.append(f"a {obj_type}")
            else:
                descriptions.append(f"{count} {obj_type}s")
        
        if len(descriptions) == 1:
            return f"I see {descriptions[0]} in this scene."
        elif len(descriptions) == 2:
            return f"I see {descriptions[0]} and {descriptions[1]} in this scene."
        else:
            return f"I see {', '.join(descriptions[:-1])}, and {descriptions[-1]} in this scene."


class MockAIDetector:
    def __init__(self):
        self.enabled = False
    
    def is_ready(self):
        return False
    
    def should_trigger_fallback(self, yolo_detections):
        return False
    
    def detect_objects(self, frame):
        return []
    
    def describe_scene(self, frame):
        return "AI detection not available"


def create_simple_ai_detector():
    """Factory function to create simple AI detector"""
    return SimpleAIDetector()
