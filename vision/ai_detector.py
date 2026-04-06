"""
vision/ai_detector.py
====================
AI Vision Fallback Detector using LLaVA model.
Provides secondary detection when YOLO fails to detect objects.
"""

import os
import cv2
import numpy as np
import threading
import time
from typing import List, Dict, Optional, Tuple
import re

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    from PIL import Image
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("[WARN] transformers/torch not found. AI detection disabled.")


class AIDetector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_detection_time = 0
        self.min_interval = 5.0  # Minimum seconds between AI detections
        
        # Configuration from environment
        self.enabled = os.environ.get("VISION_AI_ENABLED", "true").lower() == "true"
        self.confidence_threshold = float(os.environ.get("VISION_AI_CONF", "0.5"))
        self.model_name = os.environ.get("VISION_AI_MODEL", "openai/clip-vit-base-patch32")
        
        if self.enabled and AI_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load CLIP model in background thread"""
        def load():
            try:
                print("[AI] Loading CLIP model for object detection...")
                # Use CLIP for zero-shot object detection
                from transformers import CLIPModel, CLIPProcessor
                
                # Load with explicit model loading
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    cache_dir="./models_cache",
                    local_files_only=False
                )
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_name,
                    cache_dir="./models_cache",
                    local_files_only=False
                )
                
                # Define common object labels for zero-shot classification
                self.candidate_labels = [
                    "person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat",
                    "chair", "table", "couch", "bed", "toilet", "tv", "laptop", "cell phone",
                    "book", "clock", "vase", "bottle", "cup", "fork", "knife", "spoon", "bowl",
                    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza",
                    "door", "window", "mirror", "sink", "refrigerator", "microwave", "oven",
                    "keyboard", "mouse", "remote", "backpack", "umbrella", "handbag", "suitcase"
                ]
                
                print("[AI] CLIP model loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load AI model: {e}")
                print("[AI] Falling back to mock detector...")
                self.enabled = False
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def is_ready(self) -> bool:
        """Check if AI model is loaded and ready"""
        return self.enabled and self.model is not None and self.processor is not None
    
    def should_trigger_fallback(self, yolo_detections: List[Dict]) -> bool:
        """Determine if AI fallback should be triggered"""
        if not self.is_ready():
            return False
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.min_interval:
            return False
        
        # Trigger if YOLO found very few objects or low confidence
        if len(yolo_detections) < 2:
            return True
        
        # Check if all detections have low confidence
        avg_confidence = np.mean([det.get('confidence', 0) for det in yolo_detections])
        if avg_confidence < 0.4:
            return True
        
        return False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Perform AI detection on frame using CLIP zero-shot"""
        if not self.is_ready():
            return []
        
        try:
            # Preprocess frame
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create text prompts for each candidate label
            text_inputs = [f"a photo of a {label}" for label in self.candidate_labels]
            
            # Process inputs
            inputs = self.processor(
                text=text_inputs, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, min(5, len(self.candidate_labels)))
            
            objects = []
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                if prob > self.confidence_threshold:
                    label = self.candidate_labels[idx.item()]
                    objects.append({
                        'class': label,
                        'confidence': prob.item(),
                        'bbox': None,  # CLIP doesn't provide bounding boxes
                        'source': 'ai'
                    })
            
            self.last_detection_time = time.time()
            return objects
            
        except Exception as e:
            print(f"[ERROR] AI detection failed: {e}")
            return []
    
    def _parse_objects_from_response(self, response: str) -> List[Dict]:
        """Parse AI response to extract object information"""
        try:
            # Extract the assistant's response part
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            # Find objects in the response
            objects = []
            
            # Look for comma-separated objects
            if "," in response:
                object_list = [obj.strip() for obj in response.split(",")]
            else:
                # Try to extract objects from sentences
                words = response.lower().replace(".", "").replace(":", "").split()
                object_list = [word for word in words if len(word) > 2 and word not in 
                              ["the", "see", "can", "i", "you", "image", "picture", "there"]]
            
            # Clean and filter objects
            common_objects = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat", 
                            "chair", "table", "couch", "bed", "toilet", "tv", "laptop", "mouse", 
                            "keyboard", "cell phone", "book", "clock", "vase", "scissors", 
                            "teddy bear", "hair drier", "toothbrush", "bottle", "wine glass", 
                            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
            
            for obj in object_list[:10]:  # Limit to 10 objects
                obj = obj.lower().strip()
                if obj and len(obj) > 1:
                    # Check if it's a common object or seems reasonable
                    if obj in common_objects or (len(obj) > 3 and obj.isalpha()):
                        objects.append({
                            'class': obj,
                            'confidence': self.confidence_threshold,
                            'bbox': None,  # AI doesn't provide bounding boxes
                            'source': 'ai'
                        })
            
            return objects
            
        except Exception as e:
            print(f"[ERROR] Failed to parse AI response: {e}")
            return []
    
    def describe_scene(self, frame: np.ndarray) -> str:
        """Get detailed scene description from AI using CLIP"""
        if not self.is_ready():
            return "AI model not available"
        
        try:
            # Use CLIP to detect objects and create a description
            objects = self.detect_objects(frame)
            
            if not objects:
                return "I don't see any familiar objects in this scene."
            
            # Sort by confidence and take top 3
            objects.sort(key=lambda x: x['confidence'], reverse=True)
            top_objects = objects[:3]
            
            # Create description
            object_names = [obj['class'] for obj in top_objects]
            if len(object_names) == 1:
                return f"I see a {object_names[0]} in this scene."
            elif len(object_names) == 2:
                return f"I see a {object_names[0]} and a {object_names[1]} in this scene."
            else:
                return f"I see a {object_names[0]}, {object_names[1]}, and {object_names[2]} in this scene."
            
        except Exception as e:
            print(f"[ERROR] AI scene description failed: {e}")
            return "Failed to describe scene"


# Mock detector for when AI is not available
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


def create_ai_detector():
    """Factory function to create appropriate AI detector"""
    if AI_AVAILABLE:
        return AIDetector()
    else:
        return MockAIDetector()
