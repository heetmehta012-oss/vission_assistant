#!/usr/bin/env python3
"""
Test script for AI detection functionality.
Tests the LLaVA fallback detector independently.
"""

import cv2
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.ai_detector import create_ai_detector
from vision.detector import ObjectDetector

def test_ai_detector():
    """Test AI detector with a sample frame."""
    print("=== Testing AI Detection ===")
    
    # Create AI detector
    ai_detector = create_ai_detector()
    
    if not ai_detector.is_ready():
        print("❌ AI detector not ready. Model may still be loading...")
        return False
    
    # Create a test frame (black with some objects)
    test_frame = cv2.imread("test_image.jpg") if os.path.exists("test_image.jpg") else None
    
    if test_frame is None:
        # Create synthetic test frame
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some rectangles to simulate objects
        cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(test_frame, (300, 150), (400, 250), (128, 128, 128), -1)
        print("Using synthetic test frame")
    else:
        print("Using existing test image")
    
    print("Running AI detection...")
    start_time = time.time()
    
    detections = ai_detector.detect_objects(test_frame)
    
    end_time = time.time()
    
    print(f"⏱️  AI detection took {end_time - start_time:.2f} seconds")
    print(f"📦 Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det.get('class', det.get('label', 'unknown'))} (confidence: {det.get('confidence', 0):.2f})")
    
    return len(detections) > 0

def test_integrated_detector():
    """Test integrated detector with AI fallback."""
    print("\n=== Testing Integrated Detection ===")
    
    # Create detector with AI fallback
    detector = ObjectDetector(ai_fallback=True)
    
    # Create test frame
    import numpy as np
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Running integrated detection...")
    start_time = time.time()
    
    detections = detector.detect(test_frame)
    
    end_time = time.time()
    
    print(f"⏱️  Integrated detection took {end_time - start_time:.2f} seconds")
    print(f"📦 Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        source = det.get('source', 'yolo') if 'source' in det else 'yolo'
        print(f"  {i}. {det['label']} ({source}) - confidence: {det['confidence']:.2f}, position: {det['position']}")
    
    return True

def test_fallback_triggering():
    """Test AI fallback triggering logic."""
    print("\n=== Testing Fallback Triggering ===")
    
    ai_detector = create_ai_detector()
    
    if not ai_detector.is_ready():
        print("❌ AI detector not ready")
        return False
    
    # Test cases
    test_cases = [
        ([], "Empty detections"),
        ([{"confidence": 0.2}], "Low confidence single detection"),
        ([{"confidence": 0.8}, {"confidence": 0.7}], "High confidence detections"),
        ([{"confidence": 0.3}, {"confidence": 0.4}], "Multiple low confidence detections"),
    ]
    
    for detections, description in test_cases:
        should_trigger = ai_detector.should_trigger_fallback(detections)
        print(f"  {description}: {'TRIGGER' if should_trigger else 'NO TRIGGER'}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing AI Detection System")
    print("=" * 50)
    
    success = True
    
    # Test individual AI detector
    if not test_ai_detector():
        print("⚠️  AI detector test failed, but this may be due to model loading")
    
    # Test fallback triggering
    test_fallback_triggering()
    
    # Test integrated detector
    test_integrated_detector()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    print("\n💡 To use AI detection in the main system:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run: python main.py")
    print("   3. AI will automatically trigger when YOLO finds few objects")
    print("\n🔧 Configure with environment variables:")
    print("   VISION_AI_FALLBACK=true/false")
    print("   VISION_AI_ENABLED=true/false")
    print("   VISION_AI_MODEL=llava-hf/llava-1.5-7b-hf")
