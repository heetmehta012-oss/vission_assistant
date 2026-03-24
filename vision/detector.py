"""
vision/detector.py
==================
Object detection using YOLOv8 (ultralytics).
Detects common objects and returns structured results.
"""

import cv2
import numpy as np

# We use ultralytics YOLOv8 - lightweight "nano" model (yolov8n)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found. Using mock detector.")


# Objects we especially care about for assistive use
PRIORITY_OBJECTS = {
    "person", "chair", "door", "table", "laptop", "cell phone",
    "book", "cup", "bottle", "keyboard", "mouse", "tv", "monitor",
    "couch", "bed", "sink", "toilet", "car", "bicycle", "dog", "cat",
    "stairs", "backpack", "handbag", "umbrella", "traffic light", "stop sign"
}


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", confidence=0.45):
        """
        Load YOLOv8 nano model.
        model_name : 'yolov8n.pt' is the smallest/fastest model (~6MB).
        confidence : only report detections above this threshold.
        """
        self.confidence = confidence
        self.model = None

        if YOLO_AVAILABLE:
            try:
                # Downloads automatically on first run
                self.model = YOLO(model_name)
                print(f"[DETECTOR] YOLOv8 loaded: {model_name}")
            except Exception as e:
                print(f"[DETECTOR] Could not load YOLO model: {e}")
        else:
            print("[DETECTOR] Running in DEMO mode (no real detection).")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns a list of dicts: [{label, confidence, box, position}, ...]
        box = (x1, y1, x2, y2) in pixels
        position = "left" | "center" | "right" based on frame width
        """
        if self.model is None:
            return self._mock_detections()

        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2  # horizontal center of bounding box

            # Determine left / center / right position
            if cx < w // 3:
                position = "left"
            elif cx < 2 * w // 3:
                position = "ahead"
            else:
                position = "right"

            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "position": position,
            })

        return detections

    def draw_boxes(self, frame, detections):
        """Draw bounding boxes and labels on frame for the live preview."""
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['label']} ({d['position']})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        return annotated

    def _mock_detections(self):
        """Fallback demo detections when YOLO is unavailable."""
        return [
            {"label": "person", "confidence": 0.9, "box": (100, 100, 300, 400), "position": "ahead"},
            {"label": "chair", "confidence": 0.8, "box": (400, 200, 600, 450), "position": "right"},
        ]
