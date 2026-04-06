"""
vision/detector.py
==================
Object detection using YOLOv8 (ultralytics).
Detects common objects and returns structured results.
"""

import os
import cv2
import time

# We use ultralytics YOLOv8 — default "small" balances accuracy vs CPU/GPU speed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found. Using mock detector.")


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# COCO class names (lowercase) relevant to navigation / indoor assistive use.
# Note: COCO uses "dining table", not "table"; "cell phone" not "phone".
PRIORITY_OBJECTS = {
    "person", "chair", "door", "table", "dining table", "laptop", "cell phone",
    "book", "cup", "bottle", "keyboard", "mouse", "tv", "monitor",
    "couch", "bed", "sink", "toilet", "car", "bicycle", "motorcycle", "bus", "truck",
    "dog", "cat",
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "backpack", "handbag", "suitcase", "umbrella",
    "potted plant", "microwave", "oven", "refrigerator", "clock",
    "remote", "toaster", "vase",
}


def _iou(box_a, box_b) -> float:
    """IoU for xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ObjectDetector:
    def __init__(
        self,
        model_name=None,
        confidence=None,
        iou=None,
        imgsz=None,
        max_det=None,
        smooth=True,
        smooth_alpha=0.35,
        iou_match=0.45,
        ai_fallback=True,
    ):
        """
        model_name  : weights file, e.g. yolov8s.pt (default), yolov8m.pt for more accuracy.
        confidence  : min class confidence (default from env VISION_YOLO_CONF or 0.38).
        iou         : NMS IoU (default VISION_YOLO_IOU or 0.50 — slightly stricter duplicate merge).
        imgsz       : inference square size (default VISION_YOLO_IMGSZ or 640; 832 helps small objects, slower).
        max_det     : cap boxes per frame (default 80).
        smooth      : temporal box smoothing (env VISION_YOLO_SMOOTH=0 to disable).
        smooth_alpha: blend prev box weight when matched (EMA on xyxy).
        iou_match   : min IoU to treat as same object across frames.
        ai_fallback : enable AI fallback detection (default True).
        """
        self.model_name = model_name or os.environ.get("VISION_YOLO_MODEL", "yolov8s.pt")
        self.confidence = confidence if confidence is not None else _env_float("VISION_YOLO_CONF", 0.38)
        self.iou = iou if iou is not None else _env_float("VISION_YOLO_IOU", 0.50)
        self.imgsz = imgsz if imgsz is not None else _env_int("VISION_YOLO_IMGSZ", 640)
        self.max_det = max_det if max_det is not None else _env_int("VISION_YOLO_MAX_DET", 80)
        self.smooth = smooth
        _sm = os.environ.get("VISION_YOLO_SMOOTH", "").strip().lower()
        if _sm in ("0", "false", "no", "off"):
            self.smooth = False
        elif _sm in ("1", "true", "yes", "on"):
            self.smooth = True
        self.smooth_alpha = _env_float("VISION_YOLO_SMOOTH_ALPHA", smooth_alpha)
        self.iou_match = _env_float("VISION_YOLO_MATCH_IOU", iou_match)
        self.device = os.environ.get("VISION_DEVICE", "").strip() or None
        
        # AI fallback configuration
        self.ai_fallback = ai_fallback and os.environ.get("VISION_AI_FALLBACK", "true").lower() == "true"

        self.model = None
        self._half = False
        self._prev_dets = []
        self.ai_detector = None
        
        # Initialize AI detector if enabled
        if self.ai_fallback:
            try:
                from .simple_ai_detector import create_simple_ai_detector
                self.ai_detector = create_simple_ai_detector()
                print("[DETECTOR] Simple AI fallback detector enabled (no downloads)")
            except Exception as e:
                print(f"[DETECTOR] Could not initialize AI detector: {e}")
                self.ai_fallback = False

        if YOLO_AVAILABLE:
            try:
                self._half = _cuda_available()
                self.model = YOLO(self.model_name)
                print(f"[DETECTOR] YOLO loaded: {self.model_name} (conf≥{self.confidence}, imgsz={self.imgsz})")
            except Exception as e:
                print(f"[DETECTOR] Could not load YOLO model: {e}")
        else:
            print("[DETECTOR] Running in DEMO mode (no real detection).")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns a list of dicts: [{label, confidence, box, position}, ...]
        box = (x1, y1, x2, y2) in pixels
        position = "left" | "ahead" | "right" based on frame width
        """
        if self.model is None:
            return self._mock_detections()

        h, w = frame.shape[:2]
        kwargs = dict(
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
            half=self._half,
        )
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(source=frame, **kwargs)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            if cx < w // 3:
                position = "left"
            elif cx < 2 * w // 3:
                position = "ahead"
            else:
                position = "right"

            if label not in PRIORITY_OBJECTS:
                continue

            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "position": position,
            })

        # AI Fallback: Check periodically (not every frame) and trigger if YOLO found very few objects
        if self.ai_fallback and self.ai_detector:
            # Only check AI fallback every 2 seconds to reduce CPU usage
            current_time = time.time()
            if not hasattr(self, '_last_ai_check'):
                self._last_ai_check = 0
            
            if current_time - self._last_ai_check > 2.0:  # Check every 2 seconds
                self._last_ai_check = current_time
                if self.ai_detector.should_trigger_fallback(detections):
                    print("[AI] Triggering fallback detection...")
                    ai_detections = self.ai_detector.detect_objects(frame)
                    
                    # Convert AI detections to expected format
                    for ai_det in ai_detections:
                        # AI detections don't have bounding boxes, so we'll use frame center
                        ai_det["position"] = "ahead"  # Default to center
                        ai_det["box"] = (w//4, h//4, 3*w//4, 3*h//4)  # Large central box
                        ai_det["label"] = ai_det.get("class", ai_det.get("label", "unknown"))
                    
                    # Merge AI detections with YOLO results (avoid duplicates)
                    yolo_labels = {d["label"] for d in detections}
                    for ai_det in ai_detections:
                        if ai_det["label"] not in yolo_labels:
                            detections.append(ai_det)
                            print(f"[AI] Added: {ai_det['label']}")

        if self.smooth and detections:
            detections = self._temporal_smooth(detections, w)
        self._prev_dets = [d.copy() for d in detections]

        return detections

    def _temporal_smooth(self, current, frame_w):
        """EMA on matched boxes to reduce jitter; recompute position from smoothed center."""
        if not self._prev_dets:
            return current

        alpha = self.smooth_alpha
        used_prev = set()
        out = []

        for d in current:
            best_j = -1
            best_iou = 0.0
            for j, p in enumerate(self._prev_dets):
                if j in used_prev or p["label"] != d["label"]:
                    continue
                iou = _iou(d["box"], p["box"])
                if iou > best_iou and iou >= self.iou_match:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                used_prev.add(best_j)
                pb = self._prev_dets[best_j]["box"]
                db = d["box"]
                sx1 = int(db[0] * (1 - alpha) + pb[0] * alpha)
                sy1 = int(db[1] * (1 - alpha) + pb[1] * alpha)
                sx2 = int(db[2] * (1 - alpha) + pb[2] * alpha)
                sy2 = int(db[3] * (1 - alpha) + pb[3] * alpha)
                cx = (sx1 + sx2) // 2
                if cx < frame_w // 3:
                    pos = "left"
                elif cx < 2 * frame_w // 3:
                    pos = "ahead"
                else:
                    pos = "right"
                out.append({
                    "label": d["label"],
                    "confidence": max(d["confidence"], self._prev_dets[best_j]["confidence"] * 0.99),
                    "box": (sx1, sy1, sx2, sy2),
                    "position": pos,
                })
            else:
                out.append(d)

        return out

    def draw_boxes(self, frame, detections):
        """Draw bounding boxes and labels on frame for the live preview."""
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['label']} ({d['position']}) {d['confidence']:.2f}"
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


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
