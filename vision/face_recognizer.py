"""
vision/face_recognizer.py
=========================
Offline face recognition using the `face_recognition` library (dlib under the hood).

Two modes:
  1. ENROLL  — learn a person's face from the webcam and save it.
  2. IDENTIFY — recognise known faces in live frames.

All face data is stored locally in `known_faces/` as a pickle file.
No internet required. No cloud. Fully private.

Dependencies:
  pip install face-recognition face-recognition-models
  (dlib is pulled in automatically)
  On Windows dlib may need: pip install cmake && pip install dlib
"""

import os
import pickle
import time
import cv2
import numpy as np
import threading
from PIL import Image

try:
    import face_recognition
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    print("[FACE] face_recognition not installed. Face ID disabled.")
    print("       Install: pip install face-recognition")


# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "known_faces")
DATA_FILE  = os.path.join(DATA_DIR, "encodings.pkl")
TOLERANCE  = 0.55   # lower = stricter match (0.4–0.6 works well)
ENROLL_SEC = 5      # seconds of frames used per enrolment

_dlib_reject_logged = False


def _normalize_bgr_uint8(frame):
    """Return contiguous BGR uint8 (H,W,3), or None."""
    if frame is None or frame.size == 0:
        return None
    img = np.asarray(frame)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and float(np.max(img)) <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img)


def _dlib_rgb_and_gray(bgr_uint8):
    """
    dlib + NumPy 2.x on Windows often reject raw OpenCV arrays. Use the same
    ndarray layout as face_recognition.load_image_file() (PIL -> array).
    """
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2GRAY)
    rgb_out = np.asarray(Image.fromarray(rgb, mode="RGB"), dtype=np.uint8)
    gray_out = np.asarray(Image.fromarray(gray, mode="L"), dtype=np.uint8)
    return rgb_out, gray_out


def _hog_face_locations(rgb_u8, gray_u8):
    """HOG detector: try grayscale first (2-D), then RGB. Returns [] if dlib rejects both."""
    global _dlib_reject_logged
    for img in (gray_u8, rgb_u8):
        try:
            return face_recognition.face_locations(img, model="hog")
        except RuntimeError:
            continue
    if not _dlib_reject_logged:
        print(
            "[FACE] dlib could not read webcam frames (NumPy/dlib layout). "
            "Try: pip install \"numpy<2\" then reinstall. Face ID disabled until fixed."
        )
        _dlib_reject_logged = True
    return []


class FaceRecognizer:
    """
    Manages a local database of face encodings and performs recognition.
    Thread-safe for concurrent camera + command threads.
    """

    def __init__(self):
        self.available = FR_AVAILABLE
        self._lock = threading.Lock()
        # known_names[i] matches known_encodings[i]
        self.known_encodings: list = []
        self.known_names: list     = []

        if not self.available:
            return

        os.makedirs(DATA_DIR, exist_ok=True)
        self._load()
        n = len(set(self.known_names))
        print(f"[FACE] Face recognizer ready. {n} person(s) enrolled.")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        """Load saved encodings from disk."""
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "rb") as f:
                    data = pickle.load(f)
                self.known_encodings = data.get("encodings", [])
                self.known_names     = data.get("names", [])
            except Exception as e:
                print(f"[FACE] Could not load saved faces: {e}")

    def _save(self):
        """Persist encodings to disk."""
        with open(DATA_FILE, "wb") as f:
            pickle.dump({"encodings": self.known_encodings,
                         "names":     self.known_names}, f)

    # ── Enrolment ────────────────────────────────────────────────────────────

    def enroll(self, name: str, state, state_lock, frame_lock, speaker) -> bool:
        """
        Collect ENROLL_SEC seconds of frames from the shared camera feed
        (via state['latest_frame']) so only the camera thread reads VideoCapture.

        state / state_lock : for enroll_status overlay and coordination
        frame_lock         : protects latest_frame snapshots
        speaker            : Speaker instance (for spoken feedback)
        """
        if not self.available:
            speaker.speak("Face recognition is not available.")
            return False

        name = name.strip().title()
        with state_lock:
            state["enroll_status"] = f"Enrolling {name} — look at the camera"

        speaker.speak(
            f"Enrolling {name}. Please look directly at the camera. "
            f"I will capture your face for {ENROLL_SEC} seconds."
        )
        print(f"[FACE] Enrolling '{name}' for {ENROLL_SEC} seconds...")

        collected = []
        start = time.time()

        try:
            while time.time() - start < ENROLL_SEC:
                with state_lock:
                    if not state["running"]:
                        return False

                with frame_lock:
                    frame = state["latest_frame"]
                    frame_copy = frame.copy() if frame is not None else None

                if frame_copy is None:
                    time.sleep(0.05)
                    continue

                bgr = _normalize_bgr_uint8(frame_copy)
                if bgr is None:
                    time.sleep(0.05)
                    continue

                rgb_u8, gray_u8 = _dlib_rgb_and_gray(bgr)
                boxes = _hog_face_locations(rgb_u8, gray_u8)
                try:
                    encodings = face_recognition.face_encodings(rgb_u8, boxes)
                except RuntimeError:
                    encodings = []

                if encodings:
                    collected.append(encodings[0])
                    with state_lock:
                        state["enroll_status"] = f"Enrolling {name} — face detected, hold still"
                else:
                    with state_lock:
                        state["enroll_status"] = f"Enrolling {name} — no face yet, move closer"

                time.sleep(0.04)

            if len(collected) < 5:
                speaker.speak(
                    f"Not enough face samples captured for {name}. "
                    "Please try again with better lighting and face the camera directly."
                )
                return False

            mean_encoding = np.mean(collected, axis=0)

            with self._lock:
                paired = [(e, n) for e, n in zip(self.known_encodings, self.known_names)
                          if n != name]
                self.known_encodings = [p[0] for p in paired]
                self.known_names     = [p[1] for p in paired]

                self.known_encodings.append(mean_encoding)
                self.known_names.append(name)
                self._save()

            speaker.speak(
                f"Got it! I have learned {name}'s face from "
                f"{len(collected)} samples. I will recognise them next time."
            )
            print(f"[FACE] Enrolled '{name}' with {len(collected)} samples.")
            return True
        finally:
            with state_lock:
                state["enroll_status"] = None

    def forget(self, name: str, speaker):
        """Remove a person from the database."""
        if not self.available:
            return
        name = name.strip().title()
        with self._lock:
            paired = [(e, n) for e, n in zip(self.known_encodings, self.known_names)
                      if n != name]
            before = len(self.known_names)
            self.known_encodings = [p[0] for p in paired]
            self.known_names     = [p[1] for p in paired]
            self._save()
        removed = before - len(self.known_names)
        if removed:
            speaker.speak(f"I have forgotten {name}.")
        else:
            speaker.speak(f"I don't have anyone named {name} in my memory.")

    def list_known(self) -> list:
        """Return list of enrolled names (deduplicated)."""
        with self._lock:
            return sorted(set(self.known_names))

    # ── Recognition ──────────────────────────────────────────────────────────

    def identify(self, frame) -> list:
        """
        Identify all faces in a single frame.
        Returns list of dicts:
          { name, confidence, box:(top,right,bottom,left), position }
        `name` is "Unknown" if no match found.
        """
        if not self.available:
            return []

        bgr = _normalize_bgr_uint8(frame)
        if bgr is None:
            return []

        rgb_u8, gray_u8 = _dlib_rgb_and_gray(bgr)
        h, w = rgb_u8.shape[:2]
        boxes = _hog_face_locations(rgb_u8, gray_u8)
        try:
            encodings = face_recognition.face_encodings(rgb_u8, boxes)
        except RuntimeError:
            return []

        results = []
        with self._lock:
            for enc, box in zip(encodings, boxes):
                top, right, bottom, left = box
                cx = (left + right) // 2

                name       = "Unknown"
                confidence = 0.0

                if self.known_encodings:
                    distances = face_recognition.face_distance(self.known_encodings, enc)
                    best_idx  = int(np.argmin(distances))
                    best_dist = float(distances[best_idx])

                    if best_dist <= TOLERANCE:
                        name       = self.known_names[best_idx]
                        confidence = round(1.0 - best_dist, 2)

                # Horizontal position
                if cx < w // 3:
                    position = "left"
                elif cx < 2 * w // 3:
                    position = "ahead"
                else:
                    position = "right"

                results.append({
                    "name":       name,
                    "confidence": confidence,
                    "box":        box,          # (top, right, bottom, left)
                    "position":   position,
                })

        return results

    def draw_faces(self, frame, face_results):
        """Annotate frame with face boxes and names (for live preview)."""
        annotated = frame.copy()
        for f in face_results:
            top, right, bottom, left = f["box"]
            color = (0, 180, 255) if f["name"] != "Unknown" else (0, 80, 200)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            label = f['name']
            if f["confidence"] > 0:
                label += f"  {int(f['confidence']*100)}%"
            cv2.putText(annotated, label, (left, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated
