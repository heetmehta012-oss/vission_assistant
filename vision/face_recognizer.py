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

    def enroll(self, name: str, cap, speaker) -> bool:
        """
        Collect ENROLL_SEC seconds of frames from `cap` and build an
        average encoding for `name`. Returns True on success.

        cap     : an open cv2.VideoCapture object
        speaker : Speaker instance (for spoken feedback)
        """
        if not self.available:
            speaker.speak("Face recognition is not available.")
            return False

        name = name.strip().title()
        speaker.speak(
            f"Enrolling {name}. Please look directly at the camera. "
            f"I will capture your face for {ENROLL_SEC} seconds."
        )
        print(f"[FACE] Enrolling '{name}' for {ENROLL_SEC} seconds...")

        collected = []
        start = time.time()

        while time.time() - start < ENROLL_SEC:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes    = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings:
                collected.append(encodings[0])
                # Visual feedback: green rectangle while capturing
                for (top, right, bottom, left) in boxes:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
                cv2.putText(frame, f"Enrolling {name}...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            else:
                cv2.putText(frame, "No face detected – move closer", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

            cv2.imshow("AI Vision Assistant (Press Q to quit)", frame)
            cv2.waitKey(1)

        if len(collected) < 5:
            speaker.speak(
                f"Not enough face samples captured for {name}. "
                "Please try again with better lighting and face the camera directly."
            )
            return False

        # Average the collected encodings for a more robust representation
        mean_encoding = np.mean(collected, axis=0)

        with self._lock:
            # Remove old encodings for the same name (re-enrolment)
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

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = frame.shape[:2]
        # Use hog (CPU) model – faster than cnn on laptop
        boxes     = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

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
