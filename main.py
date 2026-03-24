"""
AI Assistive Vision System - Main Entry Point
=============================================
Capabilities:
  - YOLOv8 object detection (webcam)
  - Face recognition (offline, local DB)
  - Wake-word activation ("Hey Vision" / "computer")
  - Voice commands via microphone
  - Text-to-speech responses
  - OCR (optional, requires Tesseract)
"""

import threading
import time
import cv2
import sys

from vision.detector        import ObjectDetector
from vision.ocr_reader      import OCRReader
from vision.face_recognizer import FaceRecognizer
from voice.speaker          import Speaker
from voice.listener         import VoiceListener
from voice.wake_word        import WakeWordDetector
from core.scene_builder     import SceneBuilder
from core.command_handler   import CommandHandler


# How long (seconds) the system stays "awake" after a wake word
ACTIVE_WINDOW = 15


def main():
    print("=" * 60)
    print("  AI Assistive Vision System  v2.0")
    print("  Face Recognition + Wake Word Edition")
    print("=" * 60)

    # ── Initialise all modules ────────────────────────────────────────────────
    print("[INIT] Loading object detector (YOLOv8n)...")
    detector = ObjectDetector()

    print("[INIT] Loading face recognizer...")
    face_recognizer = FaceRecognizer()

    print("[INIT] Loading OCR engine...")
    ocr = OCRReader()

    print("[INIT] Setting up text-to-speech...")
    speaker = Speaker()

    print("[INIT] Setting up scene builder...")
    scene_builder = SceneBuilder()

    print("[INIT] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INIT] Starting wake-word detector...")
    wake_detector = WakeWordDetector()

    print("[INIT] Starting voice listener...")
    listener = VoiceListener()

    # ── Shared state ──────────────────────────────────────────────────────────
    state = {
        "latest_frame":      None,
        "latest_detections": [],
        "latest_faces":      [],
        "running":           True,
        "active":            False,   # True = listening for commands
        "active_until":      0.0,     # epoch time when active window expires
    }
    frame_lock = threading.Lock()

    # ── Command handler ───────────────────────────────────────────────────────
    command_handler = CommandHandler(
        state=state,
        frame_lock=frame_lock,
        detector=detector,
        ocr=ocr,
        speaker=speaker,
        scene_builder=scene_builder,
        face_recognizer=face_recognizer,
        cap=cap,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 1: Camera + Detection + Face Recognition
    # ─────────────────────────────────────────────────────────────────────────
    def camera_loop():
        print("[CAMERA] Camera loop started.")
        face_tick = 0   # run face recognition every N frames (slower)
        FACE_EVERY = 10

        while state["running"]:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = detector.detect(frame)

            # Face recognition — run every FACE_EVERY frames to stay fast
            faces = []
            if face_recognizer.available:
                face_tick += 1
                if face_tick >= FACE_EVERY:
                    face_tick = 0
                    faces = face_recognizer.identify(frame)

            with frame_lock:
                state["latest_frame"]      = frame.copy()
                state["latest_detections"] = detections
                if faces:                         # keep last known faces if none this tick
                    state["latest_faces"] = faces

            # Draw object boxes
            annotated = detector.draw_boxes(frame, detections)
            # Draw face boxes on top
            if face_recognizer.available:
                with frame_lock:
                    f = list(state["latest_faces"])
                annotated = face_recognizer.draw_faces(annotated, f)

            # Status overlay
            status = "ACTIVE — listening" if state["active"] else \
                     f"SLEEPING — say '{wake_detector.mode == 'porcupine' and 'computer' or 'hey vision'}' to wake"
            cv2.putText(annotated, status, (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0) if state["active"] else (100, 100, 100), 1)

            cv2.imshow("AI Vision Assistant (press Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                state["running"] = False
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 2: Wake-word listener → activates command window
    # ─────────────────────────────────────────────────────────────────────────
    def wake_loop():
        print("[WAKE] Wake-word loop started.")
        time.sleep(2)

        while state["running"]:
            # Block until wake word fires (poll every 0.1s to check running)
            fired = wake_detector.detected(timeout=0.5)
            if not fired:
                continue

            # Activate command window
            state["active"]       = True
            state["active_until"] = time.time() + ACTIVE_WINDOW
            speaker.speak("Yes? I'm listening.")
            print(f"[WAKE] Active for {ACTIVE_WINDOW}s.")

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 3: Active-window expiry watchdog
    # ─────────────────────────────────────────────────────────────────────────
    def active_watchdog():
        while state["running"]:
            if state["active"] and time.time() > state["active_until"]:
                state["active"] = False
                print("[WAKE] Active window expired. Back to sleep.")
            time.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 4: Voice command listener (only processes when active)
    # ─────────────────────────────────────────────────────────────────────────
    def voice_loop():
        print("[VOICE] Voice command loop started.")
        time.sleep(3)

        while state["running"]:
            if not state["active"]:
                time.sleep(0.2)
                continue

            command = listener.listen()
            if command:
                # Reset the active window on each successful command
                state["active_until"] = time.time() + ACTIVE_WINDOW
                command_handler.handle(command)

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 5: Auto-describe (only while sleeping, every 10s)
    # ─────────────────────────────────────────────────────────────────────────
    def auto_describe_loop():
        print("[AUTO] Auto-describe loop started.")
        time.sleep(5)
        while state["running"]:
            if not state["active"]:
                with frame_lock:
                    detections = list(state["latest_detections"])
                    faces      = list(state["latest_faces"])
                if detections:
                    desc = scene_builder.build_description(detections)
                    # Prepend known face names if any
                    known = [f["name"] for f in faces if f["name"] != "Unknown"]
                    if known:
                        desc = ", ".join(known) + " detected. " + desc
                    speaker.speak(desc)
            time.sleep(10)

    # ── Launch threads ────────────────────────────────────────────────────────
    threads = [
        threading.Thread(target=camera_loop,       daemon=True),
        threading.Thread(target=wake_loop,          daemon=True),
        threading.Thread(target=active_watchdog,    daemon=True),
        threading.Thread(target=voice_loop,         daemon=True),
        threading.Thread(target=auto_describe_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    ww_phrase = "computer" if wake_detector.mode == "porcupine" else "hey vision"
    enrolled = face_recognizer.list_known() if face_recognizer.available else []

    print("\n[READY] System is running!")
    print(f"  Wake word : say '{ww_phrase}' to activate")
    print(f"  Commands  : describe scene / who is there / remember <name> / read text / stop")
    print(f"  Enrolled  : {enrolled if enrolled else 'nobody yet — say: remember <name>'}")
    print("  Press Q in the camera window to quit\n")

    speaker.speak(
        f"Vision assistant ready. Say {ww_phrase} to activate me. "
        + (f"I recognise {len(enrolled)} {'person' if len(enrolled)==1 else 'people'}."
           if enrolled else "No faces enrolled yet.")
    )

    # Main thread keeps alive until camera loop exits
    threads[0].join()
    state["running"] = False
    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Vision Assistant shut down.")


if __name__ == "__main__":
    main()
