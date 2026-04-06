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


# How long (seconds) system stays "awake" after a wake word
ACTIVE_WINDOW = 15

# Auto-describe while sleeping: interval between checks, and min seconds before repeating the same line
AUTO_DESCRIBE_INTERVAL_SEC = 10
AUTO_DESCRIBE_REPEAT_SAME_AFTER_SEC = 45


def wake_prompt_for_mode(mode: str) -> str:
    if mode == "porcupine":
        return "computer"
    if mode == "sr_fallback":
        return "hey vision"
    return "Enter (in this terminal)"


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INIT] Starting wake-word detector...")
    wake_detector = WakeWordDetector()

    print("[INIT] Starting voice listener...")
    listener = VoiceListener()

    # ── Shared state ──────────────────────────────────────────────────
    state_lock = threading.Lock()
    frame_lock = threading.Lock()
    state = {
        "latest_frame":      None,
        "latest_detections": [],
        "latest_faces":      [],
        "running":           True,
        "active":            False,   # True = listening for commands
        "active_until":      0.0,     # epoch time when active window expires
        "enroll_status":     None,    # str overlay during face enrolment
        "last_auto_describe":       "",   # debounce identical auto-narration
        "last_auto_describe_at":    0.0,  # epoch when we last spoke auto-describe
    }

    # ── Command handler ───────────────────────────────────────────────
    command_handler = CommandHandler(
        state=state,
        state_lock=state_lock,
        frame_lock=frame_lock,
        detector=detector,
        ocr=ocr,
        speaker=speaker,
        scene_builder=scene_builder,
        face_recognizer=face_recognizer,
    )

    # ── Threads ───────────────────────────────────────────────
    ww_phrase = wake_prompt_for_mode(wake_detector.mode)
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

    # ─────────────────────────────────────────────────────────────────
    # Thread 1: Camera + Detection + Face Recognition
    # ─────────────────────────────────────────────────────────────────
    def camera_loop():
        print("[CAMERA] Camera loop started.")
        face_tick = 0   # run face recognition every N frames (slower)
        FACE_EVERY = 10

        while True:
            with state_lock:
                if not state["running"]:
                    break

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
                if faces:
                    state["latest_faces"] = faces

            # Draw object boxes
            annotated = detector.draw_boxes(frame, detections)
            # Draw face boxes on top
            if face_recognizer.available:
                with frame_lock:
                    f = list(state["latest_faces"])
                annotated = face_recognizer.draw_faces(annotated, f)

            cv2.imshow("Vision Assistant", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                with state_lock:
                    state["running"] = False
                break

    # Thread 2: Wake-word detection (Porcupine or SR fallback)
    # ─────────────────────────────────────────────────────────────────
    def wake_loop():
        print("[WAKE] Wake-word loop started.")
        time.sleep(2)

        while True:
            with state_lock:
                if not state["running"]:
                    break
            fired = wake_detector.detected(timeout=0.5)
            if not fired:
                continue

            with state_lock:
                state["active_until"] = time.time() + ACTIVE_WINDOW
                state["active"] = True
            print("[WAKE] Wake word detected!")
            speaker.speak("Yes?")
            # Give a moment for the voice listener to take over
            time.sleep(1)

    # Thread 3: Voice command listener (active only after wake word)
    # ─────────────────────────────────────────────────────────────────
    def voice_loop():
        print("[VOICE] Voice command loop started.")
        time.sleep(3)

        while True:
            with state_lock:
                if not state["running"]:
                    break
                # Auto-deactivate after ACTIVE_WINDOW seconds of inactivity
                if state["active"] and time.time() > state["active_until"]:
                    state["active"] = False
                    continue

            # Listen only when "active" (after wake word)
            if state["active"]:
                command = listener.listen()
                if command:
                    command_handler.handle(command)
                    # Reset activity window on any command
                    with state_lock:
                        state["active_until"] = time.time() + ACTIVE_WINDOW
                # Stay active for continuous commands

    # Thread 4: Auto-describe while sleeping (no recent commands)
    # ─────────────────────────────────────────────────────────────────
    def auto_describe_loop():
        print("[AUTO] Auto-describe loop started.")
        time.sleep(5)

        while True:
            with state_lock:
                if not state["running"]:
                    break
                # Don't auto-describe if user has recently interacted
                if state["active"]:
                    continue
                # Don't auto-describe if we spoke the same line recently
                if time.time() - state["last_auto_describe_at"] < AUTO_DESCRIBE_REPEAT_SAME_AFTER_SEC:
                    continue

            with frame_lock:
                detections = list(state["latest_detections"])
            description = scene_builder.build_description(detections)
            if description and description != state["last_auto_describe"]:
                speaker.speak(description)
                state["last_auto_describe"] = description
                state["last_auto_describe_at"] = time.time()

            time.sleep(AUTO_DESCRIBE_INTERVAL_SEC)

    # Thread 5: Watchdog to keep the process alive on Windows
    # ─────────────────────────────────────────────────────────────────
    def active_watchdog():
        while True:
            with state_lock:
                if not state["running"]:
                    break
            time.sleep(1)

    # ── Start all threads ───────────────────────────────────────────────
    threads = [
        threading.Thread(target=camera_loop,     daemon=True),
        threading.Thread(target=wake_loop,        daemon=True),
        threading.Thread(target=voice_loop,         daemon=True),
        threading.Thread(target=auto_describe_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    # Main thread keeps alive until camera loop exits
    threads[0].join()
    with state_lock:
        state["running"] = False
    wake_detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Vision Assistant shut down.")


if __name__ == "__main__":
    main()