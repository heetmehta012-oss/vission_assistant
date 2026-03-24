"""
voice/wake_word.py
==================
Always-on wake-word detection with graceful fallback when PyAudio is missing.

Tier 1 — pvporcupine (needs PICOVOICE_KEY env var)
Tier 2 — SpeechRecognition keyword scan
Tier 3 — keyboard fallback (press Enter to activate)
"""

import os
import time
import threading

PORCUPINE_AVAILABLE = False
SR_AVAILABLE = False
PYAUDIO_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pass

try:
    import pvporcupine
    import struct
    if PYAUDIO_AVAILABLE:
        PORCUPINE_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr
    if PYAUDIO_AVAILABLE:
        SR_AVAILABLE = True
except ImportError:
    pass

FALLBACK_TRIGGERS = ["hey vision", "hey assistant", "wake up", "vision", "computer"]
PICOVOICE_KEY     = os.environ.get("PICOVOICE_KEY", "")
PORCUPINE_KEYWORD = "computer"
WAKE_KEYWORD_PATH = os.environ.get("WAKE_KEYWORD_PATH", "")


class WakeWordDetector:
    def __init__(self):
        self._event   = threading.Event()
        self._running = True
        self.mode     = "none"

        if PORCUPINE_AVAILABLE and PICOVOICE_KEY:
            self._start_porcupine()
        elif SR_AVAILABLE:
            self._start_sr_fallback()
        else:
            self._start_keyboard_fallback()

    def detected(self, timeout=None) -> bool:
        fired = self._event.wait(timeout=timeout)
        if fired:
            self._event.clear()
        return fired

    def stop(self):
        self._running = False

    # ── Porcupine ────────────────────────────────────────────────────────────

    def _start_porcupine(self):
        threading.Thread(target=self._porcupine_loop, daemon=True).start()
        self.mode = "porcupine"
        print(f"[WAKE] Porcupine started. Say '{PORCUPINE_KEYWORD}' to activate.")

    def _porcupine_loop(self):
        try:
            if WAKE_KEYWORD_PATH and os.path.exists(WAKE_KEYWORD_PATH):
                porcupine = pvporcupine.create(access_key=PICOVOICE_KEY,
                                               keyword_paths=[WAKE_KEYWORD_PATH])
            else:
                porcupine = pvporcupine.create(access_key=PICOVOICE_KEY,
                                               keywords=[PORCUPINE_KEYWORD])
        except Exception as e:
            print(f"[WAKE] Porcupine failed: {e}. Falling back.")
            self._start_sr_fallback()
            return

        pa = pyaudio.PyAudio()
        stream = pa.open(rate=porcupine.sample_rate, channels=1,
                         format=pyaudio.paInt16, input=True,
                         frames_per_buffer=porcupine.frame_length)
        try:
            while self._running:
                raw = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, raw)
                if porcupine.process(pcm) >= 0:
                    print("[WAKE] Wake word detected (Porcupine)!")
                    self._event.set()
        finally:
            stream.stop_stream(); stream.close()
            pa.terminate(); porcupine.delete()

    # ── SR fallback ───────────────────────────────────────────────────────────

    def _start_sr_fallback(self):
        threading.Thread(target=self._sr_loop, daemon=True).start()
        self.mode = "sr_fallback"
        print(f"[WAKE] SR fallback started. Say 'hey vision' to activate.")

    def _sr_loop(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold       = 250
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold        = 0.5
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)

        while self._running:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                try:
                    text = recognizer.recognize_google(audio).lower()
                except (sr.RequestError, sr.UnknownValueError):
                    try:
                        text = recognizer.recognize_sphinx(audio).lower()
                    except Exception:
                        continue
                if any(t in text for t in FALLBACK_TRIGGERS):
                    print(f"[WAKE] Wake phrase detected: '{text}'")
                    self._event.set()
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[WAKE] SR error: {e}")
                time.sleep(0.5)

    # ── Keyboard fallback (when no mic/pyaudio) ───────────────────────────────

    def _start_keyboard_fallback(self):
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        self.mode = "keyboard"
        print("[WAKE] No microphone available — press ENTER in this terminal to activate.")

    def _keyboard_loop(self):
        while self._running:
            try:
                input()   # blocks until Enter is pressed
                print("[WAKE] Activated via keyboard.")
                self._event.set()
            except EOFError:
                time.sleep(1)
