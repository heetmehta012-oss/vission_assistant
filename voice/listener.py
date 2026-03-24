"""
voice/listener.py
=================
Microphone-based speech recognition.
Handles missing PyAudio gracefully — the app still starts without it.
"""

import time

SR_AVAILABLE = False
PYAUDIO_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr
    # Only mark available if pyaudio is also present
    if PYAUDIO_AVAILABLE:
        SR_AVAILABLE = True
    else:
        print("[MIC] PyAudio not found — microphone disabled.")
        print("      Fix: pip install pipwin  then  pipwin install pyaudio")
except ImportError:
    print("[MIC] SpeechRecognition not found. Voice commands disabled.")


class VoiceListener:
    def __init__(self, timeout=5, phrase_limit=6):
        self.available = SR_AVAILABLE
        self.timeout = timeout
        self.phrase_limit = phrase_limit
        self.recognizer = None
        self.mic = None

        if not self.available:
            print("[MIC] Running without voice commands (PyAudio missing).")
            return

        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 0.8
            self.recognizer.dynamic_energy_threshold = True
            self.mic = sr.Microphone()

            print("[MIC] Calibrating microphone for ambient noise (2s)...")
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("[MIC] Microphone ready.")
        except Exception as e:
            print(f"[MIC] Microphone init failed: {e}")
            self.available = False

    def listen(self):
        if not self.available:
            time.sleep(5)
            return None

        try:
            with self.mic as source:
                print("[MIC] Listening for command...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_limit,
                )
            try:
                text = self.recognizer.recognize_google(audio).lower()
                print(f"[MIC] Heard: '{text}'")
                return text
            except sr.UnknownValueError:
                print("[MIC] Could not understand audio.")
                return None
            except sr.RequestError:
                print("[MIC] Google API unavailable. Trying Sphinx...")
                try:
                    text = self.recognizer.recognize_sphinx(audio).lower()
                    print(f"[MIC] Sphinx heard: '{text}'")
                    return text
                except Exception:
                    return None
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[MIC] Listener error: {e}")
            return None
