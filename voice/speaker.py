"""
voice/speaker.py
================
Text-to-speech output using pyttsx3 (works offline on Windows, macOS, Linux).
Runs speech in a separate thread so it does not block the main loop.
"""

import threading
import queue

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[TTS] pyttsx3 not found. Voice output will be text-only.")


class Speaker:
    def __init__(self, rate=160, volume=1.0):
        """
        rate   : words per minute (default 160, lower = slower/clearer)
        volume : 0.0 to 1.0
        """
        self.queue = queue.Queue()
        self.available = TTS_AVAILABLE
        self._speaking = False

        if self.available:
            # pyttsx3 must run in its own thread (not thread-safe from multiple threads)
            self._thread = threading.Thread(target=self._run_tts, daemon=True)
            self._thread.start()
            self._rate = rate
            self._volume = volume
            print("[TTS] Speaker ready.")
        else:
            print("[TTS] Running in text-only mode.")

    def speak(self, text):
        """Queue a message to be spoken aloud."""
        print(f"[VOICE OUT] {text}")
        if self.available:
            self.queue.put(text)

    def _run_tts(self):
        """Worker thread: initializes engine and processes the speech queue."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        # Prefer a clear English voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en" in v.id.lower():
                engine.setProperty("voice", v.id)
                break

        while True:
            try:
                text = self.queue.get(timeout=0.5)
                self._speaking = True
                engine.say(text)
                engine.runAndWait()
                self._speaking = False
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Error: {e}")
                self._speaking = False

    @property
    def is_speaking(self):
        return self._speaking
