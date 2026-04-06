"""
core/command_handler.py
=======================
Maps spoken voice commands to system actions.
Now includes face recognition commands (enrol, forget, identify, list).
"""

import re
import threading


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _has_word(text: str, *words: str) -> bool:
    t = _tokens(text)
    return any(w in t for w in words)


def _has_shutdown_intent(text: str) -> bool:
    return bool(re.search(r"\b(stop|exit|quit|shutdown)\b", text))


class CommandHandler:
    def __init__(self, state, state_lock, frame_lock, detector, ocr, speaker, scene_builder,
                 face_recognizer=None):
        self.state           = state
        self.state_lock      = state_lock
        self.frame_lock      = frame_lock
        self.detector        = detector
        self.ocr             = ocr
        self.speaker         = speaker
        self.scene_builder   = scene_builder
        self.face_recognizer = face_recognizer

    def handle(self, command_text):
        t = threading.Thread(target=self._dispatch, args=(command_text,), daemon=True)
        t.start()

    def _dispatch(self, text):
        text = text.lower().strip()
        print(f"[CMD] Processing: '{text}'")

        if _has_shutdown_intent(text):
            self.speaker.speak("Shutting down. Goodbye.")
            with self.state_lock:
                self.state["running"] = False

        elif any(kw in text for kw in ["remember", "enrol", "enroll",
                                        "learn my face", "learn face",
                                        "add face", "teach"]):
            name = self._extract_name(text, ["remember", "enrol", "enroll",
                                              "learn", "add", "teach", "as", "me"])
            if not name:
                self.speaker.speak("Who should I remember? Say 'remember' followed by a name.")
                return
            self._enrol_face(name)

        elif any(kw in text for kw in ["forget", "remove face", "delete face", "unlearn"]):
            name = self._extract_name(text, ["forget", "remove", "delete", "unlearn", "face"])
            if not name:
                self.speaker.speak("Who should I forget? Say 'forget' followed by a name.")
                return
            self._forget_face(name)

        elif any(kw in text for kw in ["who do you know", "list faces",
                                        "who have you learned", "known people"]):
            self._list_known_faces()

        elif any(
            kw in text
            for kw in [
                "who is there", "who's there", "who are there",
                "identify", "who do you see", "who is in front", "who's in front",
                "who is that", "who's that", "recognize anyone", "recognise anyone",
            ]
        ):
            self._identify_people()

        elif (
            _has_word(text, "describe")
            or re.search(r"\bscene\b", text)
            or any(
                p in text
                for p in [
                    "what do you see",
                    "what is around",
                    "surroundings",
                    "environment",
                    "what's there",
                    "what is there",
                    "look around",
                ]
            )
        ):
            self._describe_scene()

        elif (
            any(p in text for p in ["in front", "what is in front", "what's in front"])
            or _has_word(text, "ahead", "forward")
            or re.search(r"\bfront\b", text)
        ):
            self._describe_front()

        elif (
            _has_word(text, "read", "ocr", "text")
            or any(
                p in text
                for p in [
                    "sign says",
                    "the sign",
                    "label",
                    "writing on",
                    "what does it say",
                    "read the",
                    "read that",
                    "read this",
                ]
            )
        ):
            self._read_text()

        elif any(kw in text for kw in ["help", "commands", "what can you do"]):
            self._say_help()

        elif any(kw in text for kw in ["enable ai", "turn on ai", "activate ai", "ai on"]):
            self._toggle_ai(True)

        elif any(kw in text for kw in ["disable ai", "turn off ai", "deactivate ai", "ai off"]):
            self._toggle_ai(False)

        elif any(kw in text for kw in ["ai detect", "use ai", "ai scan", "ai find", "ai search"]):
            self._force_ai_detection()

        elif any(kw in text for kw in ["ai status", "ai mode", "check ai"]):
            self._ai_status()

        else:
            self.speaker.speak("I didn't understand that. Say 'help' for available commands.")

    def _extract_name(self, text, stop_words):
        words = text.split()
        skip_set = set(stop_words)
        result = []
        collecting = False
        for w in words:
            if w in skip_set:
                collecting = True
                continue
            if collecting:
                result.append(w)
        name = " ".join(result).strip().title()
        if len(name) < 2 or name.lower() in {"a", "the", "me", "my", "face"}:
            return ""
        return name

    def _describe_scene(self):
        with self.frame_lock:
            detections = list(self.state["latest_detections"])
        self.speaker.speak(self.scene_builder.build_description(detections))

    def _describe_front(self):
        with self.frame_lock:
            detections = list(self.state["latest_detections"])
        front = [d for d in detections if d["position"] == "ahead"]
        if not front:
            self.speaker.speak("The path ahead appears to be clear.")
        else:
            self.speaker.speak("Directly in front of you: " +
                               self.scene_builder.build_description(front))

    def _read_text(self):
        with self.frame_lock:
            frame = self.state["latest_frame"]
            frame_copy = frame.copy() if frame is not None else None
        if frame_copy is None:
            self.speaker.speak("No camera frame available.")
            return
        if not self.ocr.available:
            self.speaker.speak("Text reading requires Tesseract OCR to be installed.")
            return
        self.speaker.speak("Reading text, please hold still.")
        text = self.ocr.read_text(frame_copy)
        if text:
            self.speaker.speak(f"I can see: {text}")
        else:
            self.speaker.speak("I could not read any clear text in this view.")

    def _identify_people(self):
        with self.frame_lock:
            frame_copy = (
                self.state["latest_frame"].copy()
                if self.state["latest_frame"] is not None
                else None
            )
            detections = list(self.state["latest_detections"])

        if self.face_recognizer and self.face_recognizer.available and frame_copy is not None:
            faces = self.face_recognizer.identify(frame_copy)
            if not faces:
                self.speaker.speak(self.scene_builder.build_person_description(detections))
                return

            known   = [f for f in faces if f["name"] != "Unknown"]
            unknown = [f for f in faces if f["name"] == "Unknown"]
            parts   = []

            if known:
                parts.append("I can see " + ", ".join(f["name"] for f in known))
            if unknown:
                n = len(unknown)
                parts.append(f"{n} unknown {'person' if n == 1 else 'people'}")
            for f in known:
                parts.append(f"{f['name']} is on the {f['position']}")

            self.speaker.speak(". ".join(parts) + ".")
        else:
            self.speaker.speak(self.scene_builder.build_person_description(detections))

    def _enrol_face(self, name):
        if not self.face_recognizer:
            self.speaker.speak("Face recognition module is not loaded.")
            return
        if not self.face_recognizer.available:
            self.speaker.speak("Face recognition requires the face-recognition library.")
            return
        self.face_recognizer.enroll(
            name,
            self.state,
            self.state_lock,
            self.frame_lock,
            self.speaker,
        )

    def _forget_face(self, name):
        if not self.face_recognizer:
            self.speaker.speak("Face recognition module is not loaded.")
            return
        self.face_recognizer.forget(name, self.speaker)

    def _list_known_faces(self):
        if not self.face_recognizer or not self.face_recognizer.available:
            self.speaker.speak("Face recognition is not available.")
            return
        names = self.face_recognizer.list_known()
        if not names:
            self.speaker.speak(
                "I don't know anyone yet. Say 'remember' followed by a name to enrol a face."
            )
        else:
            self.speaker.speak(
                f"I know {len(names)} {'person' if len(names)==1 else 'people'}: "
                + ", ".join(names) + "."
            )

    def _say_help(self):
        self.speaker.speak(
            "Available commands: "
            "'describe the scene'. "
            "'what is in front of me'. "
            "'who is there' to identify people. "
            "'remember John' to enrol a face. "
            "'forget John' to remove someone. "
            "'who do you know' to list enrolled faces. "
            "'read text' for OCR. "
            "'enable AI' or 'disable AI' to control AI detection. "
            "'AI detect' to force AI detection now. "
            "'AI status' to check AI mode. "
            "'stop' to exit."
        )

    def _toggle_ai(self, enable):
        """Toggle AI fallback detection on/off"""
        if hasattr(self.detector, 'ai_fallback'):
            self.detector.ai_fallback = enable
            status = "enabled" if enable else "disabled"
            self.speaker.speak(f"AI detection {status}.")
        else:
            self.speaker.speak("AI detection is not available.")

    def _force_ai_detection(self):
        """Force AI detection on current frame"""
        if not hasattr(self.detector, 'ai_detector') or not self.detector.ai_detector:
            self.speaker.speak("AI detection is not available.")
            return
        
        if not self.detector.ai_detector.is_ready():
            self.speaker.speak("AI detector is not ready.")
            return
        
        with self.frame_lock:
            frame = self.state.get("latest_frame")
            if frame is None:
                self.speaker.speak("No camera frame available.")
                return
        
        self.speaker.speak("Running AI detection...")
        ai_detections = self.detector.ai_detector.detect_objects(frame)
        
        if ai_detections:
            objects = [det.get('class', det.get('label', 'unknown')) for det in ai_detections]
            description = "AI found: " + ", ".join(objects[:5])  # Limit to 5 objects
            self.speaker.speak(description)
        else:
            self.speaker.speak("AI didn't detect any additional objects.")

    def _ai_status(self):
        """Report current AI status"""
        if not hasattr(self.detector, 'ai_fallback'):
            self.speaker.speak("AI detection is not available.")
            return
        
        ai_enabled = self.detector.ai_fallback
        ai_ready = (hasattr(self.detector, 'ai_detector') and 
                   self.detector.ai_detector and 
                   self.detector.ai_detector.is_ready())
        
        status = "enabled" if ai_enabled else "disabled"
        ready_status = "ready" if ai_ready else "not ready"
        
        self.speaker.speak(f"AI detection is {status} and {ready_status}.")
