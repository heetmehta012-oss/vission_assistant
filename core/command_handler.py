"""
core/command_handler.py
=======================
Maps spoken voice commands to system actions.
Now includes face recognition commands (enrol, forget, identify, list).
"""

import threading


class CommandHandler:
    def __init__(self, state, frame_lock, detector, ocr, speaker, scene_builder,
                 face_recognizer=None, cap=None):
        self.state           = state
        self.frame_lock      = frame_lock
        self.detector        = detector
        self.ocr             = ocr
        self.speaker         = speaker
        self.scene_builder   = scene_builder
        self.face_recognizer = face_recognizer
        self.cap             = cap

    def handle(self, command_text):
        t = threading.Thread(target=self._dispatch, args=(command_text,), daemon=True)
        t.start()

    def _dispatch(self, text):
        text = text.lower().strip()
        print(f"[CMD] Processing: '{text}'")

        if any(kw in text for kw in ["stop", "exit", "quit", "shutdown"]):
            self.speaker.speak("Shutting down. Goodbye.")
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

        elif any(kw in text for kw in ["who is there", "who's there", "identify",
                                        "who do you see", "who is in front",
                                        "who", "person", "people", "anyone", "someone"]):
            self._identify_people()

        elif any(kw in text for kw in ["describe", "scene", "what do you see",
                                        "what is around", "surroundings",
                                        "environment", "what's there", "what is there"]):
            self._describe_scene()

        elif any(kw in text for kw in ["front", "ahead", "in front",
                                        "what is in front", "what's in front", "forward"]):
            self._describe_front()

        elif any(kw in text for kw in ["read", "text", "sign", "label",
                                        "words", "writing", "what does it say"]):
            self._read_text()

        elif any(kw in text for kw in ["help", "commands", "what can you do"]):
            self._say_help()

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
        if frame is None:
            self.speaker.speak("No camera frame available.")
            return
        if not self.ocr.available:
            self.speaker.speak("Text reading requires Tesseract OCR to be installed.")
            return
        self.speaker.speak("Reading text, please hold still.")
        text = self.ocr.read_text(frame)
        if text:
            self.speaker.speak(f"I can see: {text}")
        else:
            self.speaker.speak("I could not read any clear text in this view.")

    def _identify_people(self):
        with self.frame_lock:
            frame      = self.state["latest_frame"]
            detections = list(self.state["latest_detections"])

        if self.face_recognizer and self.face_recognizer.available and frame is not None:
            faces = self.face_recognizer.identify(frame)
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
        if self.cap is None:
            self.speaker.speak("Camera is not available for enrolment.")
            return
        self.face_recognizer.enroll(name, self.cap, self.speaker)

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
            "'stop' to exit."
        )
