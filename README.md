# AI Assistive Vision System

Local webcam assistant: YOLOv8 object detection, optional offline face recognition, wake word, voice commands, TTS, optional OCR, and AI fallback detection for enhanced object recognition.

## AI Fallback Detection

The system now includes **Simple AI detection** as a secondary method when YOLOv8 fails to detect sufficient objects:

- **Automatic triggering**: AI activates when YOLO finds < 2 objects or low confidence
- **No downloads required**: Uses computer vision techniques (contour analysis)
- **Shape-based detection**: Identifies objects by analyzing shapes and patterns
- **Configurable**: Can be disabled via `VISION_AI_FALLBACK=false`

**Advantages**: Instant setup, no storage usage, privacy-focused

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. **Windows**: PyAudio often needs [pipwin](https://pypi.org/project/pipwin/) or a prebuilt wheel. **dlib** may require CMake before `pip install dlib`.

3. **OCR**: Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it is on `PATH`.

4. **Wake word (optional)**: `pip install pvporcupine`, set `PICOVOICE_KEY`, or use the speech-recognition fallback / keyboard (Enter) if no mic.

## Run

From the project root (so `vision`, `voice`, and `core` import correctly):

```bash
python main.py
```

Press **Q** in the camera window to quit, or say **stop** while the command window is active.

## AI Voice Commands

The system supports voice commands to control AI detection:

- **"Enable AI" / "Turn on AI"** - Activate AI fallback detection
- **"Disable AI" / "Turn off AI"** - Deactivate AI fallback detection  
- **"AI detect" / "Use AI"** - Force AI detection on current frame
- **"AI status" / "Check AI"** - Report current AI mode and readiness

AI detection runs automatically when YOLO finds few objects, but these commands give you manual control.

## Detection tuning (YOLO)

Default model is **YOLOv8s** (`yolov8s.pt`), which is more accurate than nano at the cost of some FPS. First run downloads weights into the project folder.

| Environment variable | Meaning |
|----------------------|--------|
| `VISION_YOLO_MODEL` | e.g. `yolov8m.pt` for higher accuracy, `yolov8n.pt` for speed |
| `VISION_YOLO_CONF` | Minimum confidence (default `0.38`; lower = more detections, more noise) |
| `VISION_YOLO_IOU` | NMS IoU (default `0.50`) |
| `VISION_YOLO_IMGSZ` | Inference size (default `640`; try `832` for small/distant objects, slower) |
| `VISION_YOLO_SMOOTH` | `0` to disable temporal box smoothing |
| `VISION_DEVICE` | e.g. `0` for first CUDA GPU; omit for auto |
| `VISION_AI_FALLBACK` | `true`/`false` to enable AI fallback detection (default `true`) |
| `VISION_AI_ENABLED` | `true`/`false` to enable AI model loading (default `true`) |
| `VISION_AI_CONF` | AI detection confidence threshold (default `0.5`) |
| `VISION_AI_MODEL` | AI model name (not used for simple detector) |

Example (PowerShell): `$env:VISION_YOLO_MODEL="yolov8m.pt"; python main.py`

While sleeping, the assistant periodically describes the scene. **The same sentence is not repeated** for about 45 seconds unless the scene wording changes (e.g. objects move). Adjust `AUTO_DESCRIBE_REPEAT_SAME_AFTER_SEC` and `AUTO_DESCRIBE_INTERVAL_SEC` in `main.py` if needed.

## Privacy note

Default speech recognition may use **Google** online STT when listening for commands or wake phrases. Face data stays under `known_faces/` (gitignored).
