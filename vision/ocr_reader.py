"""
vision/ocr_reader.py
====================
Uses Tesseract OCR (via pytesseract) to read visible text in camera frames.
Falls back gracefully if Tesseract is not installed.
"""

import cv2
import numpy as np

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[OCR] pytesseract not found. OCR feature disabled.")


class OCRReader:
    def __init__(self):
        self.available = OCR_AVAILABLE
        if self.available:
            # Verify tesseract binary is reachable
            try:
                pytesseract.get_tesseract_version()
                print("[OCR] Tesseract OCR ready.")
            except Exception:
                self.available = False
                print("[OCR] Tesseract binary not found. Install tesseract-ocr system package.")
                print("      Ubuntu/Debian: sudo apt install tesseract-ocr")
                print("      macOS:         brew install tesseract")
                print("      Windows:       https://github.com/UB-Mannheim/tesseract/wiki")

    def read_text(self, frame):
        """
        Extract readable text from frame.
        Returns cleaned string of detected text, or empty string.
        """
        if not self.available:
            return ""

        # Preprocess: grayscale + threshold improves OCR accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Slight blur to reduce noise, then adaptive threshold
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            # PSM 6 = assume uniform block of text
            config = "--psm 6 --oem 3"
            raw_text = pytesseract.image_to_string(thresh, config=config)
            # Clean up: remove empty lines, strip whitespace
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            text = " ".join(lines)
            return text if len(text) > 3 else ""  # ignore very short noise
        except Exception as e:
            print(f"[OCR] Error: {e}")
            return ""
