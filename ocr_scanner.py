"""
OCR Scanner - Fixed & Optimized Version
Works with stylized fonts, bold text, screenshots
"""

import easyocr
import cv2
import numpy as np
from typing import Dict, Any

# =========================
# ⚡ INIT OCR ENGINE
# =========================
reader = easyocr.Reader(['en'], gpu=False)


# =========================
# 🔍 OCR CORE FUNCTION (FIXED)
# =========================
def extract_text_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    try:
        # Convert bytes → image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image")

        # 🔥 IMPORTANT: Resize for better OCR
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 🔥 Keep it SIMPLE (no aggressive preprocessing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔍 Run OCR
        results = reader.readtext(gray)

        print("\n🔍 RAW OCR OUTPUT:", results)

        texts = []
        confidences = []

        for _, text, prob in results:
            if prob > 0.3:  # lower threshold
                texts.append(text.strip())
                confidences.append(prob)

        final_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "success": True,
            "text": final_text,
            "confidence": round(avg_conf, 3),
            "segments": len(texts)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "confidence": 0.0
        }


# =========================
# 🧠 OPTIONAL DETECTOR INTEGRATION
# =========================
def scan_image_with_detector(image_bytes: bytes, detector_func) -> Dict[str, Any]:
    ocr_result = extract_text_from_bytes(image_bytes)

    if not ocr_result["success"]:
        return {
            "verdict": "ERROR",
            "error": ocr_result.get("error")
        }

    text = ocr_result["text"]

    if not text:
        return {
            "verdict": "NO_TEXT",
            "confidence": 0.0
        }

    detection = detector_func(text)

    return {
        "verdict": detection.get("verdict"),
        "confidence": detection.get("confidence"),
        "source": "OCR_IMAGE",
        "extracted_text": text,
        "ocr_confidence": ocr_result["confidence"],
        "rules_triggered": detection.get("rules_triggered", []),
        "details": detection
    }


# =========================
# 🧪 TEST FUNCTION (RUN THIS)
# =========================
def test_ocr(image_path: str):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = extract_text_from_bytes(image_bytes)

        print("\n================ OCR RESULT ================")
        print("Success        :", result["success"])
        print("Extracted Text :", result["text"])
        print("Confidence     :", result["confidence"])
        print("Segments       :", result["segments"])

        if not result["success"]:
            print("Error          :", result.get("error"))

    except Exception as e:
        print("❌ Test failed:", str(e))


# =========================
# ▶️ RUN DIRECT TEST
# =========================
if __name__ == "__main__":
    # 🔥 PUT YOUR IMAGE NAME HERE
    test_ocr("a.jpeg")