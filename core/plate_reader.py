"""
Plate Reader — ALPR pipeline using YOLO plate detection + TrOCR.

Two-stage pipeline:
    Stage 1: YOLO plate detector finds plate bounding box in the vehicle crop.
    Stage 2: Crop, preprocess (RGB, resize), and pass to TrOCR for character recognition.

CRITICAL: This module is designed to run ONCE per unique Track_ID, at the
frame where the vehicle's bounding box area is largest (closest to camera).
Results are cached and never re-computed for the same vehicle.
"""

import cv2
import logging
import numpy as np
from typing import Optional, List, Dict

from core.model_registry import ModelRegistry
from utils.constants import PLATE_CONFIDENCE, DEBUG_PLATE_OCR
import os

logger = logging.getLogger(__name__)


def read_plate(vehicle_crop_bgr: np.ndarray, track_id: Optional[int] = None, frame_num: Optional[int] = None) -> List[Dict]:
    """
    Detect and read a license plate from a vehicle crop.

    Args:
        vehicle_crop_bgr: Cropped vehicle image in BGR format.

    Returns:
        List of dicts with keys: 
        plate_text, plate_bbox, detection_confidence, recognition_status
        Returns [] if no plate is detected or recognition fails.
    """
    if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
        logger.warning("Invalid or empty crop provided to read_plate.")
        return []

    registry = ModelRegistry()

    debug_dir = ""
    if DEBUG_PLATE_OCR and track_id is not None:
        debug_dir = f"outputs/debug_plate_ocr/track_{track_id}"
        os.makedirs(debug_dir, exist_ok=True)
        vh, vw = vehicle_crop_bgr.shape[:2]
        cv2.imwrite(f"{debug_dir}/f{frame_num}_1_vehicle_crop_{vw}x{vh}.jpg", vehicle_crop_bgr)

    # --- Stage 1: Plate Detection with YOLO ---
    detection_result = _detect_plate_region(vehicle_crop_bgr, registry)

    if not detection_result:
        return []

    plate_crop, bbox, conf = detection_result

    if DEBUG_PLATE_OCR and debug_dir:
        ph, pw = plate_crop.shape[:2]
        cv2.imwrite(f"{debug_dir}/f{frame_num}_2_plate_crop_{pw}x{ph}.jpg", plate_crop)

    # Skip OCR if crop is too small (e.g., car is far away)
    h, w = plate_crop.shape[:2]
    if h < 15 or w < 30:
        return [{
            "plate_text": "Low Quality",
            "plate_bbox": bbox,
            "detection_confidence": float(conf),
            "recognition_status": "LOW_QUALITY"
        }]

    # --- Stage 2: TrOCR on the plate crop ---
    debug_path = ""
    if DEBUG_PLATE_OCR and debug_dir:
        debug_path = f"{debug_dir}/f{frame_num}_3_trocr_input_"
        
    plate_text = _run_trocr(plate_crop, registry, debug_path)
    
    if not plate_text:
        recognition_status = "FAILED"
        plate_text = "Unknown"
    else:
        recognition_status = "SUCCESS"

    result = {
        "plate_text": plate_text,
        "plate_bbox": bbox,
        "detection_confidence": float(conf),
        "recognition_status": recognition_status
    }

    return [result]


def _detect_plate_region(
    vehicle_crop: np.ndarray,
    registry: ModelRegistry,
) -> Optional[tuple[np.ndarray, list[int], float]]:
    """
    Use YOLO plate detector to find the license plate within the vehicle crop.

    Returns:
        tuple (cropped_plate_image, [x1, y1, x2, y2], confidence) or None
    """
    detector = registry.get_plate_detector()

    if detector is None:
        return None

    try:
        results = detector(vehicle_crop, verbose=False, conf=PLATE_CONFIDENCE)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        # Take the highest-confidence detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())
        best_conf = float(confidences[best_idx])

        # Validate box dimensions
        h, w = vehicle_crop.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 5:
            return None

        plate_crop = vehicle_crop[y1:y2, x1:x2]
        return plate_crop, [x1, y1, x2, y2], best_conf

    except Exception as e:
        logger.warning(f"Plate detection failed: {e}")
        return None


def _run_trocr(plate_crop: np.ndarray, registry: ModelRegistry, debug_path: str = "") -> str:
    """
    Run TrOCR on a cropped plate image.

    Preprocessing steps:
        1. Convert BGR to RGB
        2. Resize/upscale for TrOCR visibility
    """
    processor, model = registry.get_trocr_model()

    if processor is None or model is None:
        return ""

    try:
        # --- Preprocess plate crop for TrOCR ---
        processed = _preprocess_plate(plate_crop)

        # TrOCR expects RGB images (PIL or numpy array in RGB)
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(registry.device)

        generated_ids = model.generate(pixel_values)
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not raw_text:
            return ""

        # Clean up
        plate_text = _clean_plate_text(raw_text)

        if debug_path:
            clean_res = plate_text if plate_text else "FAILED"
            cv2.imwrite(f"{debug_path}{clean_res}.jpg", processed)

        if len(plate_text) < 3:
            # Too short to be a real plate
            return ""

        return plate_text

    except Exception as e:
        logger.warning(f"TrOCR failed: {e}")
        return ""


def _preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess plate crop for better TrOCR accuracy.
    Resize small crops so TrOCR can see the characters clearly.
    """
    target_h = 64
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return plate_crop

    # Upscale if the plate is too small
    if h < target_h:
        scale = target_h / h
        target_w = int(w * scale)
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Add padding to maintain a wide aspect ratio (TrOCR prefers wide images)
        if target_w < target_h * 2:
            pad_w = target_h * 2 - target_w
            left = pad_w // 2
            right = pad_w - left
            resized = cv2.copyMakeBorder(resized, 0, 0, left, right, cv2.BORDER_REPLICATE)
            
        return resized

    return plate_crop


def _clean_plate_text(raw_text: str) -> str:
    """
    Clean OCR output: remove noise characters, normalize spacing.
    """
    import re

    # Uppercase
    cleaned = raw_text.upper()
    # Keep only alphanumeric, spaces, and dashes
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

    return cleaned


if __name__ == '__main__':
    # Test script for local verification
    import sys
    import os
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            test_img = cv2.imread(img_path)
            print(f"Processing {img_path}...")
            results = read_plate(test_img)
            print(f"Results: {results}")
        else:
            print(f"Image not found: {img_path}")
    else:
        print("Usage: python plate_reader.py <path_to_image>")
