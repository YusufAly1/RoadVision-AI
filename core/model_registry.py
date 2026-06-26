"""
Model Registry — Centralized lazy-loading and lifecycle management
for all AI models in the multi-model pipeline.

Models are loaded on first access and cached as singletons.
Thread-safe access is ensured via threading locks.

Model inventory:
    1. YOLOv8 Tracker (master) — already loaded by TrafficAnalyzer
    2. BLIP Image Captioner — vehicle type & color description
    3. YOLO Plate Detector — license plate bounding box detection
    4. EasyOCR Reader — OCR on cropped plate images
    5. CLIP Zero-Shot — driver monitoring (seatbelt, phone)
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for all AI models.

    All heavy models are loaded lazily on first call to their getter.
    This prevents a 30+ second cold start when the application launches.
    Failed model loads are caught and logged — the feature is disabled
    gracefully rather than crashing the application.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True



        self._plate_detector = None
        self._plate_lock = threading.Lock()
        self._plate_available = True

        self._trocr_processor = None
        self._trocr_model = None
        self._trocr_lock = threading.Lock()
        self._trocr_available = True

        # Device detection
        self._device = self._detect_device()
        logger.info(f"ModelRegistry initialized — device: {self._device}")

    @staticmethod
    def _detect_device() -> str:
        """Detect the optimal device available."""
        try:
            import torch
            device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
            return device
        except ImportError:
            pass
        return "cpu"

    @property
    def device(self) -> str:
        return self._device

    # -------------------------------------------------
    # Plate Detector — YOLOv8 fine-tuned for plates
    # -------------------------------------------------

    def get_plate_detector(self):
        """
        Returns a YOLO model for license plate detection.
        Disabled (returns None) due to HF auth issues, triggering heuristic fallback.
        """
        if not self._plate_available:
            return None

        if self._plate_detector is None:
            with self._plate_lock:
                if self._plate_detector is None:
                    try:
                        import os
                        from ultralytics import YOLO
                        from utils.constants import PLATE_YOLO_MODEL
                        
                        logger.info(f"Loading YOLO Plate Detector: {PLATE_YOLO_MODEL}...")
                        if not os.path.exists(PLATE_YOLO_MODEL):
                            logger.error(f"YOLO Plate Detector weights missing at {PLATE_YOLO_MODEL}")
                            self._plate_available = False
                            return None
                            
                        self._plate_detector = YOLO(PLATE_YOLO_MODEL)
                        logger.info("YOLO Plate Detector loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load YOLO Plate Detector: {e}")
                        self._plate_available = False
                        return None

        return self._plate_detector

    @property
    def plate_available(self) -> bool:
        return self._plate_available

    # -------------------------------------------------
    # TrOCR — License Plate Character Recognition
    # -------------------------------------------------

    def get_trocr_model(self):
        """
        Returns (processor, model) for TrOCR.
        Loads on first call from local models/_weights. Returns (None, None) if unavailable.
        """
        if not self._trocr_available:
            return None, None

        if self._trocr_model is None:
            with self._trocr_lock:
                if self._trocr_model is None:
                    try:
                        import os
                        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                        from utils.constants import TROCR_MODEL_PATH

                        logger.info(f"Loading TrOCR from {TROCR_MODEL_PATH}...")
                        if not os.path.exists(TROCR_MODEL_PATH):
                            logger.error(f"TrOCR path missing: {TROCR_MODEL_PATH}")
                            self._trocr_available = False
                            return None, None

                        self._trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH, local_files_only=True)
                        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH, local_files_only=True)
                        
                        self._trocr_model = self._trocr_model.to(self._device)
                        self._trocr_model.eval()
                        
                        logger.info(f"TrOCR loaded successfully on device: {self._device}.")
                    except Exception as e:
                        logger.error(f"Failed to load TrOCR: {e}")
                        self._trocr_available = False
                        return None, None

        return self._trocr_processor, self._trocr_model

    @property
    def trocr_available(self) -> bool:
        return self._trocr_available

    # -------------------------------------------------
    # Lifecycle
    # -------------------------------------------------

    def unload_all(self):
        """Release all models from memory."""
        import gc

        self._plate_detector = None
        self._trocr_processor = None
        self._trocr_model = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("All models unloaded.")

    def status_report(self) -> dict:
        """Get a summary of which models are loaded and available."""
        return {
            "device": self._device,
            "plate_detector": {
                "available": self._plate_available,
                "loaded": self._plate_detector is not None,
            },
            "trocr": {
                "available": self._trocr_available,
                "loaded": self._trocr_model is not None,
            },
        }
