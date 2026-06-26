"""
Video processing thread — Master-Slave architecture (v2.1).

v2.1 Bug Fixes & Features:
    - FIXED: Added session_data dict for persistent vehicle data across frames.
    - FIXED: session_update signal for flicker-free UI table updates.
    - FIXED: Bbox area thresholds for slave model dispatch.
    - FIXED: Robust error handling in slave workers (Unknown/Unreadable fallbacks).
    - NEW: finished_processing signal for end-of-session summary.
    - NEW: Automatic CSV summary export on session end.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  VideoThread (QThread)                                   │
    │  ├── Master: YOLO track() — every frame                  │
    │  ├── Session Data: Persistent dict[Track_ID] → {...}     │
    │  └── Slaves dispatched to ThreadPoolExecutor:            │
    │       ├── BLIP vehicle description (bbox > 5000px²)      │
    │       ├── YOLO+EasyOCR plate read (bbox > 5000px²)       │
    │       └── CLIP DMS analysis (bbox > 8000px²)             │
    │  Results → session_data → session_update signal → UI     │
    └──────────────────────────────────────────────────────────┘
"""

import csv
import os
import cv2
import time
import logging
import numpy as np
import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

def sanitize_data(obj):
    if isinstance(obj, dict):
        return {k: sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_data(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_data(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt6.QtGui import QImage

from core.traffic_analyzer import TrafficAnalyzer, FrameResult
from core.model_registry import ModelRegistry
from utils.constants import (
    SLAVE_THREAD_POOL_WORKERS,
    MIN_BBOX_AREA_FOR_SLAVE,
    SUMMARY_CSV_FILENAME,
    VIDEOMAE_CHECKPOINT,
    VIDEOMAE_NUM_FRAMES,
    FULL_SCAN_THRESHOLD_SEC,
    VIDEOMAE_INPUT_SIZE,
    VIDEOMAE_ACCIDENT_CLASS_INDEX
)

from modules.accident_detection.accident_detector import VideoMAEAccidentDetector
from modules.accident_detection.llm_verifier import gemini_gatekeeper

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Video source types."""
    CAMERA = "camera"
    VIDEO_FILE = "video_file"
    IP_CAMERA = "ip_camera"


class VideoThread(QThread):
    """
    Worker thread for video capture and multi-model AI analysis.

    v2.1 Changes:
        - session_data dict persists all vehicle attributes across frames.
        - session_update signal emits complete state for flicker-free UI.
        - finished_processing signal triggers end-of-session summary + CSV.
        - Slave models only dispatched when bbox area exceeds threshold.
        - Robust error handling with "Unknown"/"Unreadable" fallbacks.
    """

    # Signals
    frame_ready = pyqtSignal(QImage, object)     # processed frame + FrameResult
    error_occurred = pyqtSignal(str)             # error message
    source_ended = pyqtSignal()                  # video file ended
    model_loaded = pyqtSignal()                  # YOLO model loaded successfully
    session_update = pyqtSignal(dict, int)       # session_data + current_frame
    finished_processing = pyqtSignal(dict)       # final session_data for summary
    accident_detected = pyqtSignal(dict)         # dict with accident verification info
    scan_complete = pyqtSignal(dict)             # dict with false alarm / rejected scan info

    def __init__(self, roi_config=None, parent=None):
        super().__init__(parent)
        self.analyzer = TrafficAnalyzer(roi_config=roi_config)

        # State
        self._source_type: SourceType = SourceType.CAMERA
        self._source_path: str = ""
        self._running = False
        self._paused = False

        # Threading primitives
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

        # Slave model thread pool
        self._slave_pool: ThreadPoolExecutor = None

        # Persistent session data — keyed by Track_ID, survives across frames
        self.session_data: dict = {}

        # Accident Event Collection State
        self.active_event_buffer: list = []
        self.frames_to_collect = 0
        self.event_trigger_frame = 0
        self.is_analyzing_accident = False
        
        # Intelligent Full Scan Mode State
        self.is_full_scan_mode = False
        self.is_full_scan_complete = False

    def set_source(self, source_type: SourceType, source_path: str = ""):
        """Configure the video source before starting."""
        self._source_type = source_type
        self._source_path = source_path

    def pause(self):
        """Pause frame processing."""
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        """Resume frame processing."""
        self._mutex.lock()
        self._paused = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()

    def stop(self):
        """Stop the thread gracefully."""
        self._mutex.lock()
        self._running = False
        self._paused = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def run(self):
        """
        Main thread loop — Master-Slave pipeline with session persistence.

        v2.1: Added session_data tracking, session_update emission,
        and finished_processing signal on loop exit (both natural
        end-of-stream and manual stop).
        """
        self._running = True
        self._paused = False

        # Load master YOLO model
        try:
            self.analyzer.load_model()
            self.model_loaded.emit()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load YOLO model: {str(e)}")
            return

        # Reset tracking state and session data
        self.analyzer.reset()
        self.session_data = {}
        self.session_events = {}

        # Load VideoMAE Detector
        logger.debug(f"[ACCIDENT_DEBUG] init_start")
        logger.debug(f"[ACCIDENT_DEBUG] checkpoint_path={VIDEOMAE_CHECKPOINT}")
        try:
            self.accident_detector = VideoMAEAccidentDetector(
                checkpoint_path=VIDEOMAE_CHECKPOINT,
                num_frames=VIDEOMAE_NUM_FRAMES,
                input_size=VIDEOMAE_INPUT_SIZE,
                accident_class_index=VIDEOMAE_ACCIDENT_CLASS_INDEX
            )
            logger.debug(f"[ACCIDENT_DEBUG] init_success=True (model instance: {self.accident_detector is not None})")
        except Exception as e:
            logger.debug(f"[ACCIDENT_DEBUG] init_success=False (error: {e})")
            logger.critical(f"FATAL: Failed to load VideoMAE detector: {e}")
            self.error_occurred.emit(f"Accident Detection Failed to initialize: {str(e)}")
            self.accident_detector = None
            # If timm is missing, we raise to surface it properly
            if "timm" in str(e).lower() or "vit_base_patch16_224" in str(e).lower():
                raise ImportError(f"Missing Accident Detection Dependencies: {e}")

        # Initialize slave thread pool
        self._slave_pool = ThreadPoolExecutor(
            max_workers=SLAVE_THREAD_POOL_WORKERS,
            thread_name_prefix="slave_model",
        )

        # Determine video source
        if self._source_type == SourceType.CAMERA:
            source = 0
        else:
            source = self._source_path

        # Open capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error_occurred.emit(
                f"Could not open video source: {source}"
            )
            self._slave_pool.shutdown(wait=False)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # Determine Intelligent Mode
        if self._source_type == SourceType.VIDEO_FILE and duration_sec < FULL_SCAN_THRESHOLD_SEC and duration_sec > 0:
            logger.info(f"Video duration ({duration_sec:.1f}s) is under {FULL_SCAN_THRESHOLD_SEC}s threshold. Activating Intelligent Full Scan Mode.")
            self.is_full_scan_mode = True
            if self.accident_detector and self._slave_pool:
                self.is_analyzing_accident = True
                self._slave_pool.submit(self._run_full_video_scan, self._source_path)
                
        frame_count = 0
        stream_ended_naturally = False

        while self._running:
            # Handle pause
            self._mutex.lock()
            if self._paused:
                self._pause_condition.wait(self._mutex)
            self._mutex.unlock()

            if not self._running:
                break

            ret, frame = cap.read()
            if not ret:
                stream_ended_naturally = True
                break

            frame_count += 1
            
            # Heartbeat check for diagnostics
            if frame_count % 150 == 0:
                logger.debug(f"[Heartbeat] VideoThread rendering loop active. Frame: {frame_count}")

            # =============================================
            # MASTER: Run YOLO tracking + BEV speed + behavior
            # =============================================
            try:
                result: FrameResult = self.analyzer.analyze_frame(
                    frame, frame_count, fps
                )
            except Exception as e:
                self.error_occurred.emit(
                    f"Analysis error (frame {frame_count}): {str(e)}"
                )
                continue

            # =============================================
            # SESSION DATA: Update persistent state FIRST
            # (creates entries before slaves access them)
            # =============================================
            self._update_session_data(result)

            # =============================================
            # SLAVE DISPATCH: Non-blocking async model calls
            # (only for vehicles with bbox > area threshold)
            # =============================================
            try:
                self._dispatch_slave_tasks(frame, result)
            except Exception as e:
                logger.warning(f"Slave dispatch error: {e}")

            # =============================================
            # EVENT COLLECTION: Non-blocking Post-Accident Wait State
            # =============================================
            if not self.is_full_scan_mode and getattr(self.analyzer, 'needs_accident_dispatch', False) and self.frames_to_collect == 0:
                self.analyzer.needs_accident_dispatch = False
                # Start non-blocking wait. We don't snapshot yet, wait 2.5s for aftermath.
                self.frames_to_collect = int(fps * 2.5) if fps > 0 else 75
                self.event_trigger_frame = result.frame_number
                logger.info(f"Event triggered at frame {self.event_trigger_frame}. Waiting 2.5s for aftermath...")
                
            if self.frames_to_collect > 0:
                self.frames_to_collect -= 1
                
                if self.frames_to_collect == 0:
                    logger.info("Aftermath period complete. Snapshotting buffer and Dispatching Deep Analysis Slave.")
                    self.analyzer.accident_dispatched = False  # Reset to allow new triggers
                    
                    if self.accident_detector and self._slave_pool:
                        self.is_analyzing_accident = True
                        
                        # Snapshot the rich buffer
                        event_buffer = list(self.analyzer.frame_buffer)
                        
                        # Extract Telemetry for the active window
                        active_telemetry = {}
                        start_frame = max(0, self.event_trigger_frame - int(fps * 4.5))
                        for tid, data in self.session_data.items():
                            if data["last_seen_frame"] >= start_frame:
                                active_telemetry[tid] = {
                                    "track_id": data["track_id"],
                                    "max_speed_kmh": data["max_speed_kmh"],
                                    "avg_speed_kmh": data["avg_speed_kmh"],
                                    "behavior": data["behavior"],
                                    "lane": data["lane"],
                                    "had_wrong_way": data["had_wrong_way"],
                                    "had_sudden_stop": data["had_sudden_stop"]
                                }
                                
                        self._slave_pool.submit(self._run_deep_accident_analysis, event_buffer, fps, self.event_trigger_frame, active_telemetry)
            # =============================================
            # EMIT: Frame to UI + persistent session state
            # =============================================
            # Draw UI State for Deep Analysis
            if self.is_full_scan_mode and not self.is_full_scan_complete:
                cv2.rectangle(result.annotated_frame, (0, 0), (result.annotated_frame.shape[1], 40), (255, 144, 30), -1)
                cv2.putText(result.annotated_frame, "🔍 Intelligent Mode: Full Video Deep Scan in Progress... [Please Wait]", 
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif self.is_analyzing_accident:
                cv2.rectangle(result.annotated_frame, (0, 0), (result.annotated_frame.shape[1], 40), (0, 165, 255), -1)
                cv2.putText(result.annotated_frame, "WARNING: Suspicious Event Detected. Deep AI Analysis in Progress... [Please Wait]", 
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()  # .copy() to own the data

            if hasattr(self, "pending_ui_events") and self.pending_ui_events:
                result.events.extend(self.pending_ui_events)
                self.pending_ui_events.clear()

            self.frame_ready.emit(qt_image, result)
            self.session_update.emit(dict(self.session_data), result.frame_number)

        # =============================================
        # END OF SESSION: Summary + Cleanup
        # =============================================

        # Flush partial event buffer if stream ended naturally
        if stream_ended_naturally and self.frames_to_collect > 0 and len(self.active_event_buffer) > 0:
            logger.info("Stream ended before full event window collected. Dispatching partial window.")
            if self.accident_detector and self._slave_pool:
                self.is_analyzing_accident = True
                self._slave_pool.submit(self._run_deep_accident_analysis, list(self.active_event_buffer), fps, self.event_trigger_frame)
                self.active_event_buffer = []

        # Export CSV and emit summary (fires for BOTH natural end and manual stop)
        if self.session_data:
            try:
                self._export_summary_csv()
            except Exception as e:
                logger.warning(f"CSV export failed: {e}")
            self.finished_processing.emit(dict(self.session_data))

        # Emit source_ended only for natural end-of-stream
        if stream_ended_naturally:
            self.source_ended.emit()

        # Cleanup
        cap.release()

        if self._slave_pool:
            if self.is_analyzing_accident:
                logger.info("Waiting for Deep Analysis to finish before shutting down thread pool...")
                self._slave_pool.shutdown(wait=True)
            else:
                self._slave_pool.shutdown(wait=False, cancel_futures=True)
            self._slave_pool = None

        # Unload heavy models to free memory
        try:
            ModelRegistry().unload_all()
        except Exception:
            pass

    # ==========================================
    # 📊 SESSION DATA PERSISTENCE (v2.1)
    # ==========================================

    def _update_session_data(self, result: FrameResult):
        """
        Update the persistent session_data dictionary from current frame.

        This dict accumulates ALL vehicle data across the entire session.
        Slave model results (desc, plate, DMS) are preserved even after
        a vehicle leaves the frame — this prevents UI data loss.

        Uses incremental average speed (O(1) memory) instead of storing
        all speed samples.
        """
        active_ids = set()

        for v in result.vehicles:
            tid = v.track_id
            active_ids.add(tid)

            if tid not in self.session_data:
                # First time seeing this vehicle — initialize entry
                self.session_data[tid] = {
                    "track_id": tid,
                    "speed_kmh": 0.0,
                    "max_speed_kmh": 0.0,
                    "avg_speed_kmh": 0.0,
                    "_speed_count": 0,
                    "_speed_total": 0.0,
                    "direction": "UNKNOWN",
                    "lane": "UNKNOWN",
                    "behavior": "NORMAL",
                    "plate_number": "—",
                    "coco_class": v.coco_class,
                    "first_seen_frame": result.frame_number,
                    "last_seen_frame": result.frame_number,
                    "active": True,
                    "had_wrong_way": False,
                    "wrong_way_duration_s": 0.0,
                    "had_sudden_stop": False,
                    "sudden_stop_duration_s": 0.0,
                    "snapshot_path": "",
                    "vehicle_crop_path": "",
                }

            entry = self.session_data[tid]

            # Update live metrics
            entry["speed_kmh"] = v.speed_kmh
            entry["max_speed_kmh"] = max(entry["max_speed_kmh"], v.speed_kmh)

            # Incremental average speed (O(1) memory)
            if v.speed_kmh > 0:
                entry["_speed_count"] += 1
                entry["_speed_total"] += v.speed_kmh
                entry["avg_speed_kmh"] = int(
                    entry["_speed_total"] / entry["_speed_count"]
                )

            # Only overwrite direction if definitive
            if v.direction != "UNKNOWN":
                entry["direction"] = v.direction

            entry["lane"] = v.lane
            entry["behavior"] = v.behavior
            entry["coco_class"] = v.coco_class
            entry["last_seen_frame"] = result.frame_number
            entry["active"] = True

            # Track behavioral flags (persist even if behavior changes)
            if v.behavior == "WRONG WAY":
                entry["had_wrong_way"] = True
            if v.behavior == "SUDDEN STOP":
                entry["had_sudden_stop"] = True

            # Persist enrichment only if it's a real result (not placeholder)
            if v.plate_number and v.plate_number not in ("—", ""):
                entry["plate_number"] = v.plate_number

        # Extract evidence paths and maintain distinct session events
        for event in result.events:
            ev_id = getattr(event, 'event_id', "")
            if ev_id:
                if ev_id not in self.session_events:
                    self.session_events[ev_id] = {
                        "event_id": ev_id,
                        "track_id": event.vehicle_id,
                        "violation_type": event.event_type,
                        "start_time": event.timestamp,
                        "duration_s": getattr(event, 'duration', 0.0),
                        "max_speed": getattr(event, 'max_speed', 0.0),
                        "plate_number": event.plate_number,
                        "snapshot_path": getattr(event, 'snapshot_path', ""),
                        "vehicle_crop_path": getattr(event, 'vehicle_crop_path', ""),
                        "severity": "HIGH",
                        "description": getattr(event, 'vehicle_desc', "")
                    }
                else:
                    sev = self.session_events[ev_id]
                    sev["duration_s"] = getattr(event, 'duration', 0.0)
                    sev["max_speed"] = max(sev.get("max_speed", 0.0), getattr(event, 'max_speed', 0.0))
                    if event.plate_number and event.plate_number != "—":
                        sev["plate_number"] = event.plate_number
                    if getattr(event, 'snapshot_path', ""):
                        sev["snapshot_path"] = getattr(event, 'snapshot_path', "")
                    if getattr(event, 'vehicle_crop_path', ""):
                        sev["vehicle_crop_path"] = getattr(event, 'vehicle_crop_path', "")
                        
            if event.vehicle_id in self.session_data:
                sd = self.session_data[event.vehicle_id]
                if getattr(event, 'snapshot_path', ""):
                    sd["snapshot_path"] = getattr(event, 'snapshot_path', "")
                if getattr(event, 'vehicle_crop_path', ""):
                    sd["vehicle_crop_path"] = getattr(event, 'vehicle_crop_path', "")
                    
                if event.event_type == "WRONG WAY":
                    sd["wrong_way_duration_s"] = max(sd.get("wrong_way_duration_s", 0.0), getattr(event, 'duration', 0.0))
                elif event.event_type == "SUDDEN STOP":
                    sd["sudden_stop_duration_s"] = max(sd.get("sudden_stop_duration_s", 0.0), getattr(event, 'duration', 0.0))

        # Mark vehicles no longer visible as inactive
        for tid in self.session_data:
            if tid not in active_ids:
                if self.session_data[tid]["active"]:
                    self.session_data[tid]["active"] = False
                    
                    # Cleanup temporary plate states when vehicle leaves scene
                    sd = self.session_data[tid]
                    plate = sd.get("plate_number", "")
                    
                    if "(?)" in plate:
                        clean_plate = plate.replace(" (?)", "").strip()
                        if len(clean_plate) >= 5 and clean_plate not in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY"):
                            sd["plate_number"] = clean_plate
                        else:
                            sd["plate_number"] = "Plate Not Clear"
                    elif plate in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY") or len(plate) < 3:
                        sd["plate_number"] = "Plate Not Clear"
                        
                        # Update any associated session events for CSV
                        for ev_id, sev in self.session_events.items():
                            if sev.get("track_id") == tid:
                                sev_plate = sev.get("plate_number", "")
                                if "(?)" in sev_plate:
                                    clean_plate = sev_plate.replace(" (?)", "").strip()
                                    if len(clean_plate) >= 5 and clean_plate not in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY"):
                                        sev["plate_number"] = clean_plate
                                    else:
                                        sev["plate_number"] = "Plate Not Clear"
                                elif sev_plate in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY") or len(sev_plate) < 3:
                                    sev["plate_number"] = "Plate Not Clear"
                                    
                                    # Inject a UI update event so the Event Alerts panel updates in-place
                                    from core.traffic_analyzer import VehicleEvent
                                    result.events.append(VehicleEvent(
                                        event_id=ev_id,
                                        vehicle_id=tid,
                                        event_type=sev.get("violation_type", ""),
                                        timestamp=sev.get("start_time", ""),
                                        lane=sd.get("lane", "UNKNOWN"),
                                        plate_number="Plate Not Clear",
                                        snapshot_path=sev.get("snapshot_path", ""),
                                        vehicle_crop_path=sev.get("vehicle_crop_path", ""),
                                        duration=sev.get("duration_s", 0.0),
                                        is_update=True,
                                        speed=0.0,
                                        max_speed=sev.get("max_speed", 0.0)
                                    ))

    def _export_summary_csv(self):
        """Export session summary to a timestamped CSV in the project directory."""
        if not self.session_data:
            return

        # Final end-of-video cleanup pass for vehicles still active when stream ended
        for tid, sd in self.session_data.items():
            plate = sd.get("plate_number", "")
            if "(?)" in plate:
                clean_plate = plate.replace(" (?)", "").strip()
                if len(clean_plate) >= 5 and clean_plate not in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY"):
                    sd["plate_number"] = clean_plate
                else:
                    sd["plate_number"] = "Plate Not Clear"
            elif plate in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY") or len(plate) < 3:
                sd["plate_number"] = "Plate Not Clear"
                
        for ev_id, sev in self.session_events.items():
            plate = sev.get("plate_number", "")
            if "(?)" in plate:
                clean_plate = plate.replace(" (?)", "").strip()
                if len(clean_plate) >= 5 and clean_plate not in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY"):
                    sev["plate_number"] = clean_plate
                else:
                    sev["plate_number"] = "Plate Not Clear"
            elif plate in ("Pending", "Low Quality", "Unknown", "—", "N/A", "LOW_QUALITY") or len(plate) < 3:
                sev["plate_number"] = "Plate Not Clear"

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{SUMMARY_CSV_FILENAME}_{timestamp}.csv"
            project_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            filepath = os.path.join(project_dir, filename)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Track ID", "Plate Number",
                    "Max Speed (km/h)", "Avg Speed (km/h)",
                    "Direction", "Lane", "Last Behavior",
                    "Wrong Way", "Wrong Way Duration (s)", 
                    "Sudden Stop", "Sudden Stop Duration (s)",
                    "First Frame", "Last Frame",
                    "Snapshot Path", "Vehicle Crop Path"
                ])
                for tid in sorted(self.session_data.keys()):
                    v = self.session_data[tid]
                    writer.writerow([
                        v["track_id"],
                        v["plate_number"],
                        v["max_speed_kmh"],
                        v["avg_speed_kmh"],
                        v["direction"],
                        v["lane"],
                        v["behavior"],
                        "YES" if v.get("had_wrong_way") else "NO",
                        round(v.get("wrong_way_duration_s", 0.0), 1),
                        "YES" if v.get("had_sudden_stop") else "NO",
                        round(v.get("sudden_stop_duration_s", 0.0), 1),
                        v["first_seen_frame"],
                        v["last_seen_frame"],
                        v.get("snapshot_path", ""),
                        v.get("vehicle_crop_path", ""),
                    ])

            logger.info(f"Session summary exported: {filepath}")

            if hasattr(self, 'session_events') and self.session_events:
                events_filename = f"violations_events_{timestamp}.csv"
                events_filepath = os.path.join(project_dir, events_filename)
                with open(events_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Event ID", "Track ID", "Violation Type", "Start Time", 
                        "Duration (s)", "Max Speed (km/h)", "Plate Number", 
                        "Severity", "Description", "Snapshot Path", "Crop Path"
                    ])
                    for ev in self.session_events.values():
                        writer.writerow([
                            ev["event_id"],
                            ev["track_id"],
                            ev["violation_type"],
                            ev["start_time"],
                            round(ev["duration_s"], 1),
                            ev["max_speed"],
                            ev["plate_number"],
                            ev["severity"],
                            ev["description"],
                            ev["snapshot_path"],
                            ev["vehicle_crop_path"]
                        ])
                logger.info(f"Violations events exported: {events_filepath}")

        except Exception as e:
            logger.error(f"CSV export failed: {e}")

    # ==========================================
    # 🔧 SLAVE MODEL DISPATCH (v2.1 — area-gated)
    # ==========================================

    def _dispatch_slave_tasks(self, frame: np.ndarray, result: FrameResult):
        """
        Dispatch slave model tasks with bbox area thresholds.

        v2.1 Changes:
            - Only dispatches when bbox_area >= MIN_BBOX_AREA_FOR_SLAVE.
            - DMS requires even larger area (MIN_BBOX_AREA_FOR_DMS).
            - Prevents wasting inference on tiny, unreadable vehicle crops.
        """
        if self._slave_pool is None:
            return

        for vehicle in result.vehicles:
            tid = vehicle.track_id
            x1, y1, x2, y2 = self._get_vehicle_bbox(vehicle, frame)
            bbox_area = (x2 - x1) * (y2 - y1)

            # --- Gate: skip vehicles with bbox too small ---
            if bbox_area < MIN_BBOX_AREA_FOR_SLAVE:
                continue

            # --- Plate Reading (YOLO + EasyOCR) ---
            if self.analyzer.should_dispatch_plate(tid, bbox_area, result.frame_number):
                best_crop = self.analyzer.best_bbox_crop.get(tid)
                if best_crop is not None:
                    self.analyzer.plate_dispatched.add(tid)
                    crop_copy = best_crop.copy()
                    self._slave_pool.submit(
                        self._run_plate_read, tid, crop_copy, result.frame_number
                    )

    def _get_vehicle_bbox(self, vehicle, frame):
        """Extract bbox coordinates from VehicleInfo via tracking history."""
        hist = self.analyzer.track_history.get(vehicle.track_id)
        if hist and hist["centers_x"] and hist["centers_y"] and hist["heights"]:
            cx = hist["centers_x"][-1]
            cy = hist["centers_y"][-1]
            bh = hist["heights"][-1]
            bw = bh * 1.2  # Approximate aspect ratio
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            return x1, y1, x2, y2

        # Fallback — use frame center
        h, w = frame.shape[:2]
        return 0, 0, w, h

    # ==========================================
    # ⚙️ SLAVE MODEL WORKERS (v2.1 — robust error handling)
    # ==========================================
    # These methods run in ThreadPoolExecutor threads.
    # They update BOTH analyzer caches AND session_data directly.
    # Failures return meaningful fallback values, never crash.

    def _run_plate_read(self, track_id: int, crop: np.ndarray, frame_number: int = 0):
        """Run YOLO+TrOCR plate reading with temporal voting."""
        try:
            from core.plate_reader import read_plate
            import collections
            
            results = read_plate(crop, track_id=track_id, frame_num=frame_number)
            
            if results and results[0].get("recognition_status") == "SUCCESS":
                plate_text = results[0]["plate_text"]
                if plate_text and len(plate_text) >= 3:
                    self.analyzer.plate_history[track_id].append(plate_text)
                
            reads = self.analyzer.plate_history[track_id]
            
            if not reads:
                if results and results[0].get("recognition_status") == "LOW_QUALITY":
                    result_text = "Low Quality"
                else:
                    result_text = "Pending"
            else:
                # Get the most common read (mode)
                most_common = collections.Counter(reads).most_common(1)[0]
                best_plate, count = most_common[0], most_common[1]
                
                invalid_plates = ("Unknown", "Low Quality", "Plate Not Clear", "N/A", "—", "LOW_QUALITY")
                
                # Lock if we have 5 consistent consensus reads of a valid plate
                if count >= 5 and len(best_plate) >= 5 and best_plate not in invalid_plates:
                    self.analyzer.plate_locked.add(track_id)
                    result_text = best_plate
                    logger.debug(f"[PLATE] Track {track_id} LOCKED on '{best_plate}' after {count} identical reads.")
                else:
                    # Append a pending indicator to show it's still voting
                    result_text = f"{best_plate} (?)"
                    logger.debug(f"[PLATE] Track {track_id} voting: '{best_plate}' ({count} identical reads).")

        except Exception as e:
            logger.warning(f"[PLATE] Track {track_id} failed: {e}")
            result_text = "Unknown"
        finally:
            self.analyzer.plate_dispatched.discard(track_id)

        self.analyzer.plate_cache[track_id] = result_text
        if track_id in self.session_data:
            self.session_data[track_id]["plate_number"] = result_text
        logger.debug(f"[PLATE] Track {track_id}: {result_text}")

    def _build_structured_report(self, trigger_frame, conf, gemini_res, evidence_paths, incident_id, incident_folder, report_path, error_msg=None, local_fallback_used=False):
        import os
        
        status_val = "Unknown"
        gemini_status = "Not queried"
        gemini_enabled = True
        
        fault_vid = "Undetermined"
        fault_plate = "Unknown"
        fault_reason = "No AI verification available. Local fallback used."
        what_happened = ["Local VideoMAE analysis detected a sudden collision anomaly.", f"Trigger frame detected at {int(trigger_frame)}."]
        summary_val = "AI Traffic Analysis found an anomaly with high confidence."
        
        if error_msg:
            status_val = "System Error"
            gemini_status = "Error"
            local_fallback_used = True
            gemini_enabled = False
            fault_reason = f"System Error: {error_msg}"
            what_happened = ["System crashed during intelligent scan.", error_msg]
        elif not gemini_res and conf < 0.2:
            status_val = "Normal / No Accident"
            gemini_status = "Not queried"
            gemini_enabled = False
            fault_reason = f"Analysis Complete. Intelligent scan found no accident (Max conf {conf:.2f})."
            what_happened = ["System completed intelligent scan.", "No significant threshold breaks."]
        else:
            if gemini_res:
                gemini_status_flag = gemini_res.get("gemini_status", "Success")
                if gemini_status_flag == "failed":
                    status_val = "AI Verification Unavailable"
                    gemini_status = "failed"
                    local_fallback_used = True
                    fault_reason = "API failed or timed out. Local VideoMAE confidence used."
                    fault_vid = "Undetermined"
                else:
                    if gemini_res.get("accident_confirmed") is True:
                        status_val = "Confirmed Accident"
                        gemini_status = "Success"
                        local_fallback_used = False
                    elif gemini_res.get("accident_confirmed") is False:
                        status_val = "Suspected Accident / False Alarm"
                        gemini_status = "Success"
                        local_fallback_used = False
                    else:
                        status_val = "AI Verification Unavailable"
                        gemini_status = gemini_res.get("reason", "API Error or Quota Exceeded")
                        local_fallback_used = True
                        
                if not local_fallback_used:
                    fault_info = gemini_res.get("likely_at_fault", {})
                    decision = fault_info.get("decision", "insufficient_evidence")
                    reasoning = fault_info.get("reasoning", "No reason provided.")
                    tid = fault_info.get("track_id")
                    
                    if decision == "likely_at_fault" and tid is not None:
                        fault_vid = f"Vehicle {tid}"
                        # Try parsing as int to lookup session data
                        try:
                            v_id = int(tid)
                            if v_id in self.session_data:
                                fault_plate = self.session_data[v_id].get("plate_number", "Unknown")
                        except ValueError:
                            pass
                            
                    fault_reason = reasoning
                    
                    # Build what_happened from sequence_of_events
                    seq = gemini_res.get("sequence_of_events", [])
                    if seq:
                        what_happened = [f"{s.get('description', '')}" for s in seq]
                    else:
                        what_happened = [
                            "AI Analysis identified accident event.",
                            f"Fault reasoning: {fault_reason[:150]}..."
                        ]
                        
                    # Build summary
                    exec_summary = gemini_res.get("summary")
                    if exec_summary:
                        summary_val = exec_summary
                        
        # Extract vehicle list for UI
        ui_vehicles = []
        if gemini_res and not local_fallback_used:
            ui_vehicles = gemini_res.get("vehicles_involved", [])

        return {
            "report_title": "Accident Investigation Report",
            "system_name": "RoadVision AI",
            "subtitle": "Real-Time Traffic Intelligence System",
            "incident_id": incident_id,
            "status": status_val,
            "accident_confidence": float(conf),
            "accident_confidence_percent": f"{conf*100:.1f}%",
            "frame": int(trigger_frame),
            "timestamp_sec": "Unknown",
            "incident_folder": incident_folder,
            "report_path": report_path,
            "pdf_report_path": os.path.join(incident_folder, "accident_report.pdf") if incident_folder else None,
            "evidence_paths": evidence_paths,
            "evidence_meta": gemini_res.get("evidence_meta", []) if gemini_res else [],
            "video_source": "Unknown",
            
            # Legacy Fields for UI backward compatibility
            "primary_at_fault_id": None,
            "fault_analysis": fault_reason,
            "confirmed_accident": gemini_res.get("accident_confirmed", False) if gemini_res else False,
            
            # New Extracted Fields
            "executive_summary": summary_val,
            "evidence_review": gemini_res.get("summary", "Local VideoMAE analysis used.") if gemini_res else "Local VideoMAE analysis used.",
            "vehicles": ui_vehicles,
            "sequence_of_events": what_happened,
            "contributing_factors": gemini_res.get("contributing_factors", []) if gemini_res else [],
            
            "fault_assessment": {
                "likely_responsible_vehicle_id": str(fault_vid),
                "likely_responsible_plate": str(fault_plate),
                "confidence": gemini_res.get("likely_at_fault", {}).get("confidence", "Unknown") if gemini_res else "Low",
                "reason": str(fault_reason)
            },
            
            "confidence_assessment": "",
            "final_determination": summary_val,
            
            # Legacy wrapper object (some old UI code may use it)
            "fault_vehicle": {
                "vehicle_id": str(fault_vid),
                "plate_number": str(fault_plate),
                "reason": str(fault_reason)
            },
            "summary": summary_val,
            "what_happened": what_happened,
            "evidence": {
                "evidence_frames": evidence_paths,
                "snapshot_paths": []
            },
            "technical_details": {
                "model": "VideoMAE",
                "gemini_enabled": gemini_enabled,
                "gemini_status": gemini_status,
                "raw_gemini_log": gemini_res.get("raw", "") if gemini_res else "",
                "local_fallback_used": local_fallback_used
            }
        }

    def _run_deep_accident_analysis(self, event_buffer: list, fps: float, trigger_frame: int, telemetry: dict = None):
        import traceback
        import os
        import json
        import time
        import numpy as np
        import cv2
        import logging
        logger = logging.getLogger(__name__)
        from utils.constants import (
            ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION, ACCIDENT_GEMINI_JPEG_QUALITY,
            ACCIDENT_ANALYSIS_MAX_FRAMES, ACCIDENT_BLUR_THRESHOLD,
            ACCIDENT_DUPLICATE_THRESHOLD, DEBUG_ACCIDENT_FRAME_SELECTION,
            CV_COLOR_NORMAL, ACCIDENT_MAX_EVIDENCE_IMAGES
        )
        from modules.accident_detection.llm_verifier import gemini_gatekeeper

        out_dir = os.path.abspath("accident_outputs")
        incident_id = f"INC-{time.strftime('%Y%m%d-%H%M%S')}-F{trigger_frame}"
        incident_folder = os.path.join(out_dir, incident_id)
        evidence_folder = os.path.join(incident_folder, "clean_evidence")
        analysis_folder = os.path.join(incident_folder, "analysis_frames")
        report_path = os.path.join(incident_folder, "accident_report.json")
        metadata_path = os.path.join(incident_folder, "metadata.json")

        os.makedirs(evidence_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)
        if DEBUG_ACCIDENT_FRAME_SELECTION:
            os.makedirs(os.path.join(incident_folder, "debug"), exist_ok=True)

        final_result = None
        is_true_accident = False
        start_prep_time = time.time()

        try:
            logger.info(f"[Deep Analysis] Scanning {len(event_buffer)}-frame buffer")
            frames_for_mae = [f["small_frame"] for f in event_buffer]
            scan_result = self.accident_detector.scan_buffer_with_videomae(frames_for_mae, step_frames=5)
            
            conf = scan_result['accident_confidence']
            is_accident = scan_result['prediction'] == 'Accident'
            
            if is_accident and conf > 0.5:
                is_true_accident = True
                
                # Intelligent Frame Sampling
                peak_start = scan_result['peak_start_idx']
                peak_end = scan_result['peak_end_idx']
                impact_idx = (peak_start + peak_end) // 2
                
                # Align exact trigger if it's nearby
                for i, f in enumerate(event_buffer):
                    if f["is_trigger"]:
                        if abs(i - impact_idx) < int(fps):
                            impact_idx = i
                        break
                        
                impact_time = event_buffer[impact_idx]["time_sec"]
                
                target_offsets = {
                    "before_context": [-4.0, -3.0, -2.2, -1.6],
                    "final_approach": [-1.2, -0.9, -0.6, -0.3],
                    "collision": [-0.15, 0.0, 0.15, 0.35],
                    "aftermath": [0.8, 1.6, 2.5]
                }
                
                valid_indices = [i for i, f in enumerate(event_buffer) if f["jpeg_bytes"] is not None or i == impact_idx]
                if not valid_indices:
                    valid_indices = [i for i, f in enumerate(event_buffer)]
                    
                selected_frames_meta = []
                frame_counter = 1
                
                for category, offsets in target_offsets.items():
                    for offset in offsets:
                        if len(selected_frames_meta) >= ACCIDENT_ANALYSIS_MAX_FRAMES:
                            break
                        target_t = impact_time + offset
                        
                        closest_idx = min(valid_indices, key=lambda x: abs(event_buffer[x]["time_sec"] - target_t))
                        
                        if any(s["buffer_idx"] == closest_idx for s in selected_frames_meta):
                            continue
                            
                        if offset == 0.0:
                            closest_idx = impact_idx
                            
                        buf_frame = event_buffer[closest_idx]
                        if buf_frame.get("jpeg_bytes") is not None:
                            img_arr = np.frombuffer(buf_frame["jpeg_bytes"], np.uint8)
                            clean_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        else:
                            clean_img = buf_frame["small_frame"]
                            
                        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        is_dup = False
                        dup_score = 0.0
                        if len(selected_frames_meta) > 0:
                            prev_gray = selected_frames_meta[-1]["gray_cache"]
                            hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                            hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                            res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                            dup_score = res[0] if isinstance(res, tuple) else res
                            if dup_score > ACCIDENT_DUPLICATE_THRESHOLD and offset != 0.0:
                                is_dup = True
                                
                        if (blur_score < ACCIDENT_BLUR_THRESHOLD or is_dup) and offset != 0.0:
                            best_neighbor = closest_idx
                            best_score = blur_score
                            for neighbor_idx in valid_indices:
                                if neighbor_idx != closest_idx and abs(event_buffer[neighbor_idx]["time_sec"] - target_t) < 0.3:
                                    n_frame = event_buffer[neighbor_idx]
                                    if n_frame.get("jpeg_bytes"):
                                        n_img_arr = np.frombuffer(n_frame["jpeg_bytes"], np.uint8)
                                        n_clean_img = cv2.imdecode(n_img_arr, cv2.IMREAD_COLOR)
                                        n_gray = cv2.cvtColor(n_clean_img, cv2.COLOR_BGR2GRAY)
                                        n_blur = cv2.Laplacian(n_gray, cv2.CV_64F).var()
                                        if n_blur > best_score:
                                            best_score = n_blur
                                            best_neighbor = neighbor_idx
                                            clean_img = n_clean_img
                                            gray = n_gray
                            closest_idx = best_neighbor
                            buf_frame = event_buffer[closest_idx]
                            
                        fid = f"F{frame_counter:02d}"
                        clean_path = os.path.join(evidence_folder, f"{fid}_clean.jpg")
                        cv2.imwrite(clean_path, clean_img)
                        
                        h, w = clean_img.shape[:2]
                        if max(h, w) > ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION:
                            scale = ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION / max(h, w)
                            analysis_img = cv2.resize(clean_img, (int(w * scale), int(h * scale)))
                        else:
                            analysis_img = clean_img.copy()
                            scale = 1.0
                            
                        for tid, tdata in buf_frame.get("tracking_data", {}).items():
                            bbox = tdata["bbox"]
                            x1, y1 = int(bbox[0]*scale), int(bbox[1]*scale)
                            x2, y2 = int(bbox[2]*scale), int(bbox[3]*scale)
                            cv2.rectangle(analysis_img, (x1, y1), (x2, y2), CV_COLOR_NORMAL, 2)
                            cv2.putText(analysis_img, f"ID: {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CV_COLOR_NORMAL, 2)
                            
                        analysis_path = os.path.join(analysis_folder, f"{fid}_analysis.jpg")
                        cv2.imwrite(analysis_path, analysis_img, [cv2.IMWRITE_JPEG_QUALITY, ACCIDENT_GEMINI_JPEG_QUALITY])
                        
                        selected_frames_meta.append({
                            "analysis_frame_id": fid,
                            "video_frame_number": buf_frame["frame_num"],
                            "timestamp_seconds": round(buf_frame["time_sec"], 2),
                            "relative_to_impact_seconds": round(buf_frame["time_sec"] - impact_time, 2),
                            "category": category,
                            "selection_reason": f"Sampled for {category}",
                            "blur_score": blur_score,
                            "duplicate_score": float(dup_score),
                            "clean_path": clean_path,
                            "analysis_path": analysis_path,
                            "buffer_idx": closest_idx,
                            "gray_cache": gray
                        })
                        frame_counter += 1

                prep_time = time.time() - start_prep_time
                logger.info(f"[REPORT TIMING] frame_selection_and_prep={prep_time:.2f}s")
                
                if DEBUG_ACCIDENT_FRAME_SELECTION:
                    with open(os.path.join(incident_folder, "debug", "frame_selection.json"), "w") as df:
                        debug_data = [{k: v for k, v in meta.items() if k != "gray_cache"} for meta in selected_frames_meta]
                        json.dump(debug_data, df, indent=2)

                start_gemini_time = time.time()
                logger.info("[Gemini] Querying Gemini for verification...")
                gemini_res = gemini_gatekeeper(selected_frames_meta, telemetry or {}, use_gemini=True)
                gemini_time = time.time() - start_gemini_time
                logger.info(f"[REPORT TIMING] gemini_request={gemini_time:.2f}s")
                
                start_pdf_time = time.time()
                if gemini_res.get("gemini_status") == "failed" or not gemini_res.get("accident_confirmed"):
                    logger.warning("[Gemini] API failed/timeout or No Accident. Local fallback used.")
                    evidence_paths = [m["clean_path"] for m in selected_frames_meta][:ACCIDENT_MAX_EVIDENCE_IMAGES]
                else:
                    valid_ids = gemini_res.get("evidence_frame_ids", [])
                    evidence_paths = []
                    for fid in valid_ids:
                        for m in selected_frames_meta:
                            if m["analysis_frame_id"] == fid:
                                evidence_paths.append(m["clean_path"])
                                break
                    if not evidence_paths:
                        evidence_paths = [m["clean_path"] for m in selected_frames_meta][:ACCIDENT_MAX_EVIDENCE_IMAGES]
                        
                evidence_paths = evidence_paths[:ACCIDENT_MAX_EVIDENCE_IMAGES]
                
                evidence_meta = []
                for ep in evidence_paths:
                    fid = os.path.basename(ep).split("_")[0]
                    for m in selected_frames_meta:
                        if m["analysis_frame_id"] == fid:
                            evidence_meta.append({
                                "path": ep,
                                "id": fid,
                                "timestamp": m["timestamp_seconds"],
                                "relative_time": m["relative_to_impact_seconds"],
                                "category": m["category"],
                                "selection_reason": m["selection_reason"]
                            })
                            break
                if gemini_res:
                    gemini_res["evidence_meta"] = evidence_meta
                        
                final_result = self._build_structured_report(trigger_frame, conf, gemini_res, evidence_paths, incident_id, incident_folder, report_path)
                pdf_time = time.time() - start_pdf_time
                logger.info(f"[REPORT TIMING] json_and_pdf_generation={pdf_time:.2f}s")
                logger.info(f"[REPORT TIMING] total={(prep_time + gemini_time + pdf_time):.2f}s")
            else:
                gemini_res = {
                    "accident_confirmed": False,
                    "accident_confidence": float(conf),
                    "gemini_status": "skipped",
                    "reason": f"VideoMAE rejected the event. Peak confidence ({conf:.2f}) was below threshold."
                }
                final_result = self._build_structured_report(trigger_frame, conf, gemini_res, [], incident_id, incident_folder, report_path)
                
        except Exception as e:
            err_msg = traceback.format_exc()
            logger.error(f"[Deep Analysis] Failed: {e}\n{err_msg}")
            final_result = self._build_structured_report(trigger_frame, 0.0, None, [], incident_id, incident_folder, report_path, error_msg=str(e))
            
        finally:
            if final_result:
                with open(report_path, "w") as f:
                    from modules.accident_detection.llm_verifier import sanitize_data
                    clean_report = sanitize_data(final_result)
                    json.dump(clean_report, f, indent=4)
                    
                with open(metadata_path, "w") as f:
                    json.dump({
                        "incident_id": incident_id,
                        "timestamp": time.time(),
                        "trigger_frame": trigger_frame,
                        "status": final_result.get("status")
                    }, f, indent=4)
                    
                if is_true_accident:
                    self._link_accident_sudden_stops(final_result)
                    self.accident_detected.emit(final_result)
                else:
                    self.scan_complete.emit(final_result)
            
            self.is_analyzing_accident = False

    def _run_full_video_scan(self, video_path: str):
        import traceback
        import os
        import json
        import time
        import numpy as np
        import cv2
        import logging
        logger = logging.getLogger(__name__)
        from utils.constants import (
            ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION, ACCIDENT_GEMINI_JPEG_QUALITY,
            ACCIDENT_ANALYSIS_MAX_FRAMES, ACCIDENT_BLUR_THRESHOLD,
            ACCIDENT_DUPLICATE_THRESHOLD, DEBUG_ACCIDENT_FRAME_SELECTION,
            CV_COLOR_NORMAL, ACCIDENT_MAX_EVIDENCE_IMAGES
        )
        from modules.accident_detection.llm_verifier import gemini_gatekeeper

        out_dir = os.path.abspath("accident_outputs")
        incident_id = f"INC-{time.strftime('%Y%m%d-%H%M%S')}-SCAN"
        incident_folder = os.path.join(out_dir, incident_id)
        evidence_folder = os.path.join(incident_folder, "clean_evidence")
        analysis_folder = os.path.join(incident_folder, "analysis_frames")
        report_path = os.path.join(incident_folder, "accident_report.json")
        metadata_path = os.path.join(incident_folder, "metadata.json")
        
        os.makedirs(evidence_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)
        final_result = None
        MIN_CONFIDENCE = 0.2
        is_true_accident = False
        
        try:
            logger.info(f"[Intelligent Scan] Scanning entire video: {video_path}")
            df = self.accident_detector.scan_video_with_videomae(video_path, window_seconds=5.0, step_seconds=1.0)
            
            if df is None or df.empty:
                raise ValueError("VideoMAE returned an empty dataframe. Scan failed.")
                
            best_row = df.loc[df['accident_confidence'].idxmax()]
            conf = float(best_row['accident_confidence'])
            is_accident = (conf > MIN_CONFIDENCE)
            trigger_frame = best_row['frame_indices'][len(best_row['frame_indices'])//2] if 'frame_indices' in best_row and len(best_row['frame_indices']) > 0 else 0
            
            incident_id = f"INC-{time.strftime('%Y%m%d-%H%M%S')}-F{int(trigger_frame)}"
            incident_folder = os.path.join(out_dir, incident_id)
            evidence_folder = os.path.join(incident_folder, "clean_evidence")
            analysis_folder = os.path.join(incident_folder, "analysis_frames")
            report_path = os.path.join(incident_folder, "accident_report.json")
            metadata_path = os.path.join(incident_folder, "metadata.json")
            os.makedirs(evidence_folder, exist_ok=True)
            os.makedirs(analysis_folder, exist_ok=True)
            
            logger.info(f"[Intelligent Scan] Peak Window: {best_row['start_sec']}s - {best_row['end_sec']}s. Conf: {conf:.2f}")
            
            if is_accident:
                is_true_accident = True
                from modules.accident_detection.video_helpers import read_video_frames
                
                frames, _ = read_video_frames(video_path, float(best_row['start_sec']), float(best_row['end_sec']), self.accident_detector.num_frames)
                indices = np.linspace(0, len(frames) - 1, 9, dtype=int)
                extracted_frames = [frames[i] for i in indices]
                
                selected_frames_meta = []
                evidence_paths = []
                for i, f in enumerate(extracted_frames):
                    fid = f"F{i+1:02d}"
                    p = os.path.join(evidence_folder, f"{fid}_clean.jpg")
                    cv2.imwrite(p, f)
                    evidence_paths.append(p)
                    
                    h, w = f.shape[:2]
                    if max(h, w) > ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION:
                        scale = ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION / max(h, w)
                        analysis_img = cv2.resize(f, (int(w * scale), int(h * scale)))
                    else:
                        analysis_img = f.copy()
                        
                    ap = os.path.join(analysis_folder, f"{fid}_analysis.jpg")
                    cv2.imwrite(ap, analysis_img, [cv2.IMWRITE_JPEG_QUALITY, ACCIDENT_GEMINI_JPEG_QUALITY])
                    
                    selected_frames_meta.append({
                        "analysis_frame_id": fid,
                        "video_frame_number": int(trigger_frame),
                        "timestamp_seconds": float(best_row['start_sec']) + (i * 0.5),
                        "relative_to_impact_seconds": round(-2.0 + (i * 0.5), 2),
                        "category": "scan_window",
                        "selection_reason": "Equidistant sampling across peak scan window.",
                        "blur_score": 0.0,
                        "duplicate_score": 0.0,
                        "clean_path": p,
                        "analysis_path": ap
                    })

                active_telemetry = {}
                for tid, data in self.session_data.items():
                    active_telemetry[tid] = {
                        "track_id": data["track_id"],
                        "max_speed_kmh": data["max_speed_kmh"],
                        "avg_speed_kmh": data["avg_speed_kmh"],
                        "behavior": data["behavior"],
                        "lane": data["lane"],
                        "had_wrong_way": data["had_wrong_way"],
                        "had_sudden_stop": data["had_sudden_stop"]
                    }

                logger.info("[Gemini] Querying Gemini for verification...")
                gemini_res = gemini_gatekeeper(selected_frames_meta, active_telemetry, use_gemini=True)
                
                if gemini_res.get("gemini_status") == "failed" or not gemini_res.get("accident_confirmed"):
                    evidence_paths = evidence_paths[:ACCIDENT_MAX_EVIDENCE_IMAGES]
                else:
                    valid_ids = gemini_res.get("evidence_frame_ids", [])
                    evidence_paths = []
                    for fid in valid_ids:
                        for m in selected_frames_meta:
                            if m["analysis_frame_id"] == fid:
                                evidence_paths.append(m["clean_path"])
                                break
                    if not evidence_paths:
                        evidence_paths = [m["clean_path"] for m in selected_frames_meta][:ACCIDENT_MAX_EVIDENCE_IMAGES]
                        
                evidence_paths = evidence_paths[:ACCIDENT_MAX_EVIDENCE_IMAGES]
                
                evidence_meta = []
                for ep in evidence_paths:
                    fid = os.path.basename(ep).split("_")[0]
                    for m in selected_frames_meta:
                        if m["analysis_frame_id"] == fid:
                            evidence_meta.append({
                                "path": ep,
                                "id": fid,
                                "timestamp": m["timestamp_seconds"],
                                "relative_time": m["relative_to_impact_seconds"],
                                "category": m["category"],
                                "selection_reason": m["selection_reason"]
                            })
                            break
                if gemini_res:
                    gemini_res["evidence_meta"] = evidence_meta
                
                final_result = self._build_structured_report(trigger_frame, conf, gemini_res, evidence_paths, incident_id, incident_folder, report_path)
            else:
                final_result = self._build_structured_report(trigger_frame, conf, None, [], incident_id, incident_folder, report_path, local_fallback_used=True)
                
        except Exception as e:
            err_msg = traceback.format_exc()
            logger.error(f"[Intelligent Scan] Failed: {e}\n{err_msg}")
            if not final_result:
                incident_id = f"INC-{time.strftime('%Y%m%d-%H%M%S')}-FAIL"
                incident_folder = os.path.join(out_dir, incident_id)
                report_path = os.path.join(incident_folder, "accident_report.json")
                os.makedirs(incident_folder, exist_ok=True)
            final_result = self._build_structured_report(0, 0.0, None, [], incident_id, incident_folder, report_path, error_msg=str(e))
            
        finally:
            if final_result:
                with open(report_path, "w") as f:
                    from modules.accident_detection.llm_verifier import sanitize_data
                    clean_report = sanitize_data(final_result)
                    json.dump(clean_report, f, indent=4)
                    
                try:
                    from modules.reporting.pdf_report_generator import generate_pdf_from_json
                    pdf_out = generate_pdf_from_json(report_path)
                    if pdf_out:
                        logger.debug(f"[ACCIDENT_DEBUG] PDF report generated at {pdf_out}")
                except Exception as e:
                    logger.error(f"Failed to generate PDF report: {e}")
                    
                metadata_path = os.path.join(incident_folder, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump({
                        "incident_id": final_result.get("incident_id"),
                        "timestamp": time.time(),
                        "trigger_frame": final_result.get("frame"),
                        "status": final_result.get("status"),
                        "video_source": video_path
                    }, f, indent=4)
                    
                if is_true_accident:
                    self.accident_detected.emit(final_result)
                else:
                    self.scan_complete.emit(final_result)
            
            self.is_analyzing_accident = False
            self.is_full_scan_complete = True
