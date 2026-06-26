"""
Traffic Analyzer — Core AI processing logic (v2.0 — Multi-Model Pipeline).

Major changes from v1.0:
    1. REPLACED: K/bbox_height speed estimation → Bird's-Eye View (BEV)
       homography-based calibrated speed computation.
    2. ADDED: Slave model orchestration for vehicle description, plate
       reading, and driver monitoring.
    3. EXPANDED: FrameResult now includes per-vehicle VehicleInfo with
       type, color, plate number, and driver status.
    4. MAINTAINED: Wrong-way detection logic (unchanged).

Architecture:
    Master: YOLOv8 + ByteTrack (runs every frame)
    Slaves: BLIP, YOLO-Plate+EasyOCR, CLIP-DMS (run once per Track_ID)
"""

import cv2
import time
import math
import os
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional, Dict
from ultralytics import YOLO

from core.calibration import BirdEyeViewCalibrator, SpeedSmoother
from utils.constants import (
    YOLO_MODEL, YOLO_CONFIDENCE, VEHICLE_CLASSES, COCO_CLASS_NAMES,
    HISTORY_LENGTH, SPEED_FRAME_DIFF, DIRECTION_FRAME_DIFF,
    DIRECTION_THRESHOLD, DENSITY_JAM_VEHICLES, DENSITY_JAM_SPEED,
    DENSITY_HIGH_VEHICLES, DENSITY_MEDIUM_VEHICLES,
    SUDDEN_STOP_HIGH_SPEED, SUDDEN_STOP_LOW_SPEED,
    SLOW_SPEED_THRESHOLD, STOPPED_SPEED_THRESHOLD,
    SPEED_MAX_PLAUSIBLE,
    CV_COLOR_NORMAL, CV_COLOR_WRONG_WAY, CV_COLOR_SUDDEN_STOP,
    CV_COLOR_TEXT, CV_COLOR_LANE_LINE, CV_COLOR_PLATE,
    DIRECTION_MODE_STANDARD, DIRECTION_RULES,
    DIRECTION_MODE_ONEWAY_UP, DIRECTION_MODE_ONEWAY_DOWN,
    PLATE_READ_PATIENCE_FRAMES,
    VIDEOMAE_NUM_FRAMES, ACCIDENT_IOU_TRIGGER, 
    ACCIDENT_CENTER_DIST_FACTOR, ACCIDENT_SUDDEN_STOP_RATIO,
    VIOLATION_REARM_SECONDS, EVIDENCE_UPDATE_SECONDS,
    WRONG_WAY_ANGLE_THRESHOLD_DEG, WRONG_WAY_MIN_REVERSE_DISTANCE_PX,
    WRONG_WAY_MIN_CONSECUTIVE_FRAMES, WRONG_WAY_MIN_STEP_DISTANCE_PX,
    SUDDEN_STOP_MIN_PREV_SPEED, SUDDEN_STOP_MAX_FINAL_SPEED,
    SUDDEN_STOP_MIN_DROP, SUDDEN_STOP_CONFIRM_FRAMES,
    SUDDEN_STOP_COOLDOWN_SEC, ACCIDENT_SPEED_DROP_WINDOW
)

logger = logging.getLogger(__name__)


# ==========================================
# 📦 DATA CLASSES
# ==========================================

@dataclass
class VehicleEvent:
    """Represents a detected traffic event."""
    vehicle_id: int
    event_type: str      # "WRONG WAY" or "SUDDEN STOP"
    timestamp: str       # HH:MM:SS
    lane: str            # "LEFT" or "RIGHT"
    plate_number: str = ""
    snapshot_path: str = ""
    vehicle_crop_path: str = ""
    event_id: str = ""
    duration: float = 0.0
    is_update: bool = False
    speed: float = 0.0
    max_speed: float = 0.0


@dataclass
class VehicleInfo:
    """
    Rich per-vehicle data for the dashboard table.
    --- NEW in v2.0 ---
    """
    track_id: int
    speed_kmh: float
    direction: str        # "UP" / "DOWN" / "UNKNOWN"
    lane: str             # "LEFT" / "RIGHT"
    behavior: str         # "NORMAL" / "WRONG WAY" / "SUDDEN STOP" / etc.
    plate_number: str     # "ABC 1234" or "—"
    coco_class: int = 2   # COCO class ID for fallback


@dataclass
class FrameResult:
    """Contains all data from processing a single frame."""
    annotated_frame: np.ndarray
    vehicle_count: int = 0
    average_speed: int = 0
    density_status: str = "LOW"
    events: List[VehicleEvent] = field(default_factory=list)
    vehicles: List[VehicleInfo] = field(default_factory=list)  # NEW in v2.0
    fps: float = 0.0
    frame_number: int = 0


# ==========================================
# 🧠 TRAFFIC ANALYZER (v2.0)
# ==========================================

class TrafficAnalyzer:
    """
    Core traffic analysis engine with multi-model pipeline support.

    v2.0 Changes:
        - BEV-calibrated speed estimation (replaces K/bbox_height)
        - Slave model result caching and orchestration
        - Expanded per-vehicle metadata
    """

    def __init__(self, roi_config=None):
        self.model: Optional[YOLO] = None
        self._model_loaded = False

        self.roi_config = roi_config
        # --- BEV Calibrator (CHANGED: replaces K constant) ---
        self.calibrator = BirdEyeViewCalibrator()
        self._calibrator_initialized = False
        self.show_roi_overlay = True if self.roi_config is None else self.roi_config.get("show_roi_overlay", True)

        # Per-vehicle tracking history
        self.track_history: Dict[int, dict] = defaultdict(lambda: {
            "centers_x": [],
            "centers_y": [],
            "heights": [],
            "speeds": [],
            "last_seen_frame": 0,
            "speed_smoother": SpeedSmoother(),
            "sudden_stop_frames": 0,
            "last_sudden_stop_time": 0.0,
        })
        
        self.wrong_way_state: Dict[int, dict] = defaultdict(lambda: {
            "reverse_distance_px": 0.0,
            "consecutive_reverse_frames": 0,
            "confirmed": False
        })

        # These are populated asynchronously by the VideoThread's thread pool.
        self.plate_cache: Dict[int, str] = {}            # Final confirmed ALPR
        self.plate_history: Dict[int, list] = defaultdict(list)  # ALPR voting buffer
        self.plate_locked: set = set()                   # IDs with confirmed plate

        # Track which IDs have been dispatched for slave processing
        self.plate_dispatched: set = set()

        # Best bbox area tracking for plate timing optimization
        self.best_bbox_area: Dict[int, float] = {}
        self.best_bbox_crop: Dict[int, np.ndarray] = {}

        # Track OCR retry skip logic
        self.last_dispatched_area: Dict[int, float] = {}
        self.last_dispatched_frame: Dict[int, int] = {}

        # Coordinate EMA smoothing for flicker-free OSD
        self.bbox_ema: Dict[int, tuple] = {}

        # --- Accident Detection State (NEW) ---
        self.frame_buffer: List[np.ndarray] = []
        self.accident_dispatched = False
        self.needs_accident_dispatch = False
        self.recent_speed_drops = []

        # Interactive UI states
        self.divider_x_ratio = 0.5

        # Feature toggles
        self.speed_enabled = True
        self.wrong_way_enabled = True
        self.density_enabled = True

        # Traffic direction mode
        self.direction_mode = DIRECTION_MODE_STANDARD

        # Deduplication state for violation events
        self.active_violation_events: Dict[str, dict] = {}
        self.last_event_time: Dict[str, float] = {}
        
        # Ensure evidence directories exist
        os.makedirs("outputs/evidence/snapshots", exist_ok=True)
        os.makedirs("outputs/evidence/vehicle_crops", exist_ok=True)

        # FPS calculation
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0

    def load_model(self):
        """Load the YOLO model. Call once before processing."""
        if not self._model_loaded:
            self.model = YOLO(YOLO_MODEL)
            self._model_loaded = True

    def reset(self):
        """Clear all tracking state for a new video source."""
        self.plate_cache.clear()
        self.plate_history.clear()
        self.plate_locked.clear()
        self.plate_dispatched.clear()
        # Reset tracking buffers and history buffers
        self.track_history.clear()
        self.wrong_way_state.clear()
        self.best_bbox_area.clear()
        self.best_bbox_crop.clear()
        self.bbox_ema.clear()
        self.last_dispatched_area.clear()
        self.last_dispatched_frame.clear()
        
        # Reset event deduplication
        self.active_violation_events.clear()
        self.last_event_time.clear()
        self.recent_speed_drops.clear()

    # ==========================================
    # 📸 VIOLATION EVIDENCE HANDLER
    # ==========================================
    def _log_violation(self, track_id, event_type, timestamp, lane, plate_number, frame, x1, y1, x2, y2, events_list, current_speed):
        """Deduplicate events, capture evidence, and append to events list."""
        dedup_key = f"{track_id}_{event_type}"
        now = time.time()
        
        # Check active violations
        if dedup_key in self.active_violation_events:
            ev = self.active_violation_events[dedup_key]
            
            # Check rearm timeout
            if now - ev["last_seen_time"] > VIOLATION_REARM_SECONDS:
                # Rearm (create new event)
                del self.active_violation_events[dedup_key]
            else:
                # Update existing event
                ev["duration"] += now - ev["last_seen_time"]
                ev["last_seen_time"] = now
                ev["max_speed"] = max(ev["max_speed"], current_speed)
                ev["speed"] = current_speed
                if plate_number and plate_number != "—":
                    ev["plate_number"] = plate_number
                
                is_evidence_update = False
                # Check evidence update cooldown
                if now - ev["last_evidence_time"] > EVIDENCE_UPDATE_SECONDS:
                    ev["last_evidence_time"] = now
                    is_evidence_update = True
                    try:
                        cv2.imwrite(ev["snapshot_path"], frame)
                        fh, fw = frame.shape[:2]
                        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                        cx2, cy2 = min(fw, int(x2)), min(fh, int(y2))
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            cv2.imwrite(ev["vehicle_crop_path"], crop)
                    except Exception as e:
                        pass
                
                events_list.append(VehicleEvent(
                    event_id=ev["event_id"],
                    vehicle_id=int(track_id),
                    event_type=event_type,
                    timestamp=ev["start_timestamp"],
                    lane=lane,
                    plate_number=ev["plate_number"],
                    snapshot_path=ev["snapshot_path"],
                    vehicle_crop_path=ev["vehicle_crop_path"],
                    duration=ev["duration"],
                    is_update=True,
                    speed=ev["speed"],
                    max_speed=ev["max_speed"],
                ))
                return
        
        # New Event
        event_id = f"{dedup_key}_{int(now)}"
        safe_time = timestamp.replace(":", "-")
        event_slug = event_type.lower().replace(" ", "_")
        base_name = f"{safe_time}_track_{track_id}_{event_slug}"
        snap_path = f"outputs/evidence/snapshots/{base_name}.jpg"
        crop_path = f"outputs/evidence/vehicle_crops/{base_name}.jpg"
        
        try:
            cv2.imwrite(snap_path, frame)
            fh, fw = frame.shape[:2]
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(fw, int(x2)), min(fh, int(y2))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)
        except Exception as e:
            logger.warning(f"Failed to save evidence for track {track_id}: {e}")
            snap_path = ""
            crop_path = ""
            
        ev = {
            "event_id": event_id,
            "track_id": track_id,
            "event_type": event_type,
            "start_timestamp": timestamp,
            "start_time": now,
            "last_seen_time": now,
            "last_evidence_time": now,
            "duration": 0.0,
            "speed": current_speed,
            "max_speed": current_speed,
            "plate_number": plate_number,
            "lane": lane,
            "snapshot_path": snap_path,
            "vehicle_crop_path": crop_path,
        }
        self.active_violation_events[dedup_key] = ev
        
        events_list.append(VehicleEvent(
            event_id=ev["event_id"],
            vehicle_id=int(track_id),
            event_type=event_type,
            timestamp=ev["start_timestamp"],
            lane=lane,
            plate_number=ev["plate_number"],
            snapshot_path=ev["snapshot_path"],
            vehicle_crop_path=ev["vehicle_crop_path"],
            duration=ev["duration"],
            is_update=False,
            speed=ev["speed"],
            max_speed=ev["max_speed"],
        ))

    def _update_late_evidence(self, track_id, plate_number, events_list):
        """Update existing violation event with late-arriving ALPR data."""
        for dedup_key, ev in list(self.active_violation_events.items()):
            if dedup_key.startswith(f"{track_id}_"):
                updated = False
                if plate_number and plate_number not in ("—", "Pending", "Low Quality", "Unknown", "") and ev.get("plate_number") in ("—", "Pending", "Low Quality", "Unknown", ""):
                    ev["plate_number"] = plate_number
                    updated = True
                
                if updated:
                    events_list.append(VehicleEvent(
                        event_id=ev["event_id"],
                        vehicle_id=int(track_id),
                        event_type=ev["event_type"],
                        timestamp=ev["start_timestamp"],
                        lane=ev.get("lane", "UNKNOWN"),
                        plate_number=ev["plate_number"],
                        snapshot_path=ev.get("snapshot_path", ""),
                        vehicle_crop_path=ev.get("vehicle_crop_path", ""),
                        duration=ev.get("duration", 0),
                        is_update=True,
                        speed=ev.get("speed", 0),
                        max_speed=ev.get("max_speed", 0),
                    ))

        # Removed unintended state reset here

    # ==========================================
    # 🧠 DETECTION FUNCTIONS
    # ==========================================

    def get_lane(self, x_center: float, frame_width: int) -> str:
        """Classify vehicle lane based on adjustable horizontal divider."""
        return "LEFT" if x_center < (frame_width * self.divider_x_ratio) else "RIGHT"

    @staticmethod
    def detect_direction(centers_y: list) -> str:
        """Determine movement direction from Y-coordinate history."""
        if len(centers_y) < DIRECTION_FRAME_DIFF:
            return "STATIONARY"
        dy = centers_y[-1] - centers_y[-DIRECTION_FRAME_DIFF]
        if dy > DIRECTION_THRESHOLD:
            return "DOWN"
        elif dy < -DIRECTION_THRESHOLD:
            return "UP"
        return "STATIONARY"

    def check_wrong_way(self, lane: str, direction: str) -> bool:
        """
        Flag wrong-way driving based on the active direction mode.
        """
        rules = DIRECTION_RULES.get(
            self.direction_mode, DIRECTION_RULES[DIRECTION_MODE_STANDARD]
        )
        expected_direction = rules.get(lane)
        if direction != expected_direction and direction != "STATIONARY":
            return True
        return False

    def check_wrong_way_stateful(self, track_id: int, centers_x: list, centers_y: list, lane: str) -> bool:
        """
        Flag wrong-way driving using vector math, angle threshold, and accumulating reverse distance over consecutive frames.
        """
        if len(centers_x) < DIRECTION_FRAME_DIFF or len(centers_y) < DIRECTION_FRAME_DIFF:
            return False

        rules = DIRECTION_RULES.get(
            self.direction_mode, DIRECTION_RULES[DIRECTION_MODE_STANDARD]
        )
        expected_direction = rules.get(lane)
        if not expected_direction:
            return False

        # Expected vectors (screen coordinates: Y grows downwards)
        # UP means Y should decrease -> (0, -1)
        # DOWN means Y should increase -> (0, 1)
        ex, ey = (0, -1) if expected_direction == "UP" else (0, 1)

        # Movement vector
        old_x = centers_x[-DIRECTION_FRAME_DIFF]
        old_y = centers_y[-DIRECTION_FRAME_DIFF]
        curr_x = centers_x[-1]
        curr_y = centers_y[-1]

        dx = curr_x - old_x
        dy = curr_y - old_y

        dist = math.hypot(dx, dy)
        state = self.wrong_way_state[track_id]

        if dist < WRONG_WAY_MIN_STEP_DISTANCE_PX:
            # Vehicle is essentially stationary or jittering, gradually reset frames but keep distance for a bit
            state["consecutive_reverse_frames"] = max(0, state["consecutive_reverse_frames"] - 1)
            return state["confirmed"]

        # Calculate movement angle vs expected angle using dot product
        dot = dx * ex + dy * ey
        cos_theta = max(-1.0, min(1.0, dot / dist))
        angle = math.degrees(math.acos(cos_theta))

        if angle > WRONG_WAY_ANGLE_THRESHOLD_DEG:
            # Moving opposite to expected direction
            state["consecutive_reverse_frames"] += 1
            # Accumulate only the component of movement against the expected direction.
            reverse_dist = -dot
            if reverse_dist > 0:
                state["reverse_distance_px"] += reverse_dist
        else:
            # Moving normally, reset
            state["consecutive_reverse_frames"] = 0
            state["reverse_distance_px"] = 0.0

        if (state["consecutive_reverse_frames"] >= WRONG_WAY_MIN_CONSECUTIVE_FRAMES and 
            state["reverse_distance_px"] >= WRONG_WAY_MIN_REVERSE_DISTANCE_PX):
            state["confirmed"] = True

        return state["confirmed"]

    def clean_old_speed_drops(self, current_time_sec):
        """Keep memory low by discarding drops older than 30 seconds."""
        self.recent_speed_drops = [
            drop for drop in self.recent_speed_drops 
            if current_time_sec - drop["time_sec"] <= 30.0
        ]

    def check_sudden_stop(
        self,
        track_id: int,
        current_speed: float,
        hist: dict,
        x_center: float,
        y_center: float,
        frame_width: int,
        frame_height: int,
        video_time_sec: float,
        frame_count: int,
    ) -> bool:
        """Detect if vehicle dropped rapidly from highway speed to near-zero."""
        self.clean_old_speed_drops(video_time_sec)
        
        speeds = hist["speeds"]
        # Must have stable track history
        if len(speeds) < (SPEED_FRAME_DIFF * 2):
            return False
            
        # Ignore if near border (unreliable speed)
        margin_x = frame_width * 0.05
        margin_y = frame_height * 0.05
        if x_center < margin_x or x_center > (frame_width - margin_x):
            return False
        if y_center < margin_y or y_center > (frame_height - margin_y):
            return False
            
        # Calculate recent pre-stop average and current average
        # Take oldest 10 for pre-avg, newest 5 for post-avg
        pre_speeds = speeds[:10]
        post_speeds = speeds[-5:]
        
        pre_avg_speed = sum(pre_speeds) / len(pre_speeds) if len(pre_speeds) > 0 else 0
        post_avg_speed = sum(post_speeds) / len(post_speeds) if len(post_speeds) > 0 else current_speed
        
        speed_drop = pre_avg_speed - post_avg_speed
        
        # Check cooldown
        if video_time_sec - hist["last_sudden_stop_time"] < SUDDEN_STOP_COOLDOWN_SEC:
            return False

        if (pre_avg_speed >= SUDDEN_STOP_MIN_PREV_SPEED and 
            post_avg_speed <= SUDDEN_STOP_MAX_FINAL_SPEED and 
            speed_drop >= SUDDEN_STOP_MIN_DROP):
            
            hist["sudden_stop_frames"] += 1
            is_confirmed = (hist["sudden_stop_frames"] >= SUDDEN_STOP_CONFIRM_FRAMES)
            
            # Record or update buffer
            existing = None
            for drop in self.recent_speed_drops:
                if drop["track_id"] == track_id and (video_time_sec - drop["time_sec"]) < 5.0:
                    existing = drop
                    break
                    
            if existing:
                existing["confirmed_sudden_stop"] = existing["confirmed_sudden_stop"] or is_confirmed
                existing["post_avg_speed"] = post_avg_speed
                existing["drop"] = max(existing["drop"], speed_drop)
            else:
                self.recent_speed_drops.append({
                    "track_id": track_id,
                    "frame": frame_count,
                    "time_sec": video_time_sec,
                    "pre_avg_speed": pre_avg_speed,
                    "post_avg_speed": post_avg_speed,
                    "drop": speed_drop,
                    "bbox": (x_center, y_center),
                    "confirmed_sudden_stop": is_confirmed
                })
            
            if is_confirmed:
                hist["last_sudden_stop_time"] = video_time_sec
                hist["sudden_stop_frames"] = 0
                return True
        else:
            hist["sudden_stop_frames"] = 0
            
        return False

    @staticmethod
    def get_traffic_density(vehicle_count: int, avg_speed: float) -> str:
        """Classify overall traffic density."""
        if vehicle_count > DENSITY_JAM_VEHICLES and avg_speed < DENSITY_JAM_SPEED:
            return "TRAFFIC JAM"
        elif vehicle_count > DENSITY_HIGH_VEHICLES:
            return "HIGH"
        elif vehicle_count > DENSITY_MEDIUM_VEHICLES:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection
        if union == 0:
            return 0
        return intersection / union

    def detect_collision_overlap(self, boxes) -> bool:
        """Check for vehicle bounding box overlap/proximity indicating collision."""
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self.calculate_iou(boxes[i], boxes[j])
                if iou > ACCIDENT_IOU_TRIGGER:
                    # Check center distance
                    c1x, c1y = (boxes[i][0]+boxes[i][2])/2, (boxes[i][1]+boxes[i][3])/2
                    c2x, c2y = (boxes[j][0]+boxes[j][2])/2, (boxes[j][1]+boxes[j][3])/2
                    dist = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)
                    diag = np.sqrt((boxes[i][2]-boxes[i][0])**2 + (boxes[i][3]-boxes[i][1])**2)
                    if dist < diag * ACCIDENT_CENTER_DIST_FACTOR:
                        return True
        return False

    # ==========================================
    # 🏎️ BEV SPEED ESTIMATION (NEW — replaces K/bbox_height)
    # ==========================================

    def compute_speed_bev(
        self,
        hist: dict,
        x_center: float,
        y_center: float,
        fps: float,
    ) -> float:
        """
        Compute calibrated speed using Bird's-Eye View transformation.

        CHANGED from v1.0: Instead of K/bbox_height, we project the
        vehicle's center point into the BEV metric space and compute
        the Euclidean distance traveled over time.

        Args:
            hist: Per-vehicle tracking history dict.
            x_center: Current x-center in pixels.
            y_center: Current y-center in pixels.
            fps: Video FPS for time calculation.

        Returns:
            Speed in km/h, smoothed via EMA.
        """
        if not self.calibrator.is_calibrated:
            return 0.0

        if len(hist["centers_x"]) < SPEED_FRAME_DIFF:
            return 0.0

        # Previous position (SPEED_FRAME_DIFF frames ago)
        prev_x = hist["centers_x"][-SPEED_FRAME_DIFF]
        prev_y = hist["centers_y"][-SPEED_FRAME_DIFF]

        # Time delta
        dt = SPEED_FRAME_DIFF / fps

        # Compute calibrated speed via homography projection
        raw_speed = self.calibrator.compute_speed(
            prev_pos_px=(prev_x, prev_y),
            curr_pos_px=(x_center, y_center),
            dt_seconds=dt,
        )

        # Smooth with EMA to reduce jitter
        smoother: SpeedSmoother = hist["speed_smoother"]
        smoothed = smoother.update(raw_speed)

        return min(smoothed, SPEED_MAX_PLAUSIBLE)

    # ==========================================
    # 🖐️ FRAME ANNOTATION
    # ==========================================

    def _draw_lane_overlay(self, frame: np.ndarray, width: int, height: int):
        """Draw lane divider and direction labels based on active mode and dynamic slider."""
        rules = DIRECTION_RULES.get(
            self.direction_mode, DIRECTION_RULES[DIRECTION_MODE_STANDARD]
        )
        left_dir = rules["LEFT"]
        right_dir = rules["RIGHT"]

        divider_x = int(width * self.divider_x_ratio)
        left_center_x = divider_x // 2
        right_center_x = divider_x + (width - divider_x) // 2

        # Draw semi-transparent line
        overlay = frame.copy()
        
        if self.direction_mode in [DIRECTION_MODE_ONEWAY_UP, DIRECTION_MODE_ONEWAY_DOWN]:
            expected = "UP" if self.direction_mode == DIRECTION_MODE_ONEWAY_UP else "DOWN"
            center_x = width // 2
            cv2.putText(frame, f"ALL {expected}", (center_x - 40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, CV_COLOR_LANE_LINE, 2, cv2.LINE_AA)
            if expected == "DOWN":
                cv2.arrowedLine(frame, (center_x, 65), (center_x, 110),
                                CV_COLOR_LANE_LINE, 2, tipLength=0.4)
            else:
                cv2.arrowedLine(frame, (center_x, 110), (center_x, 65),
                                CV_COLOR_LANE_LINE, 2, tipLength=0.4)
            return
            
        cv2.line(overlay, (divider_x, 0), (divider_x, height), CV_COLOR_LANE_LINE, 2)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw Labels
        cv2.putText(frame, f"{left_dir}", (left_center_x - 30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, CV_COLOR_LANE_LINE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{right_dir}", (right_center_x - 30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, CV_COLOR_LANE_LINE, 2, cv2.LINE_AA)

        if left_dir == "DOWN":
            cv2.arrowedLine(frame, (left_center_x, 65), (left_center_x, 110),
                            CV_COLOR_LANE_LINE, 2, tipLength=0.4)
        else:
            cv2.arrowedLine(frame, (left_center_x, 110), (left_center_x, 65),
                            CV_COLOR_LANE_LINE, 2, tipLength=0.4)

        if right_dir == "DOWN":
            cv2.arrowedLine(frame, (right_center_x, 65), (right_center_x, 110),
                            CV_COLOR_LANE_LINE, 2, tipLength=0.4)
        else:
            cv2.arrowedLine(frame, (right_center_x, 110), (right_center_x, 65),
                            CV_COLOR_LANE_LINE, 2, tipLength=0.4)

    def _draw_vehicle_box(
        self, frame, x1, y1, x2, y2, track_id,
        lane, speed, behavior, color,
        plate: str = "",
    ):
        """
        Draw bounding box with professional translucent overlays,
        flicker-free EMA coordinate smoothing, and anti-aliased grouped text.
        """
        # --- EMA Smoothing for BBox Coordinates ---
        w_curr, h_curr = (x2 - x1), (y2 - y1)
        cx_curr, cy_curr = x1 + w_curr / 2.0, y1 + h_curr / 2.0

        if track_id not in self.bbox_ema:
            self.bbox_ema[track_id] = (cx_curr, cy_curr, w_curr, h_curr)
        else:
            # Alpha for smoothing (lower = smoother but lags, higher = responsive)
            alpha = 0.4
            cx_ema, cy_ema, w_ema, h_ema = self.bbox_ema[track_id]
            cx_ema = alpha * cx_curr + (1 - alpha) * cx_ema
            cy_ema = alpha * cy_curr + (1 - alpha) * cy_ema
            w_ema  = alpha * w_curr + (1 - alpha) * w_ema
            h_ema  = alpha * h_curr + (1 - alpha) * h_ema
            self.bbox_ema[track_id] = (cx_ema, cy_ema, w_ema, h_ema)

        cx, cy, w, h = self.bbox_ema[track_id]
        sx1, sy1 = int(cx - w / 2), int(cy - h / 2)
        sx2, sy2 = int(cx + w / 2), int(cy + h / 2)

        # Clamp to frame
        fh, fw = frame.shape[:2]
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(fw, sx2), min(fh, sy2)

        # Draw main box
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 1, cv2.LINE_AA)

        # --- Group Information ---
        header_text = f"ID:{track_id} | {speed:.0f} km/h"
        sub_text = ""
        if plate and plate not in ("—", "Unreadable", "Unknown", "N/A"):
            sub_text = f"Plate: {plate}"
        
        # Calculate background panel size
        header_font_scale, sub_font_scale = 0.5, 0.4
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        (h_w, h_h), _ = cv2.getTextSize(header_text, font, header_font_scale, thickness)
        panel_width = h_w + 10
        panel_height = h_h + 12

        if sub_text:
            (s_w, s_h), _ = cv2.getTextSize(sub_text, font, sub_font_scale, thickness)
            panel_width = max(panel_width, s_w + 10)
            panel_height += s_h + 8

        panel_x1 = sx1
        panel_y1 = max(0, sy1 - panel_height)
        panel_x2 = min(fw, panel_x1 + panel_width)
        panel_y2 = panel_y1 + panel_height

        # Render translucent dark panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
        # Add colored top border to panel
        cv2.line(overlay, (panel_x1, panel_y1), (panel_x2, panel_y1), color, 2, cv2.LINE_AA)
        
        # Apply translucency
        opacity = 0.85
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        # Draw Text
        text_y = panel_y1 + h_h + 6
        cv2.putText(frame, header_text, (panel_x1 + 5, text_y), font, header_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        if sub_text:
            text_y += s_h + 6
            cv2.putText(frame, sub_text, (panel_x1 + 5, text_y), font, sub_font_scale, (200, 255, 255), thickness, cv2.LINE_AA)

    # ==========================================
    # 🚗 SLAVE MODEL DISPATCH HELPERS
    # ==========================================

    def should_dispatch_plate(self, track_id: int, bbox_area: float, frame_count: int) -> bool:
        """
        Check if we should dispatch plate reading for this track.

        Logic: Repeatedly poll the ALPR model across multiple frames while the
        bounding box is large enough. Stop if the plate is locked or already polling.
        """
        if track_id in self.plate_locked:
            return False  # Confirmed and locked
            
        if track_id in self.plate_dispatched:
            return False  # Currently running a thread

        # Allow max 15 attempts to prevent endless polling on unreadable objects
        if len(self.plate_history[track_id]) >= 15:
            return False

        # Store the current best crop
        prev_best = self.best_bbox_area.get(track_id, 0)
        if bbox_area > prev_best:
            self.best_bbox_area[track_id] = bbox_area

        # Skip retry if the crop isn't at least 5% larger OR 10 frames haven't passed
        last_area = self.last_dispatched_area.get(track_id, 0)
        last_frame = self.last_dispatched_frame.get(track_id, -100)
        
        if bbox_area <= last_area * 1.05 and (frame_count - last_frame) < 10:
            return False

        self.last_dispatched_area[track_id] = bbox_area
        self.last_dispatched_frame[track_id] = frame_count

        return True

    def get_vehicle_crop(self, frame: np.ndarray, x1, y1, x2, y2) -> Optional[np.ndarray]:
        """Safely crop a vehicle region from the frame."""
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    # ==========================================
    # 🚀 MAIN ANALYSIS PIPELINE (v2.0)
    # ==========================================

    def _initialize_calibrator(self, frame_width: int, frame_height: int):
        """Initialize the BEV calibrator based on the selected configuration."""
        if self.roi_config is None:
            self.calibrator = BirdEyeViewCalibrator()
            self.calibrator.auto_calibrate(frame_width, frame_height)
            logger.info(f"BEV calibrator auto-initialized for {frame_width}×{frame_height} frame (default).")

        elif self.roi_config["mode"] == "auto":
            self.calibrator = BirdEyeViewCalibrator(
                real_width_m=self.roi_config["real_width_m"],
                real_height_m=self.roi_config["real_height_m"]
            )
            self.calibrator.auto_calibrate(frame_width, frame_height)
            logger.info(f"BEV calibrator auto-initialized with custom dimensions: {self.roi_config['real_width_m']}m x {self.roi_config['real_height_m']}m.")

        elif self.roi_config["mode"] == "manual":
            self.calibrator = BirdEyeViewCalibrator(
                src_points=np.float32(self.roi_config["src_points"]),
                real_width_m=self.roi_config["real_width_m"],
                real_height_m=self.roi_config["real_height_m"]
            )
            logger.info("BEV calibrator initialized with custom manual ROI points.")

        self._calibrator_initialized = True

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_count: int,
        fps: float,
    ) -> FrameResult:
        """
        Process a single video frame through the full pipeline.

        v2.0 Changes:
            - Initializes BEV calibrator on first frame.
            - Uses BEV-projected speed instead of K/bbox_height.
            - Collects slave model results from caches.
            - Returns enriched VehicleInfo list.

        Args:
            frame: Raw BGR frame from cv2.VideoCapture
            frame_count: Current frame number
            fps: Source video FPS

        Returns:
            FrameResult with annotated frame, metrics, and vehicle data.
        """
        if not self._model_loaded or self.model is None:
            self.load_model()

        # --- Initialize BEV calibrator on first frame (CHANGED) ---
        height, width = frame.shape[:2]
        if not self._calibrator_initialized:
            self._initialize_calibrator(width, height)

        # Maintain rolling frame buffer for VideoMAE (3 seconds)
        max_buffer_size = int(fps * 3) if fps > 0 else 90
        small_frame = cv2.resize(frame, (640, 640))
        self.frame_buffer.append(small_frame)
        if len(self.frame_buffer) > max_buffer_size:
            self.frame_buffer.pop(0)

        # Calculate processing FPS
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_start_time = time.time()

        video_time_sec = frame_count / fps if fps > 0 else frame_count / 30.0
        current_time_str = time.strftime("%H:%M:%S")

        # Draw lane overlay
        self._draw_lane_overlay(frame, width, height)

        # Draw BEV ROI (subtle yellow trapezoid)
        if self.show_roi_overlay:
            self.calibrator.draw_roi_overlay(frame, color=(0, 200, 255), thickness=1)

        # Run YOLO tracking (Master model)
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            verbose=False,
            conf=0.20,
            iou=0.7,
        )

        current_vehicle_count = 0
        current_frame_speeds = []
        events = []
        vehicles = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            current_vehicle_count = len(ids)

            for box, track_id, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                bbox_height = y2 - y1
                bbox_area = (x2 - x1) * (y2 - y1)

                # Update history
                hist = self.track_history[track_id]
                hist["centers_x"].append(x_center)
                hist["centers_y"].append(y_center)
                hist["heights"].append(bbox_height)
                hist["last_seen_frame"] = frame_count

                # Bound history
                if len(hist["centers_y"]) > HISTORY_LENGTH:
                    hist["centers_x"].pop(0)
                    hist["centers_y"].pop(0)
                    hist["heights"].pop(0)

                # --- Metrics ---
                lane = self.get_lane(x_center, width)
                direction = self.detect_direction(hist["centers_y"])

                # --- CHANGED: BEV-calibrated speed (replaces K/bbox_height) ---
                current_speed = 0.0
                if self.speed_enabled:
                    current_speed = self.compute_speed_bev(
                        hist, x_center, y_center, fps
                    )

                hist["speeds"].append(current_speed)
                if len(hist["speeds"]) > HISTORY_LENGTH:
                    hist["speeds"].pop(0)

                current_frame_speeds.append(current_speed)

                # --- Store best crop for plate reading ---
                if bbox_area > self.best_bbox_area.get(track_id, 0):
                    crop = self.get_vehicle_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        self.best_bbox_crop[track_id] = crop

                # --- Behavior Analysis (MAINTAINED) ---
                is_wrong_way = False
                if self.wrong_way_enabled:
                    is_wrong_way = self.check_wrong_way_stateful(track_id, hist["centers_x"], hist["centers_y"], lane)

                is_sudden_stop = self.check_sudden_stop(
                    track_id, current_speed, hist, x_center, y_center, width, height, video_time_sec, frame_count
                )

                behavior = "NORMAL"
                color = CV_COLOR_NORMAL

                plate_number = self.plate_cache.get(track_id, "—")

                if is_wrong_way:
                    behavior = "WRONG WAY"
                    color = CV_COLOR_WRONG_WAY
                    self._log_violation(track_id, "WRONG WAY", current_time_str, lane, plate_number, frame, x1, y1, x2, y2, events, current_speed)
                elif is_sudden_stop:
                    behavior = "SUDDEN STOP"
                    color = CV_COLOR_SUDDEN_STOP
                    self._log_violation(track_id, "SUDDEN STOP", current_time_str, lane, plate_number, frame, x1, y1, x2, y2, events, current_speed)
                else:
                    self._update_late_evidence(track_id, plate_number, events)

                # Draw vehicle box with enriched labels
                self._draw_vehicle_box(
                    frame, x1, y1, x2, y2, track_id,
                    lane, current_speed, behavior, color,
                    plate=plate_number if plate_number != "—" else "",
                )

                # Build VehicleInfo (NEW in v2.0)
                vehicles.append(VehicleInfo(
                    track_id=int(track_id),
                    speed_kmh=current_speed,
                    direction=direction,
                    lane=lane,
                    behavior=behavior,
                    plate_number=plate_number,
                    coco_class=int(cls_id),
                ))

            # --- Check Accident Triggers ---
            if not self.accident_dispatched and len(self.frame_buffer) > 30:
                # simplified check: if any vehicle speed is heavily dropping this frame
                # Actually we already checked `is_sudden_stop` in the loop
                pass
                # Check if any event in `events` is a sudden stop
                has_major_sudden_stop = any(e.event_type == "SUDDEN STOP" for e in events)
                
                # Trigger 2: Bounding Box Proximity
                has_overlap = self.detect_collision_overlap(boxes)
                
                if has_major_sudden_stop or has_overlap:
                    logger.info("Accident anomaly detected! Flagging for event collection...")
                    self.accident_dispatched = True
                    self.needs_accident_dispatch = True

        # --- Global metrics ---
        avg_speed = (
            int(sum(current_frame_speeds) / len(current_frame_speeds))
            if current_frame_speeds else 0
        )

        density_status = "LOW"
        if self.density_enabled:
            density_status = self.get_traffic_density(
                current_vehicle_count, avg_speed
            )

        # --- Track Memory Cleanup ---
        # Remove tracks not seen in the last 60 frames to prevent memory leaks
        stale_threshold = frame_count - 60
        stale_ids = [tid for tid, hist in self.track_history.items() if hist["last_seen_frame"] < stale_threshold]
        for tid in stale_ids:
            del self.track_history[tid]
            if tid in self.wrong_way_state:
                del self.wrong_way_state[tid]

        return FrameResult(
            annotated_frame=frame,
            vehicle_count=current_vehicle_count,
            average_speed=int(avg_speed),
            density_status=density_status,
            events=events,
            vehicles=vehicles,
            fps=round(self._current_fps, 1),
            frame_number=frame_count,
        )
