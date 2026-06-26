"""
Constants and configuration for RoadVision AI.

Upgraded with Bird's-Eye View calibration parameters, multi-model
pipeline settings, and expanded UI constants.
"""

# ==========================================
# ⚙️ AI MODEL CONFIGURATION
# ==========================================
YOLO_MODEL = "yolov8n.pt"  # Nano model for real-time speed
YOLO_CONFIDENCE = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# Plate OCR Debugging
DEBUG_PLATE_OCR = True  # Set to True to save crops and TrOCR inputs to outputs/debug_plate_ocr/


# COCO class name mapping for display
COCO_CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# ==========================================
# 📏 BIRD'S-EYE VIEW CALIBRATION
# ==========================================
# Default source points are auto-generated from frame dimensions
# if BEV_SRC_POINTS_DEFAULT is None. Override with 4 pixel coords
# [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in TL, TR, BR, BL order.
BEV_SRC_POINTS_DEFAULT = None
BEV_REAL_WIDTH_METERS = 7.0    # Standard 2-lane road width
BEV_REAL_HEIGHT_METERS = 40.0  # Depth of the ROI in meters

# Vehicle class average real-world lengths (meters) for fallback
VEHICLE_REAL_LENGTHS = {
    2: 4.5,   # car
    3: 2.0,   # motorcycle
    5: 12.0,  # bus
    7: 8.0,   # truck
}

# ==========================================
# 📏 TRACKING & SPEED CONFIGURATION
# ==========================================
HISTORY_LENGTH = 30  # Frames to keep in per-vehicle history (increased for BEV)
SPEED_FRAME_DIFF = 5   # Compare positions across N frames
DIRECTION_FRAME_DIFF = 5
DIRECTION_THRESHOLD = 5  # Pixel movement threshold to confirm direction

# Wrong-way confirmation thresholds
WRONG_WAY_ANGLE_THRESHOLD_DEG = 100
WRONG_WAY_MIN_REVERSE_DISTANCE_PX = 50.0
WRONG_WAY_MIN_CONSECUTIVE_FRAMES = 8
WRONG_WAY_MIN_STEP_DISTANCE_PX = 5.0

SPEED_SMOOTHING_WINDOW = 5    # EMA window for speed smoothing
SPEED_MAX_PLAUSIBLE = 200.0   # km/h — clamp unrealistic values

# ==========================================
# 🚦 TRAFFIC DENSITY THRESHOLDS
# ==========================================
DENSITY_JAM_VEHICLES = 35
DENSITY_JAM_SPEED = 10
DENSITY_HIGH_VEHICLES = 25
DENSITY_MEDIUM_VEHICLES = 15

# ==========================================
# 🛑 BEHAVIOR DETECTION THRESHOLDS
# ==========================================
SUDDEN_STOP_HIGH_SPEED = 20.0  # (Legacy)
SUDDEN_STOP_LOW_SPEED = 3.0    # (Legacy)

SUDDEN_STOP_MIN_PREV_SPEED = 60.0
SUDDEN_STOP_MAX_FINAL_SPEED = 5.0
SUDDEN_STOP_MIN_DROP = 50.0
SUDDEN_STOP_CONFIRM_FRAMES = 5
SUDDEN_STOP_COOLDOWN_SEC = 3.0
ACCIDENT_SPEED_DROP_WINDOW = 5.0

SLOW_SPEED_THRESHOLD = 10.0    # km/h
STOPPED_SPEED_THRESHOLD = 3.0  # km/h

# ==========================================
# 🔄 TRAFFIC DIRECTION MODES
# ==========================================
DIRECTION_MODE_STANDARD = "standard"   # LEFT→DOWN, RIGHT→UP
DIRECTION_MODE_REVERSED = "reversed"   # LEFT→UP, RIGHT→DOWN
DIRECTION_MODE_ONEWAY_UP = "oneway_up" # ALL→UP
DIRECTION_MODE_ONEWAY_DOWN = "oneway_down" # ALL→DOWN

DIRECTION_MODE_LABELS = {
    DIRECTION_MODE_STANDARD: "Two-way (Left ↓, Right ↑)",
    DIRECTION_MODE_REVERSED: "Two-way (Left ↑, Right ↓)",
    DIRECTION_MODE_ONEWAY_UP: "One-way (All ↑)",
    DIRECTION_MODE_ONEWAY_DOWN: "One-way (All ↓)",
}

DIRECTION_RULES = {
    DIRECTION_MODE_STANDARD: {"LEFT": "DOWN", "RIGHT": "UP"},
    DIRECTION_MODE_REVERSED: {"LEFT": "UP",   "RIGHT": "DOWN"},
    DIRECTION_MODE_ONEWAY_UP: {"LEFT": "UP", "RIGHT": "UP"},
    DIRECTION_MODE_ONEWAY_DOWN: {"LEFT": "DOWN", "RIGHT": "DOWN"},
}

# ==========================================
# 🤖 MULTI-MODEL PIPELINE SETTINGS
# ==========================================
# Plate detection
PLATE_YOLO_MODEL = "models/_weights/plate_detector/best_Koushim_Model.pt"
TROCR_MODEL_PATH = "models/_weights/plate_recognizer/model/"
PLATE_CONFIDENCE = 0.4

# EasyOCR
OCR_LANGUAGES = ["en"]

# Slave model dispatch
PLATE_READ_PATIENCE_FRAMES = 15  # Frames after max bbox to trigger plate read
SLAVE_THREAD_POOL_WORKERS = 2    # Max concurrent slave model workers
MIN_BBOX_AREA_FOR_SLAVE = 5000    # Min bbox area (px²) to dispatch OCR
VEHICLE_ACTIVE_TIMEOUT_FRAMES = 90  # Keep vehicle in table ~3s after leaving

# Summary export
SUMMARY_CSV_FILENAME = "session_summary"

# ==========================================
# 🚨 ACCIDENT DETECTION MODULE SETTINGS
# ==========================================
VIDEOMAE_CHECKPOINT = "models/_weights/checkpoint-best.pth"
VIDEOMAE_NUM_FRAMES = 16
FULL_SCAN_THRESHOLD_SEC = 60  # Video duration threshold for triggering Full Scan Mode
VIDEOMAE_INPUT_SIZE = 224
VIDEOMAE_ACCIDENT_CLASS_INDEX = 1

ACCIDENT_IOU_TRIGGER = 0.03
ACCIDENT_CENTER_DIST_FACTOR = 0.85
ACCIDENT_SUDDEN_STOP_RATIO = 0.55
ACCIDENT_MIN_TRACK_POINTS = 5

# Storage & Cleanup
ENABLE_INCIDENT_CLEANUP = False
MAX_INCIDENT_FOLDERS = 50

# ==========================================
# 🧠 GEMINI REPORTING & EVIDENCE SETTINGS
# ==========================================
GEMINI_REPORT_MODEL = "gemini-2.5-pro"
GEMINI_REPORT_TEMPERATURE = 0.0
GEMINI_REPORT_MAX_OUTPUT_TOKENS = 8192
GEMINI_REPORT_TIMEOUT_SECONDS = 90 # First attempt timeout
GEMINI_REPORT_TOTAL_DEADLINE = 120 # Total deadline for attempts
GEMINI_REPORT_MAX_ATTEMPTS = 2

ACCIDENT_ANALYSIS_MAX_FRAMES = 15
ACCIDENT_MAX_EVIDENCE_IMAGES = 6
ACCIDENT_GEMINI_MAX_IMAGE_DIMENSION = 384
ACCIDENT_GEMINI_JPEG_QUALITY = 82
ACCIDENT_BLUR_THRESHOLD = 50.0  # Variance of Laplacian (to be tested)
ACCIDENT_DUPLICATE_THRESHOLD = 0.95 # Structural Similarity / Histogram (to be tested)
DEBUG_ACCIDENT_FRAME_SELECTION = True

# Evidence Buffer settings
EVIDENCE_JPEG_FPS = 10
EVIDENCE_BUFFER_SECONDS = 7.0

# ==========================================
# 🎨 COLORS — OpenCV (BGR)
# ==========================================
CV_COLOR_NORMAL = (0, 255, 0)        # Green
CV_COLOR_WRONG_WAY = (0, 0, 255)     # Red
CV_COLOR_SUDDEN_STOP = (0, 165, 255) # Orange
CV_COLOR_TEXT = (255, 255, 255)      # White
CV_COLOR_LANE_LINE = (255, 255, 255) # White
CV_COLOR_OVERLAY_BG = (0, 0, 0)     # Black
CV_COLOR_PLATE = (255, 200, 0)       # Cyan-ish for plate boxes

# ==========================================
# 🎨 COLORS — Qt (Hex)
# ==========================================
# Main theme
THEME_BG_PRIMARY = "#0d1117"
THEME_BG_SECONDARY = "#161b22"
THEME_BG_TERTIARY = "#1c2333"
THEME_BG_CARD = "#21262d"
THEME_BORDER = "#30363d"
THEME_BORDER_LIGHT = "#3d444d"

# Text
THEME_TEXT_PRIMARY = "#e6edf3"
THEME_TEXT_SECONDARY = "#8b949e"
THEME_TEXT_MUTED = "#6e7681"

# Accents
THEME_ACCENT = "#58a6ff"
THEME_ACCENT_HOVER = "#79c0ff"
THEME_SUCCESS = "#3fb950"
THEME_SUCCESS_HOVER = "#56d364"
THEME_WARNING = "#d29922"
THEME_DANGER = "#f85149"
THEME_DANGER_HOVER = "#ff7b72"
THEME_INFO = "#39d2c0"

# Status colors
COLOR_DENSITY_LOW = "#3fb950"
COLOR_DENSITY_MEDIUM = "#d29922"
COLOR_DENSITY_HIGH = "#f0883e"
COLOR_DENSITY_JAM = "#f85149"

# ==========================================
# 📊 CHART CONFIGURATION
# ==========================================
CHART_MAX_POINTS = 300
CHART_WINDOW_SECONDS = 60
CHART_BG_COLOR = "#0d1117"
CHART_LINE_VEHICLE = "#58a6ff"
CHART_LINE_SPEED = "#3fb950"
CHART_LINE_DENSITY = "#d29922"
CHART_GRID_COLOR = "#21262d"
CHART_AXIS_COLOR = "#8b949e"

# ==========================================
# 📋 EVENT LOG CONFIGURATION
# ==========================================
MAX_EVENT_LOG_ROWS = 200
VIOLATION_REARM_SECONDS = 5.0
EVIDENCE_UPDATE_SECONDS = 2.0

# ==========================================
# 🖥️ UI CONFIGURATION
# ==========================================
APP_TITLE = "RoadVision AI"
WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 720
VIDEO_PANEL_RATIO = 0.65
CONTROL_PANEL_RATIO = 0.35

# Font
FONT_FAMILY = "Segoe UI"
FONT_SIZE_NORMAL = 11
FONT_SIZE_SMALL = 9
FONT_SIZE_LARGE = 14
FONT_SIZE_TITLE = 18
FONT_SIZE_METRIC = 28
