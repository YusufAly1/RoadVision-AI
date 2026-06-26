"""
Main Window — Root window that assembles all panels into the RoadVision AI dashboard layout.

v2.0 Changes:
    - Added VehicleTablePanel for rich per-vehicle data display.
    - Wired FrameResult.vehicles to the vehicle table.
    - Updated events panel to pass vehicle_desc and plate_number.
    - Added model loading progress indicator.

Layout:
┌──────────────────────────────────────────────────┐
│  TOP BAR: Title | Status | FPS                   │
├──────────────────────┬───────────────────────────┤
│                      │  Control Panel             │
│                      ├───────────────────────────┤
│   Video Panel        │  Stats Panel (3 cards)     │
│   (65% width)        ├───────────────────────────┤
│                      │  Vehicle Dashboard (NEW)   │
│                      ├───────────────────────────┤
│                      │  Charts Panel (3 graphs)   │
│                      ├───────────────────────────┤
│                      │  Events Panel (table)      │
└──────────────────────┴───────────────────────────┘
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QFrame, QFileDialog, QScrollArea,
    QMessageBox, QApplication, QDialog, QTextBrowser,
    QDialogButtonBox, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon

from ui.styles import DARK_THEME_QSS
from ui.video_panel import VideoPanel
from ui.control_panel import ControlPanel
from ui.stats_panel import StatsPanel
from ui.charts_panel import ChartsPanel
from ui.events_panel import EventsPanel
from ui.vehicle_table_panel import VehicleTablePanel  # NEW in v2.0
from ui.incident_reports_panel import IncidentReportsPanel

from core.video_thread import VideoThread, SourceType
from core.traffic_analyzer import FrameResult

from utils.constants import (
    APP_TITLE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    THEME_SUCCESS, THEME_DANGER, THEME_ACCENT,
    THEME_TEXT_PRIMARY, THEME_TEXT_MUTED,
    THEME_BG_SECONDARY, THEME_BORDER,
    FONT_FAMILY,
    DIRECTION_MODE_LABELS, DIRECTION_MODE_STANDARD,
    DIRECTION_MODE_ONEWAY_UP, DIRECTION_MODE_ONEWAY_DOWN,
)


class MainWindow(QMainWindow):
    """
    Root application window for RoadVision AI.
    Assembles all UI panels, manages the VideoThread lifecycle,
    and wires all signals together.

    v2.0: Added VehicleTablePanel and wired multi-model pipeline data.
    """

    def __init__(self):
        super().__init__()
        import os
        print(f"DEBUG: Loaded main_window.py from: {os.path.abspath(__file__)}")
        self._video_thread = None
        self._roi_config = None
        self._setup_window()
        self._setup_ui()
        self._apply_theme()

    def _setup_window(self):
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

    def _apply_theme(self):
        self.setStyleSheet(DARK_THEME_QSS)

    # ==========================================
    # 🏗️ UI CONSTRUCTION
    # ==========================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- TOP BAR ---
        root_layout.addWidget(self._build_top_bar())

        # --- MAIN CONTENT (Tabs) ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabBar::tab { background: #21262d; color: #8b949e; padding: 12px 24px; border: 1px solid #30363d; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }"
            "QTabBar::tab:selected { background: #0d1117; color: #e6edf3; border-color: #58a6ff; }"
            "QTabWidget::pane { border: 1px solid #30363d; background: transparent; }"
        )

        # --- TAB 1: Dashboard ---
        dashboard_tab = QWidget()
        dashboard_layout = QHBoxLayout(dashboard_tab)
        dashboard_layout.setContentsMargins(8, 8, 8, 8)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left: Video Panel
        self.video_panel = VideoPanel()
        splitter.addWidget(self.video_panel)

        # Right: Scrollable control area (Controls, Stats, VehicleTable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.control_panel = ControlPanel()
        self.stats_panel = StatsPanel()
        self.vehicle_table_panel = VehicleTablePanel()

        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(self.stats_panel)
        right_layout.addWidget(self.vehicle_table_panel)
        right_layout.addStretch()

        right_scroll.setWidget(right_container)
        splitter.addWidget(right_scroll)

        # Set splitter proportions (50:50)
        splitter.setSizes([500, 500])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        dashboard_layout.addWidget(splitter)
        self.tab_widget.addTab(dashboard_tab, "Dashboard")

        # --- TAB 2: Analytics ---
        self.charts_panel = ChartsPanel()
        self.tab_widget.addTab(self.charts_panel, "Analytics")

        # --- TAB 3: Event Alerts ---
        self.events_panel = EventsPanel()
        self.tab_widget.addTab(self.events_panel, "Event Alerts")

        # --- TAB 4: Incident Reports ---
        self.incident_reports_panel = IncidentReportsPanel()
        self.tab_widget.addTab(self.incident_reports_panel, "Incident Reports")

        content_layout.addWidget(self.tab_widget)
        root_layout.addWidget(content_widget, 1)

        # --- Connect control panel signals ---
        self.control_panel.select_camera_clicked.connect(self._on_select_camera)
        self.control_panel.select_video_clicked.connect(self._on_select_video)
        self.control_panel.select_ip_clicked.connect(self._on_select_ip)
        self.control_panel.start_clicked.connect(self._on_start)
        self.control_panel.stop_clicked.connect(self._on_stop)
        self.control_panel.pause_clicked.connect(self._on_pause)

        self.control_panel.speed_toggled.connect(self._on_toggle_speed)
        self.control_panel.wrong_way_toggled.connect(self._on_toggle_wrong_way)
        self.control_panel.density_toggled.connect(self._on_toggle_density)
        self.control_panel.direction_mode_changed.connect(self._on_direction_mode_changed)
        self.control_panel.divider_slider_moved.connect(self._on_divider_slider_moved)
        self.control_panel.edit_calibration_clicked.connect(self._open_calibration_dialog)

    def _build_top_bar(self) -> QFrame:
        """Build the top bar with title, status, and FPS."""
        bar = QFrame()
        bar.setObjectName("top_bar")
        bar.setFixedHeight(56)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(20, 0, 20, 0)
        bar_layout.setSpacing(16)

        # Title Area
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title = QLabel(f"🛡️  {APP_TITLE}")
        title.setObjectName("title_label")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Real-Time Traffic Intelligence System")
        subtitle.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: 500;")
        title_layout.addWidget(subtitle)
        
        bar_layout.addLayout(title_layout)

        bar_layout.addStretch()

        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fps_label")
        bar_layout.addWidget(self.fps_label)

        # Direction mode indicator in top bar
        self.mode_badge = QLabel(f"Mode: {DIRECTION_MODE_LABELS[DIRECTION_MODE_STANDARD]}")
        self.mode_badge.setStyleSheet(
            f"background-color: rgba(88, 166, 255, 0.12); "
            f"color: {THEME_ACCENT}; border-radius: 10px; "
            f"padding: 4px 10px; font-size: 10px; font-weight: 600;"
            f"background: transparent;"
        )
        bar_layout.addWidget(self.mode_badge)

        # Status indicator
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(20)
        self.status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_layout.addWidget(self.status_dot)

        self.status_label = QLabel("Stopped")
        self.status_label.setObjectName("status_label")
        bar_layout.addWidget(self.status_label)

        self._set_status_stopped()

        return bar

    # ==========================================
    # 🎛️ STATUS HELPERS
    # ==========================================

    def _on_scan_complete(self, result: dict):
        """Handle false alarms quietly."""
        # The AI rejected the event, just unlock UI
        self.control_panel.set_running_state()
        logger.info(f"Scan complete: No accident detected. Max conf: {result.get('videomae_confidence', 0):.2f}")

    # ==========================================
    # 💾 DATA EXPORT & SUMMARY
    # ==========================================

    def _set_status_running(self):
        self.status_dot.setStyleSheet(
            f"color: {THEME_SUCCESS}; font-size: 18px; background: transparent;"
        )
        self.status_label.setText("Running")
        self.status_label.setStyleSheet(
            f"background-color: rgba(63, 185, 80, 0.15); "
            f"color: {THEME_SUCCESS}; border-radius: 10px; "
            f"padding: 4px 12px; font-size: 12px; font-weight: 600;"
        )

    def _set_status_stopped(self):
        self.status_dot.setStyleSheet(
            f"color: {THEME_DANGER}; font-size: 18px; background: transparent;"
        )
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet(
            f"background-color: rgba(248, 81, 73, 0.15); "
            f"color: {THEME_DANGER}; border-radius: 10px; "
            f"padding: 4px 12px; font-size: 12px; font-weight: 600;"
        )

    def _set_status_loading(self):
        self.status_dot.setStyleSheet(
            f"color: {THEME_ACCENT}; font-size: 18px; background: transparent;"
        )
        self.status_label.setText("Loading Models…")
        self.status_label.setStyleSheet(
            f"background-color: rgba(88, 166, 255, 0.15); "
            f"color: {THEME_ACCENT}; border-radius: 10px; "
            f"padding: 4px 12px; font-size: 12px; font-weight: 600;"
        )

    # ==========================================
    # 🎬 VIDEO THREAD MANAGEMENT
    # ==========================================

    def _start_thread(self, source_type: SourceType, source_path: str = ""):
        """Create, configure, connect, and start the video thread."""
        # Stop existing thread if any
        self._stop_thread()

        self._video_thread = VideoThread(roi_config=getattr(self, '_roi_config', None), parent=self)
        self._video_thread.set_source(source_type, source_path)

        # Apply current UI state to the new analyzer instance
        self._video_thread.analyzer.divider_x_ratio = self.control_panel.slider_divider.value() / 100.0
        self._video_thread.analyzer.direction_mode = self.control_panel.combo_direction.itemData(
            self.control_panel.combo_direction.currentIndex()
        )
        self._video_thread.analyzer.speed_enabled = self.control_panel.chk_speed.isChecked()
        self._video_thread.analyzer.wrong_way_enabled = self.control_panel.chk_wrong_way.isChecked()
        self._video_thread.analyzer.density_enabled = self.control_panel.chk_density.isChecked()

        # Connect signals
        self._video_thread.frame_ready.connect(self._on_frame_ready)
        self._video_thread.error_occurred.connect(self._on_error)
        self._video_thread.source_ended.connect(self._on_source_ended)
        self._video_thread.model_loaded.connect(self._on_model_loaded)
        self._video_thread.session_update.connect(self._on_session_update)
        self._video_thread.finished_processing.connect(self._on_finished_processing)
        self._video_thread.accident_detected.connect(self._on_accident_detected)
        self._video_thread.scan_complete.connect(self._on_scan_complete)

        # Update UI state
        self._set_status_loading()
        self.control_panel.set_running_state()
        self.charts_panel.reset()
        self.events_panel.clear()
        self.vehicle_table_panel.clear()  # NEW in v2.0
        self.incident_reports_panel.clear()

        self._video_thread.start()

    def _stop_thread(self):
        """Stop the video thread gracefully."""
        if self._video_thread is not None and self._video_thread.isRunning():
            self._video_thread.stop()
            self._video_thread.wait(3000)  # Wait up to 3 seconds
            self._video_thread = None

    # ==========================================
    # 📡 SLOT: Control Panel Actions
    # ==========================================

    def _on_select_camera(self):
        self._pending_source = (SourceType.CAMERA, "")
        self.status_label.setText("Source: Camera")
        self.status_dot.setStyleSheet("color: #d29922; font-size: 18px; background: transparent;")
        self._preview_frame_raw = None
        self.control_panel.btn_start.setEnabled(True)

    def _on_select_video(self):
        import os
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)",
        )
        if file_path:
            self._pending_source = (SourceType.VIDEO_FILE, file_path)
            self.status_label.setText(f"Source: {os.path.basename(file_path)}")
            self.status_dot.setStyleSheet("color: #d29922; font-size: 18px; background: transparent;")
            self._generate_preview()

    def _on_select_ip(self, url: str):
        if not url:
            QMessageBox.warning(self, "Input Required",
                                "Please enter an IP camera URL.")
            return
        self._pending_source = (SourceType.IP_CAMERA, url)
        self.status_label.setText("Source: IP Camera")
        self.status_dot.setStyleSheet("color: #d29922; font-size: 18px; background: transparent;")
        self._preview_frame_raw = None
        self.control_panel.btn_start.setEnabled(True)

    def _generate_preview(self):
        import cv2
        from core.video_thread import logger
        self._preview_frame_raw = None
        source_type, source_path = getattr(self, '_pending_source', (None, None))
        
        if source_type == SourceType.VIDEO_FILE and source_path:
            try:
                cap = cv2.VideoCapture(source_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self._preview_frame_raw = frame
                cap.release()
            except Exception as e:
                logger.error(f"Failed to generate preview: {e}")
                
        # Enable start button if we have a source, even if preview failed
        self.control_panel.btn_start.setEnabled(True)
        
        if getattr(self, '_preview_frame_raw', None) is not None:
            self._render_preview()
            self.status_label.setText("Video loaded. Adjust direction, then press Start.")
        else:
            if source_type == SourceType.VIDEO_FILE:
                self.status_label.setText("Video loaded (no preview). Press Start.")

    def _render_preview(self):
        import cv2
        from PyQt6.QtGui import QImage
        
        if getattr(self, '_preview_frame_raw', None) is None:
            return
            
        frame = self._preview_frame_raw.copy()
        height, width = frame.shape[:2]
        
        # Get divider ratio
        divider_ratio = self.control_panel.slider_divider.value() / 100.0
        divider_x = int(width * divider_ratio)
        
        # Draw labels
        mode_key = self.control_panel.combo_direction.itemData(self.control_panel.combo_direction.currentIndex())
        lane_color = (55, 169, 235)  # BGR for EBA937
        
        if mode_key in [DIRECTION_MODE_ONEWAY_UP, DIRECTION_MODE_ONEWAY_DOWN]:
            expected = "UP WAY" if mode_key == DIRECTION_MODE_ONEWAY_UP else "DOWN WAY"
            center_x = width // 2
            cv2.putText(frame, f"ALL {expected}", (max(10, center_x - 60), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_color, 2)
        else:
            # Draw line
            cv2.line(frame, (divider_x, 0), (divider_x, height), lane_color, 2)
            
            if mode_key == DIRECTION_MODE_STANDARD:
                left_label, right_label = "DOWN WAY", "UP WAY"
            else:
                left_label, right_label = "UP WAY", "DOWN WAY"
                
            left_center = divider_x // 2
            right_center = divider_x + (width - divider_x) // 2
            
            cv2.putText(frame, left_label, (max(10, left_center - 80), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_color, 2)
            cv2.putText(frame, right_label, (max(10, right_center - 60), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_color, 2)
        
        # Draw ROI overlay for preview
        import numpy as np
        roi_config = getattr(self, '_roi_config', None)
        if not roi_config or roi_config.get("show_roi_overlay", True):
            if roi_config and roi_config.get("mode") == "manual":
                pts = np.int32(roi_config["src_points"]).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
            else:
                bottom_y = int(height * 0.90)
                bottom_left_x = int(width * 0.10)
                bottom_right_x = int(width * 0.90)
                top_y = int(height * 0.40)
                top_left_x = int(width * 0.30)
                top_right_x = int(width * 0.70)
                pts = np.int32([
                    [top_left_x, top_y],
                    [top_right_x, top_y],
                    [bottom_right_x, bottom_y],
                    [bottom_left_x, bottom_y]
                ]).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255), thickness=2)

        # Convert to QImage and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_panel.update_frame(qt_image)

    def _open_calibration_dialog(self):
        """Open the dialog to edit the BEV ROI and lane settings."""
        if getattr(self, '_preview_frame_raw', None) is None:
            QMessageBox.warning(self, "No Video Source", "Please select a video file or camera to set up calibration.")
            return

        from ui.calibration_dialog import CalibrationDialog
        dialog = CalibrationDialog(self._preview_frame_raw, getattr(self, '_roi_config', None), self)
        if dialog.exec():
            self._roi_config = dialog.current_config
            self._render_preview()

    def _on_start(self):
        source = getattr(self, '_pending_source', (SourceType.CAMERA, ""))
        self._preview_frame_raw = None
        self._start_thread(*source)

    def _on_stop(self):
        self._stop_thread()
        self._set_status_stopped()
        self.control_panel.set_stopped_state()
        self.video_panel.clear()
        self.vehicle_table_panel.clear()  # NEW in v2.0
        self.fps_label.setText("FPS: --")

    def _on_pause(self):
        if self._video_thread is None:
            return
        if self._video_thread.is_paused:
            self._video_thread.resume()
        else:
            self._video_thread.pause()

    def _on_toggle_speed(self, enabled: bool):
        if self._video_thread:
            self._video_thread.analyzer.speed_enabled = enabled

    def _on_toggle_wrong_way(self, enabled: bool):
        if self._video_thread:
            self._video_thread.analyzer.wrong_way_enabled = enabled

    def _on_toggle_density(self, enabled: bool):
        if self._video_thread:
            self._video_thread.analyzer.density_enabled = enabled

    def _on_direction_mode_changed(self, mode_key: str):
        """Update the analyzer's direction mode in real-time (no stream restart)."""
        if self._video_thread:
            self._video_thread.analyzer.direction_mode = mode_key
        else:
            self._render_preview()
        label = DIRECTION_MODE_LABELS.get(mode_key, mode_key)
        self.mode_badge.setText(f"Mode: {label}")

    def _on_divider_slider_moved(self, ratio: float):
        """Update the analyzer's dividing lane ratio."""
        if self._video_thread:
            self._video_thread.analyzer.divider_x_ratio = ratio
        else:
            self._render_preview()

    # ==========================================
    # 📡 SLOT: Video Thread Signals
    # ==========================================

    def _on_frame_ready(self, qt_image, result: FrameResult):
        """
        Handle a processed frame from the video thread.

        v2.0 Changes:
            - Updates the new VehicleTablePanel with vehicle list.
            - Passes vehicle_desc and plate_number to events panel.
        """
        # Update video display
        self.video_panel.update_frame(qt_image)

        # Update FPS
        self.fps_label.setText(f"FPS: {result.fps}")

        # Update statistics
        self.stats_panel.update_stats(
            result.vehicle_count,
            result.average_speed,
            result.density_status,
        )

        # Vehicle table is now updated via session_update signal (v2.1)

        # Update charts (throttle: every 3rd frame to reduce chart overhead)
        if result.frame_number % 3 == 0:
            self.charts_panel.update_data(
                result.vehicle_count,
                result.average_speed,
                result.density_status,
            )

        # --- CHANGED: Add events with enriched data and evidence ---
        for event in result.events:
            self.events_panel.add_event(
                event.vehicle_id,
                event.event_type,
                event.timestamp,
                event.lane,
                plate_number=event.plate_number,
                snapshot_path=getattr(event, 'snapshot_path', ""),
                event_id=getattr(event, 'event_id', ""),
                duration=getattr(event, 'duration', 0.0),
                is_update=getattr(event, 'is_update', False)
            )

    def _on_model_loaded(self):
        """Called when YOLO model finishes loading."""
        self._set_status_running()

    def _on_error(self, message: str):
        """Handle errors from the video thread."""
        self._on_stop()
        QMessageBox.critical(self, "Error", message)

    def _on_source_ended(self):
        """Handle video file reaching the end."""
        self._on_stop()

    def _on_accident_detected(self, result: dict):
        """Handle new confirmed accident from the background worker."""
        # 1. Update Incident Reports panel
        self.incident_reports_panel.add_report(result)
        
        # 2. Flash UI Alert if Confirmed
        if result.get("confirmed_accident", False):
            # Change Title Bar style to red warning
            self.status_dot.setStyleSheet(f"color: {THEME_DANGER}; font-size: 18px; background: transparent;")
            self.status_label.setText("INCIDENT DETECTED")
            self.status_label.setStyleSheet(
                f"background-color: {THEME_DANGER}; "
                f"color: #ffffff; border-radius: 10px; "
                f"padding: 4px 12px; font-size: 12px; font-weight: bold;"
            )
            # Switch to Incident Reports tab
            self.tab_widget.setCurrentIndex(1)

    # ==========================================
    # 📊 SESSION & SUMMARY (v2.1)
    # ==========================================

    def _on_session_update(self, session_data: dict, current_frame: int):
        """Update vehicle dashboard from persistent session data (flicker-free)."""
        if self._video_thread and self._video_thread.isRunning():
            self.vehicle_table_panel.update_from_session(session_data, current_frame)

    def _on_finished_processing(self, session_data: dict):
        """Handle end-of-session: display summary popup and CSV notification."""
        if not session_data:
            return
        self._show_summary_dialog(session_data)

    def _show_summary_dialog(self, session_data: dict):
        """Display a comprehensive end-of-session summary dialog."""
        total_vehicles = len(session_data)

        # Aggregate statistics
        all_avg_speeds = []
        direction_counts = {"UP": 0, "DOWN": 0, "UNKNOWN": 0}
        detected_plates = []
        wrong_way_count = 0
        sudden_stop_count = 0

        for tid, v in session_data.items():
            avg_spd = v.get("avg_speed_kmh", 0)
            if avg_spd > 0:
                all_avg_speeds.append(avg_spd)

            d = v.get("direction", "UNKNOWN")
            direction_counts[d] = direction_counts.get(d, 0) + 1

            plate = v.get("plate_number", "—")
            if plate and plate not in ("—", "Unreadable", "⏳"):
                detected_plates.append(
                    f"ID {tid}: <b>{plate}</b>"
                )

            if v.get("had_wrong_way"):
                wrong_way_count += 1
            if v.get("had_sudden_stop"):
                sudden_stop_count += 1

        avg_speed = (
            int(sum(all_avg_speeds) / len(all_avg_speeds))
            if all_avg_speeds else 0
        )
        max_speed = round(
            max(
                (v.get("max_speed_kmh", 0) for v in session_data.values()),
                default=0,
            ),
            1,
        )

        # Build plates HTML block
        if detected_plates:
            plates_items = "".join(f"<li>{p}</li>" for p in detected_plates)
            plates_html = (
                '<h3 style="color:#39d2c0;margin-top:14px;">'
                '🔢 Detected License Plates</h3>'
                f'<ul style="color:#e6edf3;line-height:1.8;">'
                f'{plates_items}</ul>'
            )
        else:
            plates_html = (
                '<p style="color:#6e7681;margin-top:14px;">'
                'No plates detected in this session.</p>'
            )

        # Alerts HTML
        alerts_html = ""
        if wrong_way_count > 0 or sudden_stop_count > 0:
            alerts_html = (
                '<h3 style="color:#f85149;margin-top:14px;">'
                '🚨 Behavioral Alerts</h3>'
                '<table style="width:100%;border-collapse:collapse;">'
            )
            if wrong_way_count > 0:
                alerts_html += (
                    '<tr><td style="padding:4px;color:#f85149;">'
                    '⚠️ Wrong-Way Vehicles</td>'
                    f'<td style="padding:4px;color:#e6edf3;text-align:right;'
                    f'font-weight:bold;">{wrong_way_count}</td></tr>'
                )
            if sudden_stop_count > 0:
                alerts_html += (
                    '<tr><td style="padding:4px;color:#d29922;">'
                    '🛑 Sudden Stop Vehicles</td>'
                    f'<td style="padding:4px;color:#e6edf3;text-align:right;'
                    f'font-weight:bold;">{sudden_stop_count}</td></tr>'
                )
            alerts_html += '</table>'

        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;padding:8px;">
            <h2 style="color:#58a6ff;text-align:center;margin-bottom:4px;">
                📊 RoadVision AI - Session Summary
            </h2>
            <hr style="border-color:#30363d;">

            <table style="width:100%;border-collapse:collapse;margin:8px 0;">
                <tr>
                    <td style="padding:7px;color:#8b949e;">Total Vehicles Tracked</td>
                    <td style="padding:7px;color:#58a6ff;font-weight:bold;
                        font-size:15px;text-align:right;">{total_vehicles}</td>
                </tr>
                <tr>
                    <td style="padding:7px;color:#8b949e;">Average Speed</td>
                    <td style="padding:7px;color:#3fb950;font-weight:bold;
                        font-size:15px;text-align:right;">{avg_speed} km/h</td>
                </tr>
                <tr>
                    <td style="padding:7px;color:#8b949e;">Max Speed Recorded</td>
                    <td style="padding:7px;color:#f85149;font-weight:bold;
                        font-size:15px;text-align:right;">{max_speed} km/h</td>
                </tr>
            </table>

            <h3 style="color:#d29922;margin-top:14px;">🧭 Direction Breakdown</h3>
            <table style="width:100%;border-collapse:collapse;margin:8px 0;">
                <tr>
                    <td style="padding:4px;color:#58a6ff;">↑ Upstream</td>
                    <td style="padding:4px;color:#e6edf3;text-align:right;">
                        {direction_counts.get('UP', 0)} vehicles</td>
                </tr>
                <tr>
                    <td style="padding:4px;color:#3fb950;">↓ Downstream</td>
                    <td style="padding:4px;color:#e6edf3;text-align:right;">
                        {direction_counts.get('DOWN', 0)} vehicles</td>
                </tr>
            </table>

            {alerts_html}
            {plates_html}

            <hr style="border-color:#30363d;margin-top:14px;">
            <p style="color:#3fb950;text-align:center;font-size:11px;">
                ✅ Summary automatically exported to CSV in project directory
            </p>
        </div>
        """

        dialog = QDialog(self)
        dialog.setWindowTitle("RoadVision AI - Session Summary")
        dialog.setMinimumSize(480, 540)
        dialog.setStyleSheet(
            "QDialog { background-color: #161b22; }"
            "QTextBrowser { background-color: #0d1117;"
            " border: 1px solid #30363d; border-radius: 8px; padding: 4px; }"
            "QPushButton { background-color: #21262d; color: #e6edf3;"
            " border: 1px solid #30363d; border-radius: 6px;"
            " padding: 8px 24px; font-weight: 600; }"
            "QPushButton:hover { background-color: #30363d;"
            " border-color: #58a6ff; }"
        )

        dlg_layout = QVBoxLayout(dialog)

        browser = QTextBrowser()
        browser.setHtml(html)
        browser.setOpenExternalLinks(False)
        dlg_layout.addWidget(browser)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        dlg_layout.addWidget(btn_box)

        dialog.exec()

    # ==========================================
    # 🧹 CLEANUP
    # ==========================================

    def closeEvent(self, event):
        """Ensure thread is stopped when window closes."""
        self._stop_thread()
        super().closeEvent(event)
