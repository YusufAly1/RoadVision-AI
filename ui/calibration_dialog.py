"""
Calibration Dialog — UI for setting ROI points and real-world dimensions.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
import cv2

class ROIOverlayWidget(QLabel):
    """Interactive label for dragging 4 ROI points over a video frame."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []  # List of points in original frame coordinates [x, y]
        self.dragging_idx = -1
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000000; border-radius: 4px;")
        
        self.original_w = 0
        self.original_h = 0
        self.qimage = None
        
    def set_image(self, qimage: QImage, original_w: int, original_h: int):
        self.original_w = original_w
        self.original_h = original_h
        self.qimage = qimage
        self.update()

    def set_points(self, points):
        """points are in original frame coordinates."""
        self.points = [[float(p[0]), float(p[1])] for p in points]
        self.update()

    def get_points(self):
        """Returns points in original frame coordinates."""
        return [[float(p[0]), float(p[1])] for p in self.points]

    def _get_geometry(self):
        if self.original_w == 0 or self.original_h == 0:
            return 1.0, 0.0, 0.0, 0.0, 0.0
        label_width = self.width()
        label_height = self.height()
        scale = min(label_width / self.original_w, label_height / self.original_h)
        displayed_width = self.original_w * scale
        displayed_height = self.original_h * scale
        offset_x = (label_width - displayed_width) / 2.0
        offset_y = (label_height - displayed_height) / 2.0
        return scale, offset_x, offset_y, displayed_width, displayed_height

    def mousePressEvent(self, event):
        if not self.points:
            return
        pos = event.position()
        scale, offset_x, offset_y, _, _ = self._get_geometry()
        
        # Find closest point
        min_dist = float('inf')
        closest_idx = -1
        for i, p in enumerate(self.points):
            disp_x = offset_x + p[0] * scale
            disp_y = offset_y + p[1] * scale
            dist = (disp_x - pos.x())**2 + (disp_y - pos.y())**2
            if dist < 400:  # 20 pixel radius
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        self.dragging_idx = closest_idx

    def mouseMoveEvent(self, event):
        if self.dragging_idx != -1:
            pos = event.position()
            scale, offset_x, offset_y, disp_w, disp_h = self._get_geometry()
            
            # Clamp to displayed image bounds
            mouse_x = max(offset_x, min(offset_x + disp_w, pos.x()))
            mouse_y = max(offset_y, min(offset_y + disp_h, pos.y()))
            
            # Convert back to original coordinates
            orig_x = (mouse_x - offset_x) / scale
            orig_y = (mouse_y - offset_y) / scale
            
            self.points[self.dragging_idx] = [orig_x, orig_y]
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_idx = -1

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.qimage:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        scale, offset_x, offset_y, disp_w, disp_h = self._get_geometry()
        
        from PyQt6.QtCore import QRectF
        target_rect = QRectF(offset_x, offset_y, disp_w, disp_h)
        painter.drawImage(target_rect, self.qimage)
        
        if len(self.points) == 4:
            disp_points = []
            for p in self.points:
                dx = offset_x + p[0] * scale
                dy = offset_y + p[1] * scale
                disp_points.append(QPointF(dx, dy))
                
            # Draw polygon
            poly = QPolygonF(disp_points)
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.setBrush(QColor(0, 200, 255, 50))
            painter.drawPolygon(poly)
            
            # Draw points
            labels = ["TL", "TR", "BR", "BL"]
            for i, p in enumerate(disp_points):
                painter.setBrush(QColor(255, 100, 100))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(p, 6, 6)
                
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(int(p.x()) + 10, int(p.y()) + 10, labels[i])


class CalibrationDialog(QDialog):
    """Dialog for editing ROI calibration."""
    
    def __init__(self, frame: np.ndarray, current_config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BEV ROI Calibration")
        self.setMinimumSize(800, 600)
        self.frame = frame
        self.original_h, self.original_w = frame.shape[:2]
        self.current_config = current_config
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Preview area
        self.overlay = ROIOverlayWidget()
        self.overlay.setMinimumSize(640, 360)
        
        rgb_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.overlay.set_image(qt_image, self.original_w, self.original_h)
        
        layout.addWidget(self.overlay, 1)

        # Controls
        ctrl_layout = QGridLayout()
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Auto Highway Mode", "Manual ROI Mode"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(QLabel("Calibration Mode:"), 0, 0)
        ctrl_layout.addWidget(self.combo_mode, 0, 1)
        
        self.combo_lanes = QComboBox()
        self.combo_lanes.addItems(["1 lane", "2 lanes", "3 lanes", "4 lanes", "Custom"])
        self.combo_lanes.setCurrentIndex(1) # Default 2 lanes
        self.combo_lanes.currentIndexChanged.connect(self._on_lanes_changed)
        ctrl_layout.addWidget(QLabel("Visible Lanes in ROI:"), 1, 0)
        ctrl_layout.addWidget(self.combo_lanes, 1, 1)
        
        self.spin_lane_width = QDoubleSpinBox()
        self.spin_lane_width.setRange(1.0, 10.0)
        self.spin_lane_width.setValue(3.5)
        self.spin_lane_width.setSuffix(" m")
        self.spin_lane_width.valueChanged.connect(self._update_real_width)
        ctrl_layout.addWidget(QLabel("Lane Width:"), 1, 2)
        ctrl_layout.addWidget(self.spin_lane_width, 1, 3)
        
        self.spin_real_width = QDoubleSpinBox()
        self.spin_real_width.setRange(1.0, 100.0)
        self.spin_real_width.setValue(7.0)
        self.spin_real_width.setSuffix(" m")
        ctrl_layout.addWidget(QLabel("Real ROI Width (m):"), 2, 0)
        ctrl_layout.addWidget(self.spin_real_width, 2, 1)
        
        self.combo_depth = QComboBox()
        self.combo_depth.addItems(["40 m", "50 m", "60 m", "Custom"])
        self.combo_depth.setCurrentIndex(1) # 50m
        self.combo_depth.currentIndexChanged.connect(self._on_depth_changed)
        ctrl_layout.addWidget(QLabel("Real ROI Depth (m):"), 2, 2)
        ctrl_layout.addWidget(self.combo_depth, 2, 3)
        
        self.spin_real_depth = QDoubleSpinBox()
        self.spin_real_depth.setRange(10.0, 200.0)
        self.spin_real_depth.setValue(50.0)
        self.spin_real_depth.setSuffix(" m")
        self.spin_real_depth.setEnabled(False)
        ctrl_layout.addWidget(QLabel("Custom Depth:"), 3, 2)
        ctrl_layout.addWidget(self.spin_real_depth, 3, 3)
        
        self.chk_overlay = QCheckBox("Show ROI Overlay on Live Video")
        self.chk_overlay.setChecked(True)
        ctrl_layout.addWidget(self.chk_overlay, 3, 0, 1, 2)
        
        layout.addLayout(ctrl_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to Auto")
        self.btn_reset.clicked.connect(self._reset_to_auto)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        self.btn_apply = QPushButton("Apply Calibration")
        self.btn_apply.setStyleSheet("background-color: #3fb950; color: white; font-weight: bold;")
        self.btn_apply.clicked.connect(self._apply)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)

    def _load_config(self):
        if self.current_config:
            if self.current_config["mode"] == "manual":
                self.combo_mode.setCurrentIndex(1)
                self.overlay.set_points(self.current_config["src_points"])
            else:
                self.combo_mode.setCurrentIndex(0)
                self._generate_auto_points()
                
            lane_count = self.current_config.get("lane_count", "2 lanes")
            idx = self.combo_lanes.findText(lane_count)
            if idx >= 0:
                self.combo_lanes.setCurrentIndex(idx)
            else:
                self.combo_lanes.setCurrentIndex(4) # Custom
                
            self.spin_lane_width.setValue(self.current_config.get("lane_width_m", 3.5))
            self.spin_real_width.setValue(self.current_config.get("real_width_m", 7.0))
            self.spin_real_depth.setValue(self.current_config.get("real_height_m", 50.0))
            self.chk_overlay.setChecked(self.current_config.get("show_roi_overlay", True))
        else:
            self._generate_auto_points()

    def _generate_auto_points(self):
        bottom_y = int(self.original_h * 0.90)
        bottom_left_x = int(self.original_w * 0.10)
        bottom_right_x = int(self.original_w * 0.90)
        top_y = int(self.original_h * 0.40)
        top_left_x = int(self.original_w * 0.30)
        top_right_x = int(self.original_w * 0.70)
        
        points = [
            [top_left_x, top_y],         # TL
            [top_right_x, top_y],        # TR
            [bottom_right_x, bottom_y],  # BR
            [bottom_left_x, bottom_y]    # BL
        ]
        self.overlay.set_points(points)

    def _on_mode_changed(self, idx):
        if idx == 0:
            self._generate_auto_points()

    def _on_lanes_changed(self, idx):
        if idx < 4:
            self._update_real_width()
        # If custom, don't auto update

    def _update_real_width(self):
        idx = self.combo_lanes.currentIndex()
        if idx < 4:
            lanes = idx + 1
            width = lanes * self.spin_lane_width.value()
            self.spin_real_width.setValue(width)

    def _on_depth_changed(self, idx):
        if idx == 0:
            self.spin_real_depth.setValue(40.0)
            self.spin_real_depth.setEnabled(False)
        elif idx == 1:
            self.spin_real_depth.setValue(50.0)
            self.spin_real_depth.setEnabled(False)
        elif idx == 2:
            self.spin_real_depth.setValue(60.0)
            self.spin_real_depth.setEnabled(False)
        else:
            self.spin_real_depth.setEnabled(True)

    def _reset_to_auto(self):
        self.combo_mode.setCurrentIndex(0)
        self.combo_lanes.setCurrentIndex(1)
        self.spin_lane_width.setValue(3.5)
        self.combo_depth.setCurrentIndex(1)
        self._generate_auto_points()

    def _is_convex(self, pts):
        """Check if 4 points form a convex polygon."""
        def cross_product(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
        res = [
            cross_product(pts[0], pts[1], pts[2]),
            cross_product(pts[1], pts[2], pts[3]),
            cross_product(pts[2], pts[3], pts[0]),
            cross_product(pts[3], pts[0], pts[1])
        ]
        return all(r > 0 for r in res) or all(r < 0 for r in res)

    def _check_intersections(self, pts):
        """Check for self-intersecting non-adjacent edges."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        # Edges are (0,1), (1,2), (2,3), (3,0)
        # Non-adjacent pairs: (0,1) and (2,3) | (1,2) and (3,0)
        if intersect(pts[0], pts[1], pts[2], pts[3]):
            return True
        if intersect(pts[1], pts[2], pts[3], pts[0]):
            return True
        return False

    def _apply(self):
        points = self.overlay.get_points()
        
        # Validation
        if len(points) != 4:
            QMessageBox.warning(self, "Invalid ROI", "ROI must have exactly 4 points.")
            return
            
        if self._check_intersections(points):
            QMessageBox.warning(self, "Invalid ROI", "ROI polygon edges cannot intersect.")
            return
            
        if not self._is_convex(points):
            QMessageBox.warning(self, "Invalid ROI", "ROI polygon must be convex.")
            return
            
        # Top points should be above bottom points (y is smaller)
        if points[0][1] >= points[3][1] or points[1][1] >= points[2][1]:
            QMessageBox.warning(self, "Invalid ROI", "Top points must be above bottom points.")
            return
            
        config = {
            "mode": "auto" if self.combo_mode.currentIndex() == 0 else "manual",
            "src_points": points,
            "real_width_m": self.spin_real_width.value(),
            "real_height_m": self.spin_real_depth.value(),
            "lane_count": self.combo_lanes.currentText(),
            "lane_width_m": self.spin_lane_width.value(),
            "show_roi_overlay": self.chk_overlay.isChecked()
        }
        
        self.current_config = config
        self.accept()
