"""
Events Panel — Live event log table showing traffic incidents.

v2.0 Changes:
    - Expanded columns to include Vehicle Description and Plate Number.
    - Now shows 6 columns: Vehicle ID | Event Type | Timestamp | Lane | Description | Plate

Displays a scrolling table of detected events (Wrong Way, Sudden Stop)
with color-coded rows and auto-scroll.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QAbstractItemView, QLabel
)
from PyQt6.QtGui import QColor, QBrush, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush

from utils.constants import (
    MAX_EVENT_LOG_ROWS, THEME_DANGER, THEME_WARNING,
    THEME_TEXT_PRIMARY, THEME_BG_SECONDARY,
)


# Row background colors (with transparency)
EVENT_COLORS = {
    "WRONG WAY": QColor(248, 81, 73, 40),     # Red tint
    "SUDDEN STOP": QColor(210, 153, 34, 40),   # Orange tint
    "Accident-Related Sudden Stop": QColor(210, 153, 34, 40), # Orange tint
}

# Text accent colors
EVENT_TEXT_COLORS = {
    "WRONG WAY": QColor(THEME_DANGER),
    "SUDDEN STOP": QColor(THEME_WARNING),
    "Accident-Related Sudden Stop": QColor(THEME_DANGER), # Danger red
}


class EventsPanel(QWidget):
    """
    Panel to display a live feed of behavioral anomalies (wrong way, sudden stop).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.event_item_refs: dict[str, list[QTableWidgetItem]] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("  🚨 EVENTS & ALERTS")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 20, 8, 8)

        # --- CHANGED: Expanded from 5 to 6 columns for Evidence ---
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Vehicle ID", "Event Type", "Timestamp", "Lane",
            "Plate", "Snapshot"
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)

        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(2, 90)
        self.table.setColumnWidth(3, 60)
        self.table.setColumnWidth(5, 100)
        self.table.setColumnWidth(6, 120)

        group_layout.addWidget(self.table)
        layout.addWidget(group)

    def add_event(
        self,
        vehicle_id: int,
        event_type: str,
        timestamp: str,
        lane: str,
        plate_number: str = "",
        snapshot_path: str = "",
        event_id: str = "",
        duration: float = 0.0,
        is_update: bool = False
    ):
        """
        Add or update an event row in the table.
        """
        display_time = timestamp
        if duration > 0:
            display_time = f"{timestamp} ({duration:.1f}s)"

        if is_update and event_id and event_id in self.event_item_refs:
            items = self.event_item_refs[event_id]
            if len(items) >= 5:
                # Support upgrading event type and color
                items[1].setText(event_type)
                text_color = EVENT_TEXT_COLORS.get(event_type, QColor(THEME_TEXT_PRIMARY))
                items[1].setForeground(QBrush(text_color))
                bg_color = EVENT_COLORS.get(event_type, QColor(0, 0, 0, 0))
                for item in items:
                    item.setBackground(QBrush(bg_color))

                items[2].setText(display_time)
                
                plate_text = plate_number if plate_number else "—"
                items[4].setText(plate_text)
                if plate_text != "—":
                    items[4].setForeground(QBrush(QColor("#39d2c0")))
                    font = items[4].font()
                    font.setBold(True)
                    items[4].setFont(font)
                
                # Update Snapshot Thumbnail
                if snapshot_path:
                    img_label = QLabel()
                    pixmap = QPixmap(snapshot_path)
                    if not pixmap.isNull():
                        img_label.setPixmap(pixmap.scaled(100, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        row = self.table.row(items[0])
                        if row >= 0:
                            self.table.setCellWidget(row, 5, img_label)
            return

        # Remove oldest rows if over max
        while self.table.rowCount() >= MAX_EVENT_LOG_ROWS:
            # We don't remove from self.event_item_refs to avoid overhead, weak dict is better but this is fine
            self.table.removeRow(self.table.rowCount() - 1)

        # Insert at top
        self.table.insertRow(0)

        items = [
            str(vehicle_id),
            event_type,
            display_time,
            lane,
            plate_number if plate_number else "—",
        ]

        bg_color = EVENT_COLORS.get(event_type, QColor(0, 0, 0, 0))
        text_color = EVENT_TEXT_COLORS.get(event_type, QColor(THEME_TEXT_PRIMARY))
        
        row_items = []

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(QBrush(bg_color))

            # Color the event type text
            if col == 1:
                item.setForeground(QBrush(text_color))
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            # Plate number styling
            if col == 4 and text != "—":
                item.setForeground(QBrush(QColor("#39d2c0")))
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self.table.setItem(0, col, item)
            row_items.append(item)
            
        if event_id:
            self.event_item_refs[event_id] = row_items
            
        # Add Snapshot Thumbnail
        if snapshot_path:
            img_label = QLabel()
            pixmap = QPixmap(snapshot_path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(100, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                img_label.setText("No Image")
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(0, 5, img_label)
        else:
            item = QTableWidgetItem("—")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, 5, item)

        # Ensure row height fits the thumbnail
        self.table.setRowHeight(0, 65)

        # Auto-scroll to top
        self.table.scrollToTop()

    def clear(self):
        """Clear all events."""
        self.table.setRowCount(0)
