import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QSplitter, QPushButton, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from ui.styles import THEME_BG_CARD, THEME_BORDER, THEME_TEXT_PRIMARY, THEME_TEXT_MUTED, THEME_SUCCESS, THEME_DANGER, THEME_ACCENT

class IncidentReportsPanel(QWidget):
    """
    Panel to display confirmed accident reports and evidence frames.
    Refactored to Master-Detail View.
    """
    def __init__(self):
        super().__init__()
        self.reports_data = {}  # Store full report data by frame_idx
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # ==========================================
        # LEFT PANE: Master List
        # ==========================================
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        header = QLabel("🚨 Live Incident Feed")
        header.setStyleSheet(f"color: {THEME_DANGER}; font-size: 16px; font-weight: bold;")
        left_layout.addWidget(header)
        
        self.master_scroll = QScrollArea()
        self.master_scroll.setWidgetResizable(True)
        self.master_scroll.setStyleSheet("QScrollArea { border: 1px solid #30363d; background: transparent; border-radius: 4px; }")
        
        self.master_container = QWidget()
        self.master_layout = QVBoxLayout(self.master_container)
        self.master_layout.setContentsMargins(8, 8, 8, 8)
        self.master_layout.setSpacing(8)
        self.master_layout.addStretch()
        
        self.master_scroll.setWidget(self.master_container)
        left_layout.addWidget(self.master_scroll)
        
        splitter.addWidget(left_pane)
        
        # ==========================================
        # RIGHT PANE: Deep Dive View
        # ==========================================
        right_pane = QWidget()
        self.right_layout = QVBoxLayout(right_pane)
        self.right_layout.setContentsMargins(16, 0, 0, 0)
        self.right_layout.setSpacing(16)
        
        # Header & Download button
        header_layout = QHBoxLayout()
        self.detail_header = QLabel("Deep Dive View")
        self.detail_header.setStyleSheet(f"color: #e6edf3; font-size: 20px; font-weight: bold;")
        
        self.btn_download = QPushButton("⬇️ Download JSON Report")
        self.btn_download.setStyleSheet(f"background-color: {THEME_ACCENT}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        self.btn_download.clicked.connect(self._on_download_clicked)
        self.btn_download.hide() # Hide until a report is selected
        
        self.btn_download_pdf = QPushButton("📄 Download PDF Report")
        self.btn_download_pdf.setStyleSheet(f"background-color: #238636; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        self.btn_download_pdf.clicked.connect(self._on_download_pdf_clicked)
        self.btn_download_pdf.hide()
        
        self.current_report_data = None
        
        header_layout.addWidget(self.detail_header)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_download_pdf)
        header_layout.addWidget(self.btn_download)
        self.right_layout.addLayout(header_layout)
        
        # Main Report Block
        self.report_block = QFrame()
        self.report_block.setStyleSheet(f"background-color: {THEME_BG_CARD}; border: 1px solid {THEME_BORDER}; border-radius: 6px;")
        report_layout = QVBoxLayout(self.report_block)
        report_layout.setContentsMargins(16, 16, 16, 16)
        
        from PyQt6.QtWidgets import QTextBrowser
        self.report_content = QTextBrowser()
        self.report_content.setOpenExternalLinks(False)
        self.report_content.setStyleSheet(f"border: none; background: transparent; color: {THEME_TEXT_PRIMARY}; font-size: 14px;")
        self.report_content.setHtml("<p style='color: #8b949e;'>Select an incident to view the official report.</p>")
        
        report_layout.addWidget(self.report_content)
        self.right_layout.addWidget(self.report_block)
        
        # Evidence Gallery
        self.gallery_block = QFrame()
        self.gallery_block.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gallery_block.setStyleSheet("border: none; background: transparent;")
        gallery_layout = QVBoxLayout(self.gallery_block)
        gallery_layout.setContentsMargins(0,0,0,0)
        
        gallery_title = QLabel("📸 Evidence Gallery")
        gallery_title.setStyleSheet("font-weight: bold; color: #e6edf3;")
        gallery_layout.addWidget(gallery_title)
        
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gallery_scroll.setStyleSheet("QScrollArea { border: 1px solid #30363d; background: transparent; border-radius: 4px; }")
        
        self.gallery_container = QWidget()
        self.gallery_container_layout = QHBoxLayout(self.gallery_container)
        self.gallery_container_layout.setContentsMargins(8, 8, 8, 8)
        self.gallery_container_layout.setSpacing(12)
        self.gallery_container_layout.addStretch()
        
        self.gallery_scroll.setWidget(self.gallery_container)
        gallery_layout.addWidget(self.gallery_scroll)
        
        self.right_layout.addWidget(self.gallery_block)
        
        splitter.addWidget(right_pane)
        splitter.setSizes([300, 700])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        self.layout.addWidget(splitter)

    def add_report(self, report_data: dict):
        frame_idx = str(report_data.get("frame", "0"))
        self.reports_data[frame_idx] = report_data
        
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background-color: {THEME_BG_CARD}; border: 1px solid {THEME_BORDER}; border-radius: 6px; }} QFrame:hover {{ border: 1px solid #58a6ff; }}")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card_layout = QVBoxLayout(card)
        
        # Backwards compatibility check
        if "status" in report_data:
            status_text = report_data.get("status")
            is_accident = "Confirmed Accident" in status_text or "Suspected Accident" in status_text
        else:
            gemini = report_data.get("gemini_verification", {})
            is_accident = gemini.get("confirmed_accident", False)
            status_text = "🔴 Accident" if is_accident else "🟢 Normal"
            
        status_color = THEME_DANGER if is_accident else THEME_SUCCESS
        
        # Use simple label on the card
        card_status = "🔴 Action Required" if is_accident else "🟢 Routine"
        
        title = QLabel(f"<b>Frame {frame_idx}</b>")
        title.setStyleSheet("color: #e6edf3; border: none;")
        status = QLabel(card_status)
        status.setStyleSheet(f"color: {status_color}; font-weight: bold; border: none;")
        
        card_layout.addWidget(title)
        card_layout.addWidget(status)
        
        # Click event to load right pane
        card.mousePressEvent = lambda e, f=frame_idx: self._load_deep_dive(f)
        
        self.master_layout.insertWidget(0, card)

    def _load_deep_dive(self, frame_idx: str):
        data = self.reports_data.get(frame_idx)
        if not data: return
        self.current_report_data = data
        
        self.detail_header.setText(f"Deep Dive View — Frame {frame_idx}")
        self.btn_download.show()
        self.btn_download_pdf.show()
        
        # Format the new structured JSON as a professional HTML report
        
        # Fallback support for older report schemas
        if "fault_vehicle" not in data:
            self.report_content.setHtml(f"<p style='color: {THEME_TEXT_PRIMARY};'>Legacy report schema detected. Please review raw JSON for details.</p>")
        else:
            status_val = data.get('status', 'Unknown')
            status_color = "#f85149" if "Accident" in status_val else "#3fb950" if "Normal" in status_val else "#d29922"
            
            fault = data.get('fault_vehicle', {})
            vid = fault.get('vehicle_id', 'Undetermined')
            plate = fault.get('plate_number', 'Unknown')
            fault_reason = fault.get('reason', '')
            
            what_happened_html = "".join([f"<li>{item}</li>" for item in data.get('what_happened', [])])
            
            tech = data.get('technical_details', {})
            tech_status = tech.get('gemini_status', '')
            tech_log = tech.get('raw_gemini_log', '')
            
            fault_ass = data.get('fault_assessment', {})
            likely_vid = fault_ass.get('likely_responsible_vehicle_id', vid)
            likely_plate = fault_ass.get('likely_responsible_plate', plate)
            fault_confidence = fault_ass.get('confidence', 'Unknown')
            reason = fault_ass.get('reason', fault_reason)
            
            exec_summary = data.get('executive_summary', data.get('summary', ''))
            evidence_review = data.get('evidence_review', '')
            
            vehicles = data.get('vehicles', [])
            vehicles_html = ""
            if vehicles:
                for v in vehicles:
                    vehicles_html += f"<li><b>Vehicle {v.get('vehicle_id', 'Unknown')}:</b> Plate {v.get('plate_number', 'Unknown')} | Speed: {v.get('speed', 'Unknown')} | Violations: {v.get('violations', 'None')}</li>"
            
            seq_events = data.get('sequence_of_events', data.get('what_happened', []))
            seq_html = "".join([f"<li>{item}</li>" for item in seq_events])
            
            contrib_factors = data.get('contributing_factors', [])
            contrib_html = "".join([f"<li>{item}</li>" for item in contrib_factors]) if contrib_factors else "<li>None specified</li>"
            
            conf_assessment = data.get('confidence_assessment', '')
            final_det = data.get('final_determination', '')
            
            html = f"""
            <h2 style='color: #58a6ff; margin-bottom: 2px;'>{data.get('report_title', 'Accident Investigation Report')}</h2>
            <p style='color: #8b949e; margin-top: 0; font-size: 12px;'><i>{data.get('system_name')} — {data.get('subtitle')} | Incident ID: {data.get('incident_id')}</i></p>
            
            <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>1. Executive Summary</h3>
            <p><b>Status:</b> <span style='color: {status_color}; font-weight: bold;'>{status_val}</span></p>
            <p>{exec_summary}</p>
            
            <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>2. Evidence Review</h3>
            <p>{evidence_review}</p>
            """
            
            if vehicles_html:
                html += f"""
                <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>3. Vehicle Involvement</h3>
                <ul>{vehicles_html}</ul>
                """
                
            html += f"""
            <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>4. Sequence of Events</h3>
            <ul>{seq_html}</ul>
            
            <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>5. Contributing Factors</h3>
            <ul>{contrib_html}</ul>
            
            <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>6. Fault Assessment</h3>
            <p><b>Likely Responsible Vehicle:</b> {likely_vid} / {likely_plate}</p>
            <p><b>Confidence Level:</b> {fault_confidence}</p>
            <p><b>Reasoning:</b> {reason}</p>
            """
            
            if conf_assessment or final_det:
                html += f"""
                <h3 style='color: #e6edf3; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px;'>7. Final Determination</h3>
                <p><i>{conf_assessment}</i></p>
                <p><b>Conclusion:</b> {final_det}</p>
                """
                
            html += f"""
            <h3 style='color: #8b949e; margin-top: 15px; border-bottom: 1px solid #30363d; padding-bottom: 5px; font-size: 12px;'>8. Technical Details</h3>
            <p style='color: #8b949e; font-size: 12px; margin-top: 5px;'>
                <b>Model:</b> {tech.get('model', 'Unknown')} (Peak Confidence: {data.get('accident_confidence_percent', 'N/A')})<br/>
                <b>AI Verification:</b> {'Enabled' if tech.get('gemini_enabled') else 'Disabled / Unavailable'} <br/>
                <b>Status:</b> {tech_status} <br/>
                <b>Local Fallback Used:</b> {tech.get('local_fallback_used', False)}
            </p>
            """
            
            # Append raw log if present and verification was unavailable
            if tech_log and "Unavailable" in status_val:
                html += f"""
                <details>
                    <summary style='color: #8b949e; font-size: 12px; cursor: pointer;'>View Raw API Log</summary>
                    <pre style='color: #8b949e; font-size: 10px; background: #0d1117; padding: 5px; border-radius: 4px;'>{str(tech_log)}</pre>
                </details>
                """
                
            self.report_content.setHtml(html)
        
        # Gallery
        while self.gallery_container_layout.count() > 1:
            item = self.gallery_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        evidence_paths = data.get("evidence_paths", [])
        if not evidence_paths:
            # Check nested evidence dict for legacy support
            evidence_paths = data.get("evidence", {}).get("evidence_frames", [])
            
        images_loaded = 0
        
        # New folder-based loading
        if evidence_paths:
            for img_path in evidence_paths:
                if os.path.exists(img_path):
                    pixmap = QPixmap(img_path)
                    if not pixmap.isNull():
                        img_label = QLabel()
                        img_label.setPixmap(pixmap.scaledToHeight(180, Qt.TransformationMode.SmoothTransformation))
                        img_label.setStyleSheet("border: 1px solid #30363d; border-radius: 4px;")
                        self.gallery_container_layout.insertWidget(self.gallery_container_layout.count() - 1, img_label)
                        images_loaded += 1
                        
        # Legacy fallback loading
        if images_loaded == 0:
            out_dir = os.path.abspath("accident_outputs")
            if os.path.exists(out_dir):
                for f in sorted(os.listdir(out_dir)):
                    if f.startswith(f"evidence_frame_{frame_idx}_"):
                        img_path = os.path.join(out_dir, f)
                        pixmap = QPixmap(img_path)
                        if not pixmap.isNull():
                            img_label = QLabel()
                            img_label.setPixmap(pixmap.scaledToHeight(180, Qt.TransformationMode.SmoothTransformation))
                            img_label.setStyleSheet("border: 1px solid #30363d; border-radius: 4px;")
                            self.gallery_container_layout.insertWidget(self.gallery_container_layout.count() - 1, img_label)
                            images_loaded += 1
                        
        if images_loaded == 0:
            no_evidence_label = QLabel("No Evidence Available")
            no_evidence_label.setStyleSheet(f"color: {THEME_TEXT_MUTED}; font-style: italic; font-size: 14px;")
            no_evidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery_container_layout.insertWidget(0, no_evidence_label)
                        
    def _on_download_clicked(self):
        if not self.current_report_data: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Accident Report", os.path.expanduser("~/Desktop/accident_report.json"), "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.current_report_data, f, indent=4)
                
    def _on_download_pdf_clicked(self):
        if not self.current_report_data: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", os.path.expanduser("~/Desktop/accident_report.pdf"), "PDF Files (*.pdf)")
        if not file_path: return
        
        import shutil
        pdf_source = self.current_report_data.get("pdf_report_path")
        
        if pdf_source and os.path.exists(pdf_source):
            try:
                shutil.copy(pdf_source, file_path)
                return
            except Exception as e:
                print(f"Failed to copy existing PDF: {e}")
                
        # Fallback: generate it now
        json_path = self.current_report_data.get("report_path")
        if json_path and os.path.exists(json_path):
            try:
                from modules.reporting.pdf_report_generator import generate_pdf_from_json
                out_pdf = generate_pdf_from_json(json_path, file_path)
                if not out_pdf:
                    print("PDF Generation returned empty.")
            except Exception as e:
                print(f"Failed to generate PDF: {e}")
        else:
            print("Source JSON not found to generate PDF.")

    def clear(self):
        """Clear all reports and reset the Deep Dive View (called when starting a new stream)."""
        self.reports_data.clear()
        self.current_report_data = None
        
        # Clear Master List (keep the stretch at the end)
        while self.master_layout.count() > 1:
            item = self.master_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Reset Deep Dive View
        self.detail_header.setText("Deep Dive View")
        self.btn_download.hide()
        self.btn_download_pdf.hide()
        self.report_content.setHtml("<p style='color: #8b949e;'>Select an incident to view the official report.</p>")
        
        # Clear Gallery
        while self.gallery_container_layout.count() > 1:
            item = self.gallery_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
