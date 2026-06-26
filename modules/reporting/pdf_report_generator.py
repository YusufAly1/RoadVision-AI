import os
import json
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

logger = logging.getLogger(__name__)

def safe_get(data: dict, keys: list, default="N/A") -> str:
    """Safely traverse a dictionary using a list of keys."""
    curr = data
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return str(curr) if curr is not None and str(curr).strip() != "" else default

def format_list_or_text(value) -> str:
    """Format lists into bullet points or return the text."""
    if isinstance(value, list):
        if not value:
            return "N/A"
        return "<br/>".join([f"&bull; {str(item)}" for item in value])
    return str(value) if value else "N/A"

def add_section_title(story: list, title: str, styles):
    """Add a styled section title to the PDF story."""
    story.append(Spacer(1, 15))
    story.append(Paragraph(title, styles['Heading2']))
    story.append(Spacer(1, 5))

def add_key_value_table(story: list, rows: list, styles):
    """Add a structured key-value table to the PDF story."""
    if not rows:
        return
    
    table_data = []
    for r in rows:
        # Wrap the values in paragraphs to ensure text wrapping
        key = Paragraph(f"<b>{r[0]}</b>", styles['Normal'])
        val = Paragraph(str(r[1]), styles['Normal'])
        table_data.append([key, val])
        
    t = Table(table_data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

def add_evidence_images(story: list, image_paths: list, evidence_meta: list = None, styles=None):
    """Add evidence images safely, respecting aspect ratio, with dynamic captions."""
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                # Use reportlab Image
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(img_path)
                width, height = ir.getSize()
                
                # Target max width of 400
                max_w = 400
                if width > max_w:
                    ratio = max_w / width
                    width = max_w
                    height = height * ratio
                    
                story.append(Image(img_path, width=width, height=height))
                
                # Add caption if metadata is available
                if evidence_meta and styles:
                    for meta in evidence_meta:
                        if meta.get("path") == img_path:
                            caption = f"<b>{meta.get('id', 'Unknown ID')}</b> | Time relative to impact: {meta.get('relative_time', '0')}s | {meta.get('selection_reason', 'Evidence Image')}"
                            from reportlab.platypus import Paragraph
                            story.append(Spacer(1, 4))
                            story.append(Paragraph(f"<font size=9 color='grey'>{caption}</font>", styles['Normal']))
                            break
                            
                story.append(Spacer(1, 15))
            except Exception as e:
                logger.warning(f"Failed to add image {img_path} to PDF: {e}")

def generate_pdf_from_json(json_path: str, output_pdf_path: str = None) -> str:
    """
    Generate a professional PDF report from an AI JSON report.
    Returns the path to the generated PDF.
    Does not crash on missing fields or errors.
    """
    try:
        logger.info(f"Generating PDF report from: {json_path}")
        
        if not os.path.exists(json_path):
            logger.error(f"JSON report not found: {json_path}")
            return ""
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not output_pdf_path:
            # Default to saving next to the json file
            base_dir = os.path.dirname(json_path)
            output_pdf_path = os.path.join(base_dir, "accident_report.pdf")
            
        # Create PDF document
        doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        
        # Add a custom title style
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=18, spaceAfter=20)
        
        story = []
        
        # Title
        title_text = data.get("report_title", "RoadVision Accident Analysis Report")
        story.append(Paragraph(title_text, title_style))
        story.append(Paragraph(f"<i>{data.get('system_name', 'System')} — {data.get('subtitle', 'Report')}</i>", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 1. Incident Details (Table)
        add_section_title(story, "1. Incident Details", styles)
        details = [
            ("Incident ID", safe_get(data, ["incident_id"])),
            ("Status", safe_get(data, ["status"])),
            ("Trigger Frame", safe_get(data, ["frame"])),
            ("Confidence", safe_get(data, ["accident_confidence_percent"])),
            ("Video Source", safe_get(data, ["video_source"]))
        ]
        add_key_value_table(story, details, styles)
        
        # 2. Executive Summary
        add_section_title(story, "2. Executive Summary", styles)
        summary = safe_get(data, ["executive_summary"])
        story.append(Paragraph(summary, styles['Normal']))
        
        # 3. Evidence Review
        add_section_title(story, "3. Evidence Review", styles)
        review = safe_get(data, ["evidence_review"])
        story.append(Paragraph(review, styles['Normal']))
        
        # 4. Vehicle Involvement
        vehicles = data.get("vehicles", [])
        if vehicles:
            add_section_title(story, "4. Vehicle Involvement", styles)
            for v in vehicles:
                v_data = [
                    ("Vehicle ID", safe_get(v, ["vehicle_id"])),
                    ("Type", safe_get(v, ["vehicle_type"])),
                    ("Plate", safe_get(v, ["plate_number"])),
                    ("Speed", safe_get(v, ["speed"])),
                    ("Violations", format_list_or_text(v.get("violations", [])))
                ]
                add_key_value_table(story, v_data, styles)
                
        # 5. Sequence of Events
        add_section_title(story, "5. Sequence of Events", styles)
        seq = data.get("sequence_of_events", [])
        story.append(Paragraph(format_list_or_text(seq), styles['Normal']))
        
        # 6. Contributing Factors
        add_section_title(story, "6. Contributing Factors", styles)
        factors = data.get("contributing_factors", [])
        story.append(Paragraph(format_list_or_text(factors), styles['Normal']))
        
        # 7. Fault Assessment
        add_section_title(story, "7. Fault Assessment", styles)
        fault_data = [
            ("Likely Responsible Vehicle", safe_get(data, ["fault_assessment", "likely_responsible_vehicle_id"])),
            ("Responsible Plate", safe_get(data, ["fault_assessment", "likely_responsible_plate"])),
            ("Confidence", safe_get(data, ["fault_assessment", "confidence"])),
            ("Reasoning", safe_get(data, ["fault_assessment", "reason"]))
        ]
        add_key_value_table(story, fault_data, styles)
        
        # 8. Final Determination
        add_section_title(story, "8. Final Determination", styles)
        conf_ass = safe_get(data, ["confidence_assessment"])
        final_det = safe_get(data, ["final_determination"])
        story.append(Paragraph(f"<i>{conf_ass}</i>", styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Conclusion:</b> {final_det}", styles['Normal']))
        
        # 9. Evidence Gallery
        evidence_paths = data.get("evidence_paths", [])
        if evidence_paths:
            add_section_title(story, "9. Evidence Gallery", styles)
            add_evidence_images(story, evidence_paths, data.get("evidence_meta", []), styles)
            
        
        # Add AI Disclaimer
        story.append(Spacer(1, 30))
        from reportlab.platypus import HRFlowable
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceBefore=1, spaceAfter=5))
        disclaimer_text = "<b>AI-assisted assessment:</b> The system identifies the vehicle most likely at fault based on the available video and telemetry. Final legal responsibility must be determined by authorized investigators."
        story.append(Paragraph(f"<font size=8 color='grey'><i>{disclaimer_text}</i></font>", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Build the PDF
        doc.build(story)
        logger.info(f"Successfully created PDF report: {output_pdf_path}")
        return output_pdf_path

    except Exception as e:
        logger.error(f"Failed to generate PDF report from JSON: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
