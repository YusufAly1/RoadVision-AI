import os
import json
from modules.reporting.pdf_report_generator import generate_pdf_from_json

def test_generation():
    test_data = {
        "report_title": "RoadVision Accident Analysis Report",
        "system_name": "RoadVision AI",
        "subtitle": "Real-Time Traffic Intelligence System",
        "incident_id": "INC-20260613-100000",
        "status": "Confirmed Accident",
        "accident_confidence": 0.95,
        "accident_confidence_percent": "95.0%",
        "frame": 120,
        "video_source": "test_video.mp4",
        "executive_summary": "A sudden collision was detected involving a red sedan and a delivery truck. The collision caused significant traffic disruption.",
        "evidence_review": "VideoMAE and Gemini verified the anomaly at frame 120.",
        "vehicles": [
            {
                "vehicle_id": "V1",
                "vehicle_type": "Sedan",
                "plate_number": "ABC-123",
                "speed": "60 km/h",
                "violations": ["Sudden Stop"]
            },
            {
                "vehicle_id": "V2",
                "vehicle_type": "Truck",
                "plate_number": "XYZ-987",
                "speed": "45 km/h",
                "violations": []
            }
        ],
        "sequence_of_events": [
            "V1 stopped suddenly in the left lane.",
            "V2 was unable to brake in time and collided with the rear of V1."
        ],
        "contributing_factors": [
            "Sudden stop by V1",
            "Insufficient following distance by V2"
        ],
        "fault_assessment": {
            "likely_responsible_vehicle_id": "V2",
            "likely_responsible_plate": "XYZ-987",
            "confidence": "High",
            "reason": "V2 failed to maintain a safe following distance and rear-ended V1."
        },
        "confidence_assessment": "The system is highly confident in this assessment due to clear visibility of the collision.",
        "final_determination": "V2 is likely responsible for the collision.",
        "evidence_paths": []
    }
    
    json_path = "test_report.json"
    with open(json_path, 'w') as f:
        json.dump(test_data, f, indent=4)
        
    try:
        out_pdf = generate_pdf_from_json(json_path)
        if out_pdf and os.path.exists(out_pdf):
            print(f"✅ Success! PDF generated at: {out_pdf}")
        else:
            print("❌ Failed: Output PDF not found.")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    test_generation()
