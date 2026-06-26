"""
LLM Verifier Module

Contains logic for querying Google's Gemini to verify accident occurrences.
"""
import os
import json
import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def sanitize_data(obj):
    import numpy as np
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

def gemini_gatekeeper(prepared_frames: List[Dict], telemetry: Dict, use_gemini: bool = True) -> Dict:
    """
    Queries Gemini 2.5 Pro with structured prompts, tracking dynamics, and OpenAPI schema.
    Uses Tenacity for safe retries and handles timeouts.
    """
    if not use_gemini:
        return {'gemini_status': 'skipped', 'accident_confirmed': False, 'reason': 'Gemini disabled'}

    import google.generativeai as genai
    from dotenv import load_dotenv
    from utils.constants import (
        GEMINI_REPORT_MODEL, GEMINI_REPORT_TEMPERATURE,
        GEMINI_REPORT_MAX_OUTPUT_TOKENS, GEMINI_REPORT_TIMEOUT_SECONDS,
        GEMINI_REPORT_MAX_ATTEMPTS, GEMINI_REPORT_TOTAL_DEADLINE
    )
    
    load_dotenv(override=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the project .env file."
        )

    genai.configure(api_key=api_key, transport="rest")
    
    # Define exact OpenAPI Dictionary Schema for 0.8.6
    openapi_schema = {
        "type": "OBJECT",
        "properties": {
            "accident_confirmed": {"type": "BOOLEAN"},
            "accident_confidence": {"type": "NUMBER"},
            "impact_frame_id": {"type": "STRING"},
            "vehicles_involved": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "track_id": {"type": "STRING"},
                        "role": {"type": "STRING"},
                        "pre_impact_behavior": {"type": "STRING"},
                        "estimated_speed_kmh": {"type": "NUMBER"},
                        "detected_violations": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                }
            },
            "sequence_of_events": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "order": {"type": "INTEGER"},
                        "frame_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "description": {"type": "STRING"}
                    }
                }
            },
            "likely_at_fault": {
                "type": "OBJECT",
                "properties": {
                    "track_id": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                    "decision": {"type": "STRING"},
                    "reasoning": {"type": "STRING"},
                    "supporting_frame_ids": {"type": "ARRAY", "items": {"type": "STRING"}}
                }
            },
            "contributing_factors": {"type": "ARRAY", "items": {"type": "STRING"}},
            "uncertainties": {"type": "ARRAY", "items": {"type": "STRING"}},
            "evidence_frame_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
            "summary": {"type": "STRING"}
        },
        "required": [
            "accident_confirmed", "accident_confidence", "impact_frame_id", 
            "vehicles_involved", "sequence_of_events", "likely_at_fault",
            "evidence_frame_ids", "summary"
        ]
    }

    try:
        model = genai.GenerativeModel(GEMINI_REPORT_MODEL)
    except Exception as e:
        return {'gemini_status': 'failed', 'accident_confirmed': False, 'reason': f'Model initialization error: {e}'}

    # Interleave text and images
    from PIL import Image
    contents = []
    
    prompt_intro = (
        "You are RoadVision AI Accident Investigation Assistant.\n"
        "Analyze the provided frames in chronological order. Each frame is interleaved with its exact metadata.\n"
        "Also analyze the following vehicle telemetry data for vehicles involved or nearby:\n"
    )
    
    clean_telemetry = sanitize_data(telemetry)
    telemetry_json_str = json.dumps(clean_telemetry, indent=2)
    prompt_intro += telemetry_json_str + "\n\n"
    
    prompt_intro += (
        "Rules:\n"
        "- NEVER invent facts not supported by the evidence.\n"
        "- Distinguish direct visual evidence from telemetry.\n"
        "- If evidence is insufficient to determine fault, return 'insufficient_evidence' in likely_at_fault.decision.\n"
        "- Do not determine fault using vehicle type, color, or license plate.\n"
        "- Prefer traffic behavior such as direction, lane movement, right-of-way, speed, braking, wrong-way movement, and collision sequence.\n"
        "- Allowed values for likely_at_fault.decision: 'likely_at_fault', 'shared_fault', 'insufficient_evidence', 'no_accident'.\n"
        "- evidence_frame_ids must contain between 3 and 6 unique valid submitted frame IDs when an accident is confirmed. (0 if no accident).\n"
    )
    
    contents.append(prompt_intro)
    
    valid_frame_ids = []
    for meta in prepared_frames:
        frame_id = meta["analysis_frame_id"]
        valid_frame_ids.append(frame_id)
        
        frame_text = (
            f"Frame {frame_id}\n"
            f"Original frame: {meta['video_frame_number']}\n"
            f"Timestamp: {meta['timestamp_seconds']} seconds\n"
            f"Relative to impact: {meta['relative_to_impact_seconds']} seconds\n"
            f"Category: {meta['category']}\n"
            f"Selection reason: {meta['selection_reason']}\n"
        )
        contents.append(frame_text)
        
        try:
            p = meta["analysis_path"]
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                contents.append(img)
        except Exception as e:
            logger.warning(f"Failed to load image {p}: {e}")

    gen_config = genai.types.GenerationConfig(
        temperature=GEMINI_REPORT_TEMPERATURE,
        max_output_tokens=GEMINI_REPORT_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=openapi_schema
    )

    # Retry Logic
    start_time = time.time()
    attempts = 0
    last_error = None
    
    while attempts < GEMINI_REPORT_MAX_ATTEMPTS:
        if time.time() - start_time > GEMINI_REPORT_TOTAL_DEADLINE:
            break
            
        attempts += 1
        try:
            response = model.generate_content(
                contents,
                generation_config=gen_config,
                request_options={"timeout": GEMINI_REPORT_TIMEOUT_SECONDS}
            )
            
            res_dict = json.loads(response.text)
            
            # Local validation of evidence IDs
            ev_ids = res_dict.get("evidence_frame_ids", [])
            valid_ev_ids = []
            for fid in ev_ids:
                if fid in valid_frame_ids and fid not in valid_ev_ids:
                    valid_ev_ids.append(fid)
            
            if res_dict.get("accident_confirmed"):
                res_dict["evidence_frame_ids"] = valid_ev_ids[:6]
            else:
                res_dict["evidence_frame_ids"] = []
                
            res_dict["gemini_status"] = "Success"
            return res_dict
            
        except Exception as e:
            err_str = str(e).lower()
            last_error = str(e)
            logger.error(f"Gemini API Error: {last_error}"); open("last_gemini_error.txt", "w").write(last_error)
            # Only retry on specific temporary errors
            if any(x in err_str for x in ['429', '500', '502', '503', '504', 'timeout']):
                time.sleep(1) # short exponential backoff in theory, just 1s here
                continue
            else:
                # Break on validation, auth, formatting errors
                break

    return {
        'gemini_status': 'failed', 
        'accident_confirmed': False, 
        'reason': f'API Error or Timeout after {attempts} attempts. Last error: {last_error}'
    }
