import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "models" / "_weights"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
(WEIGHTS_DIR / "plate_detector").mkdir(parents=True, exist_ok=True)
(WEIGHTS_DIR / "plate_recognizer" / "model").mkdir(parents=True, exist_ok=True)

def download_public_models():
    """Download public models (YOLO base and TrOCR)."""
    print("\n[1/4] Preparing YOLOv8n Base Model...")
    try:
        from ultralytics import YOLO
        # Ultralytics will automatically download to the current dir if not found
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8n is ready.")
    except Exception as e:
        print(f"⚠️ Failed to download YOLOv8n: {e}")

    print("\n[2/4] Preparing TrOCR Plate Recognizer...")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("Downloading microsoft/trocr-base-printed from Hugging Face...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
        save_path = WEIGHTS_DIR / "plate_recognizer" / "model"
        processor.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"✅ TrOCR saved to {save_path}")
    except ImportError:
        print("⚠️ transformers library is not installed. Please install it to download TrOCR.")
    except Exception as e:
        print(f"⚠️ Failed to download TrOCR: {e}")

def handle_custom_model(name, expected_path, env_url_key):
    """Handle custom weights via environment variable URL or provide instructions."""
    print(f"\n[*] Preparing {name}...")
    target_file = BASE_DIR / expected_path
    
    if target_file.exists():
        print(f"✅ {name} already exists at {target_file}")
        return

    url = os.getenv(env_url_key)
    if url:
        print(f"Downloading {name} from {url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, target_file)
            print(f"✅ Successfully downloaded {name}.")
        except Exception as e:
            print(f"⚠️ Failed to download from {url}: {e}")
            print(f"❌ MISSING MODEL: {name}")
            print(f"   Expected path: {expected_path}")
            print(f"   Please download or copy it manually.")
    else:
        print(f"❌ MISSING MODEL: {name}")
        print(f"   Expected path: {expected_path}")
        print(f"   URL not provided in .env ({env_url_key}).")
        print(f"   Please copy this custom weight manually or provide the URL in .env.")

if __name__ == "__main__":
    print("=======================================")
    print(" RoadVision - AI Model Setup Script")
    print("=======================================")
    
    download_public_models()
    
    handle_custom_model(
        name="VideoMAE Accident Detection Model",
        expected_path="models/_weights/checkpoint-best.pth",
        env_url_key="VIDEOMAE_MODEL_URL"
    )
    
    handle_custom_model(
        name="YOLO Plate Detection Model",
        expected_path="models/_weights/plate_detector/best_Koushim_Model.pt",
        env_url_key="PLATE_DETECTION_MODEL_URL"
    )
    
    print("\n=======================================")
    print(" Model setup complete.")
    print("=======================================")
