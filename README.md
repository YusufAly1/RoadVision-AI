# RoadVision AI - Traffic Intelligence System


DEMO

<video src="https://github.com/YusufAly1/RoadVision-AI/releases/download/v1.0/DEMO.mp4" controls width="100%"></video>
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

## 📖 Project Overview

RoadVision AI is a comprehensive, Real-Time Traffic Intelligence System designed to monitor roads, detect accidents, recognize license plates, and generate detailed traffic reports. Built with state-of-the-art Computer Vision models, this system provides a robust solution for traffic management and incident response by utilizing a highly optimized Master-Slave threading architecture.

## ✨ Features

- **Real-Time Object Tracking**: Tracks vehicles using advanced YOLOv8 models.
- **Accident Detection**: Accurately detects accidents in real-time utilizing VideoMAE Transformer models.
- **Automatic License Plate Recognition (ALPR)**: Extracts license plates via EasyOCR for incident reporting.
- **Speed & Anomaly Detection**: Calibrates speed and detects anomalies using Bird's-Eye View (BEV) mapping.
- **PyQt6 Dashboard**: A sleek, non-blocking user interface for real-time monitoring and alert visualization.
- **Automated Reporting**: Generates comprehensive PDF, JSON, and CSV reports for traffic sessions and incidents.
- **AI Verification**: Integrates with Gemini AI for intelligent incident verification.

## 🛠 Technologies Used

- **Programming Language**: Python 3.9+
- **Computer Vision & Deep Learning**: 
  - OpenCV
  - PyTorch & Torchvision
  - Ultralytics (YOLOv8)
  - Transformers & TIMM (VideoMAE)
- **Optical Character Recognition**: EasyOCR
- **Generative AI**: Google Generative AI (Gemini)
- **User Interface**: PyQt6, PyQtGraph
- **Data & Reporting**: Pandas, NumPy, ReportLab, Matplotlib

## ⚙️ Installation Steps

1. **Clone the Repository** (If you haven't already)
   ```bash
   git clone https://github.com/YusufAly1/RoadVision-AI.git
   cd RoadVision-AI
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI Models**
   The application requires several pre-trained models. Run the following script to prepare them:
   ```bash
   python download_models.py
   ```
   *Note: For the custom VideoMAE and YOLO Plate Detection models, either add their URLs to your `.env` file before running the script, or manually place the `.pth` and `.pt` files as instructed by the script.*

5. **Configuration**
   Copy the `.env.example` to `.env` and fill in your Gemini API key for incident verification:
   ```bash
   cp .env.example .env
   ```

6. **Provide a Test Video**
   Place your test `.mp4` video inside the `videos/` folder.

## 🚀 Usage

Ensure your virtual environment is activated, then run the main entry point to launch the application dashboard:

```bash
python main.py
```

Once the UI launches, you can connect your video source (or select the test video), and the system will begin monitoring the feed for vehicles, speeds, and accidents.

## 🔮 Future Improvements

- **Multi-Camera Support**: Expand the system to handle multiple concurrent video streams for comprehensive intersection coverage.
- **Cloud Integration**: Implement cloud-based processing and storage for historical traffic data analysis and remote dashboard access.
- **Enhanced Vehicle Re-Identification**: Improve tracking consistency across different camera angles and environmental conditions.
- **Edge Deployment**: Optimize the models further (e.g., via TensorRT or ONNX) for deployment on edge devices like NVIDIA Jetson.
- **Night Vision Mode**: Train models to better perform under low-light and adverse weather conditions.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
