# Excluded Files and Folders

The following categories were intentionally excluded from this submission package to ensure a lightweight footprint:

1. **Virtual Environments (`venv`, `venv_new`, `.venv`)**: Excluded to avoid massive redundant libraries and cross-platform compatibility issues. A `requirements.txt` is provided for fresh installation.
2. **AI Model Weights (`models/_weights/`)**: The >2 GB model files (YOLO, VideoMAE, TrOCR) were excluded. A `download_models.py` script is provided to fetch or guide the setup of these models.
3. **Cache and IDE Folders (`__pycache__`, `.idea`, `.vscode`, `.DS_Store`)**: Standard exclusions to keep the directory clean.
4. **Outputs and Reports**: Past JSON/PDF accident reports and CSV session summaries were excluded to provide a clean state.
5. **Media and Datasets**: Large test `.mp4` videos and raw image assets were excluded to keep the ZIP size under 512 MB.
6. **Secrets (`.env`)**: Real API keys were removed to protect sensitive data. An `.env.example` is provided.
7. **Debugging Scripts**: Temporary test and debug scripts were removed.
