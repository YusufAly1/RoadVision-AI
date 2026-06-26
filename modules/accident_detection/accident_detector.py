"""
VideoMAE Accident Detector Module

Encapsulates VideoMAE logic for predicting accident probabilities in video windows.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List
from torchvision import transforms

try:
    from modeling_finetune import vit_base_patch16_224
except ImportError:
    # Fallback if VideoMAE is cloned locally instead of pip installed
    sys.path.append(os.path.join(os.path.dirname(__file__), 'VideoMAE'))
    try:
        from modeling_finetune import vit_base_patch16_224
    except ImportError:
        raise ImportError("Could not import vit_base_patch16_224. Ensure VideoMAE is installed or cloned.")

from .video_helpers import read_video_frames, get_video_info


class VideoMAEAccidentDetector:
    """
    Detector class using VideoMAE to identify potential accidents in video streams.
    """
    def __init__(self, checkpoint_path: str, num_classes: int = 2, 
                 num_frames: int = 16, input_size: int = 224, 
                 accident_class_index: int = 1):
        """
        Initializes the VideoMAE model.

        Args:
            checkpoint_path (str): Path to the trained checkpoint.
            num_classes (int): Number of classes.
            num_frames (int): Number of frames the model expects.
            input_size (int): Image size for preprocessing.
            accident_class_index (int): Index representing the 'accident' class.
        """
        self.num_frames = num_frames
        self.input_size = input_size
        self.accident_class_index = accident_class_index
        
        self.device = self._get_device()
        self.model = self._build_videomae_model(checkpoint_path, num_classes)
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _get_device(self) -> str:
        """Determines the optimal device for execution."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _build_videomae_model(self, checkpoint_path: str, num_classes: int):
        """
        Builds the VideoMAE model and loads the checkpoint.
        """
        model = vit_base_patch16_224(
            pretrained=False,
            num_classes=num_classes,
            all_frames=self.num_frames,
            tubelet_size=2,
            use_mean_pooling=True,
        )

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))

        clean_state = {}
        for k, v in state.items():
            k = k.replace('module.', '')
            k = k.replace('backbone.', '').replace('encoder.', '')
            clean_state[k] = v

        model.load_state_dict(clean_state, strict=False)
        model.to(self.device).eval()
        return model

    def preprocess_frames_for_videomae(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses raw frames into a tensor suitable for VideoMAE.

        Args:
            frames (List[np.ndarray]): List of raw RGB frames.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (B, C, T, H, W).
        """
        processed = []
        for frame in frames:
            img = Image.fromarray(frame).convert('RGB').resize((self.input_size, self.input_size))
            arr = torch.from_numpy(np.array(img)).float() / 255.0
            arr = arr.permute(2, 0, 1)  # C,H,W
            arr = self.normalize(arr)
            processed.append(arr)

        video = torch.stack(processed, dim=1)  # C,T,H,W
        video = video.unsqueeze(0).to(self.device)  # B,C,T,H,W
        return video

    @torch.no_grad()
    def predict_videomae_window(self, video_path: str, start_sec: float, end_sec: float) -> Dict:
        """
        Predicts accident probability for a specific video window.

        Args:
            video_path (str): Path to the video file.
            start_sec (float): Start time of the window.
            end_sec (float): End time of the window.

        Returns:
            Dict: Dictionary containing prediction details.
        """
        frames, frame_indices = read_video_frames(video_path, start_sec, end_sec, self.num_frames)
        tensor = self.preprocess_frames_for_videomae(frames)
        logits = self.model(tensor)
        
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
            
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        
        return {
            'start_sec': float(start_sec),
            'end_sec': float(end_sec),
            'middle_sec': float((start_sec + end_sec) / 2),
            'prediction': 'Accident' if pred == self.accident_class_index else 'Normal',
            'normal_confidence': float(probs[0]) if len(probs) > 0 else None,
            'accident_confidence': float(probs[self.accident_class_index]) if len(probs) > self.accident_class_index else None,
            'frame_indices': frame_indices
        }

    def scan_video_with_videomae(self, video_path: str, window_seconds: float = 5.0, step_seconds: float = 1.0) -> pd.DataFrame:
        """
        Scans an entire video using a sliding window approach.

        Args:
            video_path (str): Path to the video file.
            window_seconds (float): Size of the sliding window in seconds.
            step_seconds (float): Step size in seconds.

        Returns:
            pd.DataFrame: DataFrame containing prediction results for all windows.
        """
        info = get_video_info(video_path)
        rows = []
        t = 0.0
        while t < info['duration']:
            start = t
            end = min(t + window_seconds, info['duration'])
            if end - start < max(1.0, window_seconds * 0.4):
                break
            try:
                rows.append(self.predict_videomae_window(video_path, start, end))
            except Exception as e:
                print(f"Window failed {start:.2f}-{end:.2f}: {e}")
            t += step_seconds
            
        return pd.DataFrame(rows)

    @torch.no_grad()
    def scan_buffer_with_videomae(self, frames: List[np.ndarray], step_frames: int = 5) -> Dict:
        """
        Scans an in-memory buffer of frames using a sliding window.
        Finds the 16-frame window with the highest accident confidence.
        
        Args:
            frames: List of raw RGB/BGR frames spanning the event window.
            step_frames: Number of frames to step the sliding window.
            
        Returns:
            Dict with peak confidence and the exact frame indices of the peak window.
        """
        best_conf = -1.0
        best_pred = 'Normal'
        best_window = []
        best_start_idx = 0
        
        # Slide window
        for i in range(0, max(1, len(frames) - self.num_frames + 1), step_frames):
            end_idx = min(i + self.num_frames, len(frames))
            window = frames[i:end_idx]
            
            # If we don't have enough frames, pad with the last frame
            while len(window) < self.num_frames:
                window.append(window[-1])
                
            tensor = self.preprocess_frames_for_videomae(window)
            logits = self.model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
                
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(probs[self.accident_class_index]) if len(probs) > self.accident_class_index else 0.0
            
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[ACCIDENT_DEBUG] window={i}-{end_idx} logits={logits.detach().cpu().numpy().tolist()} probs={probs.tolist()} accident_confidence={conf:.4f}")
            
            if conf > best_conf:
                best_conf = conf
                best_pred = 'Accident' if pred == self.accident_class_index else 'Normal'
                best_window = window
                best_start_idx = i
                
        return {
            'prediction': best_pred,
            'accident_confidence': best_conf,
            'peak_start_idx': best_start_idx,
            'peak_end_idx': best_start_idx + len(best_window) - 1,
            'peak_window_frames': best_window
        }
