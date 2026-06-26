"""
Video Helpers Module

Contains functions for video I/O, info extraction, and evidence frame extraction.
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from decord import VideoReader, cpu


def get_video_info(video_path: str) -> Dict:
    """
    Extracts basic information from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Dict: A dictionary containing fps, total_frames, width, height, and duration.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps else 0.0

    except Exception as e:
        raise RuntimeError(f"Error reading video info for {video_path}: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    return {
        'fps': float(fps),
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': float(duration)
    }


def read_video_frames(video_path: str, start_sec: float, end_sec: float, num_frames: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Reads a specific segment of a video and extracts a fixed number of evenly spaced frames.

    Args:
        video_path (str): Path to the video file.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
        num_frames (int): Number of frames to extract.

    Returns:
        Tuple[List[np.ndarray], List[int]]: A tuple containing the list of frames (as numpy arrays)
                                            and their corresponding frame indices.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"Failed to initialize VideoReader for {video_path}: {e}")

    try:
        fps = vr.get_avg_fps()
        total = len(vr)

        start_idx = max(0, int(start_sec * fps))
        end_idx = min(total - 1, int(end_sec * fps))
        if end_idx <= start_idx:
            end_idx = min(total - 1, start_idx + num_frames)

        frame_indices = np.linspace(start_idx, end_idx, num_frames).astype(int).tolist()
        frames = vr.get_batch(frame_indices).asnumpy()  # RGB format
    except Exception as e:
        raise RuntimeError(f"Error reading frames from {video_path}: {e}")

    return [f for f in frames], frame_indices


def union_box(boxes: List[List[float]], pad: int = 80, frame_shape: Tuple[int, int, int] = None) -> Tuple[int, int, int, int]:
    """
    Calculates the union bounding box that encompasses all provided boxes.

    Args:
        boxes (List[List[float]]): List of bounding boxes [x1, y1, x2, y2].
        pad (int): Padding to add around the union box.
        frame_shape (Tuple[int, int, int], optional): The shape of the frame (height, width, channels) to bound the box.

    Returns:
        Tuple[int, int, int, int]: The union bounding box coordinates (x1, y1, x2, y2).
    """
    boxes = np.array(boxes, dtype=float)
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()

    if frame_shape is not None:
        h, w = frame_shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

    return int(x1), int(y1), int(x2), int(y2)


def extract_evidence_frames(video_path: str, impact_sec: float, output_dir: str, 
                            yolo_model, vehicle_classes: List[int], yolo_conf: float,
                            seconds_offsets: List[float] = None, crop: bool = True) -> List[str]:
    """
    Extracts frames around the impact moment, annotates them with vehicle bounding boxes,
    and optionally crops the frames around the vehicles.

    Args:
        video_path (str): Path to the video file.
        impact_sec (float): Detected impact moment in seconds.
        output_dir (str): Directory to save the extracted frames.
        yolo_model: The loaded YOLO model for object detection.
        vehicle_classes (List[int]): List of class IDs representing vehicles.
        yolo_conf (float): Confidence threshold for YOLO predictions.
        seconds_offsets (List[float], optional): List of time offsets relative to impact_sec to extract.
        crop (bool): Whether to crop the frame around the detected vehicles.

    Returns:
        List[str]: A list of file paths to the saved evidence images.
    """
    if seconds_offsets is None:
        seconds_offsets = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    os.makedirs(output_dir, exist_ok=True)
    info = get_video_info(video_path)
    fps = info['fps']
    paths = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        for off in seconds_offsets:
            sec = min(max(impact_sec + off, 0), info['duration'])
            frame_id = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            save_frame = frame.copy()
            results = yolo_model.predict(frame, classes=vehicle_classes, conf=yolo_conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

            # Draw bounding boxes
            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop if multiple vehicles are detected
            if crop and len(boxes) >= 2:
                x1, y1, x2, y2 = union_box(boxes, pad=100, frame_shape=save_frame.shape)
                save_frame = save_frame[y1:y2, x1:x2]

            out_path = os.path.join(output_dir, f'evidence_t_{sec:.2f}_frame_{frame_id}.jpg')
            cv2.imwrite(out_path, save_frame)
            paths.append(out_path)

    except Exception as e:
        raise RuntimeError(f"Error during evidence extraction for {video_path}: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    return paths
