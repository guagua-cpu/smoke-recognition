"""ONNX Runtime pose and cigarette inference for the bundled YOLOv8 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - import guard for a clear message
    raise ImportError(
        "缺少 onnxruntime。请先安装：pip install -r requirements-monitor.txt"
    ) from exc


@dataclass
class PersonPose:
    box: np.ndarray  # xyxy
    score: float
    keypoints: np.ndarray  # (17, 3)


class YoloPose:
    def __init__(self, model_path: str | Path, conf: float = 0.45, iou: float = 0.5):
        self.conf = conf
        self.iou = iou
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        output = self.session.get_outputs()[0]
        shape = output.shape
        channels = _channel_count(shape)
        if channels is not None and channels < 56:
            raise ValueError(f"{model_path} 不是 YOLOv8-Pose 模型，输出通道 {channels}")

    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        blob, scale = _topleft_blob(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        preds = _as_rows(raw)
        if preds.size == 0 or preds.shape[1] < 56:
            return []
        scores = preds[:, 4]
        keep = scores >= self.conf
        preds = preds[keep]
        scores = scores[keep]
        if len(scores) == 0:
            return []
        boxes_xywh = _cxcywh_to_xywh(preds[:, :4])
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), self.conf, self.iou
        )
        indices = _nms_indices(indices)
        people: list[PersonPose] = []
        for index in indices:
            row = preds[index]
            xyxy = _xywh_to_xyxy(_cxcywh_to_xywh(row[:4])) * scale
            kpts = row[5:56].reshape(17, 3).copy()
            if float(np.max(kpts[:, :2])) <= 2.0:
                kpts[:, :2] *= 640.0
            kpts[:, :2] *= scale
            people.append(PersonPose(box=xyxy, score=float(scores[index]), keypoints=kpts))
        return people


class CigaretteDetector:
    """Specialized YOLO detector. COCO-80 models are rejected."""

    def __init__(self, model_path: str | Path, conf: float = 0.35, iou: float = 0.5):
        self.conf = conf
        self.iou = iou
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_outputs()[0].shape
        channels = _channel_count(shape)
        if channels is None or channels < 5 or channels > 9:
            raise ValueError(
                f"{model_path} 不是专用小类检测模型（输出通道 {channels}），已跳过香烟检测"
            )
        self.channels = channels

    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if image.size == 0 or image.shape[0] < 4 or image.shape[1] < 4:
            return []
        blob, scale = _topleft_blob(image)
        raw = self.session.run(None, {self.input_name: blob})[0]
        preds = _as_rows(raw)
        if preds.size == 0 or preds.shape[1] < 5:
            return []
        class_scores = preds[:, 4:]
        scores = class_scores.max(axis=1)
        keep = scores >= self.conf
        preds = preds[keep]
        scores = scores[keep]
        if len(scores) == 0:
            return []
        boxes_xywh = _cxcywh_to_xywh(preds[:, :4])
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), self.conf, self.iou
        )
        found = []
        for index in _nms_indices(indices):
            xyxy = _xywh_to_xyxy(_cxcywh_to_xywh(preds[index, :4])) * scale
            found.append((xyxy, float(scores[index])))
        return found


def mouth_crop(frame: np.ndarray, box: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop around the nose. Returns the crop and its top-left offset."""
    height, width = frame.shape[:2]
    nose_x, nose_y = float(keypoints[0, 0]), float(keypoints[0, 1])
    box_w = max(float(box[2] - box[0]), 1.0)
    box_h = max(float(box[3] - box[1]), 1.0)
    half_h = int(0.4 * box_w)
    half_w = int(0.3 * box_h)
    x1 = max(int(nose_x) - half_w, 0)
    y1 = max(int(nose_y) - half_h, 0)
    x2 = min(int(nose_x) + half_w, width)
    y2 = min(int(nose_y) + half_h, height)
    return frame[y1:y2, x1:x2], np.array([x1, y1], dtype=np.float32)


def _session(model_path: str | Path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _topleft_blob(frame: np.ndarray, size: int = 640) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    length = max(height, width)
    canvas = np.zeros((length, length, 3), np.uint8)
    canvas[0:height, 0:width] = frame
    blob = cv2.dnn.blobFromImage(
        canvas, scalefactor=1 / 255.0, size=(size, size), swapRB=True
    )
    return blob, length / float(size)


def _as_rows(raw: np.ndarray) -> np.ndarray:
    preds = np.squeeze(raw)
    if preds.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    # YOLOv8 exports (channels, anchors). Rows are anchors.
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    return preds


def _channel_count(shape) -> int | None:
    # Batch is often 1, which is smaller than the channel count. Ignore it.
    dims = [d for d in shape if isinstance(d, int) and d > 1]
    if not dims:
        return None
    return min(dims)


def _cxcywh_to_xywh(box: np.ndarray) -> np.ndarray:
    out = np.array(box, dtype=np.float32, copy=True)
    out[..., 0] = box[..., 0] - box[..., 2] / 2
    out[..., 1] = box[..., 1] - box[..., 3] / 2
    return out


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    out = np.array(box, dtype=np.float32, copy=True)
    out[..., 2] = box[..., 0] + box[..., 2]
    out[..., 3] = box[..., 1] + box[..., 3]
    return out


def _nms_indices(indices) -> list[int]:
    if indices is None:
        return []
    array = np.array(indices).reshape(-1)
    return [int(i) for i in array]
