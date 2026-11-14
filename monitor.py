"""Surveillance video monitor.

Reads a video file, camera index, or stream URL. People are detected with the
bundled YOLOv8-Pose model. Smoking reuses this repo's hand-to-mouth rule and,
when the bundled cigarette model is a small-class detector, confirms it on a
mouth crop. Public urination and defecation are pose rules on the same
skeleton. An action has to hold across frames before the alarm fires.

Example:
    python monitor.py --source /path/to/cctv.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from alarm import AlarmCenter, draw_alarm_banner
from behavior_rules import (
    ACTION_LABELS,
    DEFAULT_THRESHOLDS,
    POSE_ONLY_SMOKING_THRESHOLD,
    classify_pose,
    hand_near_mouth,
)
from pose_runtime import CigaretteDetector, YoloPose, mouth_crop

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
BOX_COLOR = {
    "smoking": (0, 0, 255),
    "urinating": (0, 80, 255),
    "defecating": (0, 140, 255),
}


class Track:
    def __init__(self, track_id: int, box: np.ndarray):
        self.id = track_id
        self.box = box
        self.missing = 0
        self.score = {name: 0 for name in ACTION_LABELS}
        self.latched = {name: False for name in ACTION_LABELS}


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes: list[np.ndarray]) -> list[Track]:
        used = set()
        matched: list[Track] = []
        for box in boxes:
            best_iou = self.iou_threshold
            best = None
            for index, track in enumerate(self.tracks):
                if index in used:
                    continue
                overlap = _iou(box, track.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best = index
            if best is None:
                track = Track(self.next_id, box)
                self.next_id += 1
                self.tracks.append(track)
                used.add(self.tracks.index(track))
                matched.append(track)
            else:
                used.add(best)
                track = self.tracks[best]
                track.box = box
                track.missing = 0
                matched.append(track)
        for index, track in enumerate(self.tracks):
            if index not in used:
                track.missing += 1
        self.tracks = [track for track in self.tracks if track.missing <= self.max_missing]
        return matched


def run(opt: argparse.Namespace) -> dict[str, int]:
    pose = YoloPose(opt.pose_model, conf=opt.conf)
    cigarette, pose_only_smoking = _load_cigarette(opt.cig_model)
    thresholds = dict(DEFAULT_THRESHOLDS)
    if pose_only_smoking:
        thresholds["smoking"] = POSE_ONLY_SMOKING_THRESHOLD
        print("香烟模型不可用，抽烟改为手靠近嘴的姿态判断，报警阈值已提高。")
    else:
        print(f"已加载香烟检测模型: {opt.cig_model}")

    capture = _open_capture(opt.source)
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {opt.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if opt.max_frames and total_frames:
        total_frames = min(total_frames, opt.max_frames)
    elif opt.max_frames:
        total_frames = opt.max_frames
    preview_path = getattr(opt, "preview_path", "") or ""
    on_frame = getattr(opt, "on_frame", None)
    writer = None
    alarms = AlarmCenter(opt.alarm_dir, beep=not opt.no_beep, webhook=opt.webhook or None)
    tracker = IoUTracker()
    counts = {name: 0 for name in ACTION_LABELS}
    frame_index = 0
    last_frame = None
    window = "behavior-monitor"
    show = _open_preview(window) if opt.show else False

    while capture.isOpened():
        if opt.max_frames and frame_index >= opt.max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if writer is None and opt.save:
            height, width = frame.shape[:2]
            Path(opt.save).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                opt.save,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        started = time.perf_counter()
        people = pose.detect(frame)
        tracks = tracker.update([person.box for person in people])
        for person, track in zip(people, tracks):
            cigarette_hit = _cigarette_hit(frame, person, cigarette, pose_only_smoking)
            flags = classify_pose(person.keypoints, cigarette_hit)
            fired = _advance(track, flags, thresholds)
            for action in fired:
                counts[action] += 1
                alarms.raise_alarm(
                    frame,
                    action,
                    track.id,
                    frame_index,
                    frame_index / fps,
                    track.score[action],
                )
            label, color = _box_style(track)
            if label:
                _draw_box(frame, person.box, label, color)
            if opt.skeleton:
                _draw_skeleton(frame, person.keypoints)
        banner = [name for name in ACTION_LABELS if any(track.latched[name] for track in tracks)]
        draw_alarm_banner(frame, banner)
        elapsed = max(time.perf_counter() - started, 1e-6)
        cv2.putText(
            frame,
            f"FPS {1 / elapsed:.1f}",
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(frame)
        last_frame = frame
        if preview_path and frame_index % 3 == 0:
            _write_preview(preview_path, frame)
        if on_frame is not None:
            on_frame(frame_index + 1, total_frames, dict(counts))
        if show:
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    if preview_path and last_frame is not None:
        _write_preview(preview_path, last_frame)
    capture.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(
        "处理完成 "
        f"frames={frame_index} "
        + " ".join(f"{ACTION_LABELS[name]}={counts[name]}" for name in counts)
    )
    if opt.save:
        print(f"标注视频: {opt.save}")
    print(f"报警记录: {alarms.log_path}")
    return counts


def _load_cigarette(path: str):
    model_path = Path(path)
    if not model_path.exists():
        print(f"未找到香烟模型 {path}")
        return None, True
    try:
        return CigaretteDetector(model_path), False
    except Exception as exc:
        print(f"香烟模型未启用: {exc}")
        return None, True


def _cigarette_hit(frame, person, cigarette, pose_only: bool) -> bool:
    if pose_only or cigarette is None:
        return hand_near_mouth(person.keypoints)
    if not hand_near_mouth(person.keypoints):
        return False
    crop, _offset = mouth_crop(frame, person.box, person.keypoints)
    return len(cigarette.detect(crop)) > 0


def _advance(track: Track, flags: dict[str, bool], thresholds: dict[str, int]) -> list[str]:
    fired = []
    for name, hit in flags.items():
        if hit:
            track.score[name] = min(40, track.score[name] + 1)
        else:
            track.score[name] = max(0, track.score[name] - 1)
        threshold = thresholds[name]
        if track.score[name] >= threshold and not track.latched[name]:
            track.latched[name] = True
            fired.append(name)
        elif track.score[name] < threshold * 0.4:
            track.latched[name] = False
    return fired


def _action_of(track: Track) -> str:
    return max(track.score, key=track.score.get)


def _box_style(track: Track) -> tuple[str, tuple[int, int, int]]:
    name = _action_of(track)
    if track.score[name] <= 0:
        return "", (0, 0, 0)
    if track.latched[name]:
        return name, BOX_COLOR[name]
    return f"suspected {name}", (0, 165, 255)


def _draw_box(frame, box, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_skeleton(frame, keypoints: np.ndarray) -> None:
    for a, b in SKELETON:
        if keypoints[a, 2] < 0.4 or keypoints[b, 2] < 0.4:
            continue
        p1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        p2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
        cv2.line(frame, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
    for index, (x, y, conf) in enumerate(keypoints):
        if conf < 0.4:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def _write_preview(path: str, frame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.jpg")
    if cv2.imwrite(str(temporary), frame):
        temporary.replace(target)


def _open_preview(window: str) -> bool:
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(f"无法打开预览窗口（{exc}）。将只保存标注视频。")
        print("若需要窗口，请安装带界面的 OpenCV：pip install opencv-python")
        return False
    return True


def _open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控视频动作识别与报警")
    parser.add_argument("--source", required=True, help="视频文件、摄像头编号或 RTSP/HTTP 地址")
    parser.add_argument("--save", default="runs/alarm_output.mp4", help="标注结果视频，空字符串则不保存")
    parser.add_argument("--alarm-dir", default="alarms", help="报警日志和截图目录")
    parser.add_argument("--pose-model", default="models/yolov8n-pose.onnx")
    parser.add_argument("--cig-model", default="models/last.onnx")
    parser.add_argument("--conf", type=float, default=0.45, help="人体检测置信度")
    parser.add_argument("--skeleton", action="store_true", help="画出骨架")
    parser.add_argument("--show", action="store_true", help="打开预览窗口，按 q 退出")
    parser.add_argument("--no-beep", action="store_true", help="关闭声音报警")
    parser.add_argument("--webhook", default="", help="报警时 POST JSON 的地址")
    parser.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧，0 表示全部")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
