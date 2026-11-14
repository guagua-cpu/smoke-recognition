"""Alarm side effects: banner state, snapshot, json log, and a system beep."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import wave
from datetime import datetime
from pathlib import Path

import cv2

from behavior_rules import ACTION_LABELS


class AlarmCenter:
    def __init__(self, alarm_dir: str | Path, beep: bool = True, webhook: str | None = None):
        self.alarm_dir = Path(alarm_dir)
        self.snapshot_dir = self.alarm_dir / "snapshots"
        self.alarm_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.alarm_dir / "events.jsonl"
        self.beep_enabled = beep
        self.webhook = webhook
        self.beep_path = self.alarm_dir / "alarm.wav"
        self._last_beep = 0.0
        if beep:
            _write_beep(self.beep_path)

    def raise_alarm(
        self,
        frame,
        action: str,
        track_id: int,
        frame_index: int,
        video_time: float,
        score: float,
    ) -> dict:
        label = ACTION_LABELS.get(action, action)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot = self.snapshot_dir / f"{stamp}_{action}_id{track_id}.jpg"
        cv2.imwrite(str(snapshot), frame)
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "video_time": round(video_time, 2),
            "frame": frame_index,
            "track_id": track_id,
            "action": action,
            "label": label,
            "score": round(score, 2),
            "snapshot": str(snapshot),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[报警] {label}  track={track_id}  t={video_time:.1f}s  {snapshot}")
        self._beep()
        self._notify(label, track_id)
        self._post(event)
        return event

    def _beep(self) -> None:
        if not self.beep_enabled or not self.beep_path.exists():
            return
        now = datetime.now().timestamp()
        if now - self._last_beep < 2.5:
            return
        self._last_beep = now
        for player in ("paplay", "aplay"):
            try:
                subprocess.Popen(
                    [player, str(self.beep_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except FileNotFoundError:
                continue

    def _notify(self, label: str, track_id: int) -> None:
        try:
            subprocess.Popen(
                ["notify-send", "监控报警", f"检测到{label}（目标 {track_id}）"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return

    def _post(self, event: dict) -> None:
        if not self.webhook:
            return
        try:
            import urllib.request

            data = json.dumps(event, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                self.webhook,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=2).read()
        except Exception as exc:
            print(f"[报警] webhook 发送失败: {exc}")


def draw_alarm_banner(frame, active_labels: list[str]) -> None:
    if not active_labels:
        return
    text = "ALARM: " + " / ".join(active_labels)
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 48), (0, 0, 220), -1)
    cv2.putText(
        frame,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _write_beep(path: Path) -> None:
    framerate = 22050
    seconds = 0.45
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(framerate)
        frames = bytearray()
        for i in range(int(framerate * seconds)):
            env = 1.0 if i < framerate * 0.35 else (1 - (i - framerate * 0.35) / (framerate * 0.1))
            env = max(0.0, env)
            sample = int(12000 * env * math.sin(2 * math.pi * 880 * i / framerate))
            frames += struct.pack("<h", sample)
        handle.writeframes(frames)
