"""Local web page for the surveillance action monitor.

Start it from this directory:

    .venv/bin/python web_app.py

Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from behavior_rules import ACTION_LABELS
from monitor import run

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"
JOB_ROOT = ROOT / "runs" / "web"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
QUEUE: queue.Queue[str] = queue.Queue()


def _worker() -> None:
    while True:
        job_id = QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        with LOCK:
            job["status"] = "running"
            job["message"] = "正在识别"
        try:
            _process(job)
        except (Exception, SystemExit) as exc:
            with LOCK:
                job["status"] = "error"
                job["message"] = str(exc) or exc.__class__.__name__


def _process(job: dict) -> None:
    job_dir = Path(job["dir"])
    raw_video = job_dir / "raw.mp4"
    output_video = job_dir / "output.mp4"
    alarm_dir = job_dir / "alarms"

    def on_frame(done: int, total: int, counts: dict) -> None:
        with LOCK:
            job["frames_done"] = done
            job["frames_total"] = total
            job["counts"] = counts
            job["events"] = _events(job)

    options = argparse.Namespace(
        source=job["source"],
        save=str(raw_video),
        alarm_dir=str(alarm_dir),
        pose_model=str(ROOT / "models" / "yolov8n-pose.onnx"),
        cig_model=str(ROOT / "models" / "last.onnx"),
        conf=0.45,
        skeleton=job["skeleton"],
        show=False,
        no_beep=True,
        webhook="",
        max_frames=0,
        preview_path=str(job_dir / "preview.jpg"),
        on_frame=on_frame,
    )
    counts = run(options)
    playable = _to_browser_mp4(raw_video, output_video)
    with LOCK:
        job["counts"] = counts
        job["events"] = _events(job)
        job["playable"] = playable
        job["status"] = "done"
        job["message"] = "识别完成"


def _events(job: dict) -> list[dict]:
    path = Path(job["dir"]) / "alarms" / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = Path(event.get("snapshot", "")).name
        event["snapshot_url"] = f"/jobs/{job['id']}/snapshots/{name}" if name else ""
        events.append(event)
    return events


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _to_browser_mp4(src: Path, dst: Path) -> bool:
    ffmpeg = _ffmpeg()
    if ffmpeg is None or not src.exists():
        if src.exists():
            shutil.copyfile(src, dst)
        return False
    result = subprocess.run(
        [
            ffmpeg, "-y", "-i", str(src),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0 and dst.exists():
        return True
    if src.exists() and not dst.exists():
        shutil.copyfile(src, dst)
    return False


@app.get("/")
def index():
    return send_file(PAGE)


@app.get("/models/<name>")
def model_file(name: str):
    safe_name = Path(name).name
    if safe_name != name or safe_name not in {"yolov8n-pose.onnx", "last.onnx"}:
        abort(404)
    path = ROOT / "models" / safe_name
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="application/octet-stream", max_age=86400)


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        abort(400, "请选择视频文件")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"
    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / f"input{suffix}"
    upload.save(source)
    job = {
        "id": job_id,
        "dir": str(job_dir),
        "source": str(source),
        "status": "queued",
        "message": "排队中",
        "frames_done": 0,
        "frames_total": 0,
        "counts": {name: 0 for name in ACTION_LABELS},
        "events": [],
        "skeleton": request.form.get("skeleton", "1") != "0",
        "playable": False,
    }
    with LOCK:
        JOBS[job_id] = job
    QUEUE.put(job_id)
    return jsonify(_public_job(job))


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    with LOCK:
        if job["status"] == "running":
            job["events"] = _events(job)
        payload = _public_job(job)
    return jsonify(payload)


@app.get("/jobs/<job_id>/preview.jpg")
def preview(job_id: str):
    path = _job_file(job_id, "preview.jpg")
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=0)


@app.get("/jobs/<job_id>/video")
def video(job_id: str):
    path = _job_file(job_id, "output.mp4")
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/jobs/<job_id>/snapshots/<name>")
def snapshot(job_id: str, name: str):
    safe_name = Path(name).name
    if safe_name != name:
        abort(404)
    path = _job_dir(job_id) / "alarms" / "snapshots" / safe_name
    if not path.is_file():
        abort(404)
    return send_from_directory(path.parent, safe_name)


def _job_dir(job_id: str) -> Path:
    if job_id not in JOBS:
        abort(404)
    return JOB_ROOT / job_id


def _job_file(job_id: str, name: str) -> Path:
    return _job_dir(job_id) / name


def _public_job(job: dict) -> dict:
    done = job["status"] == "done"
    return {
        "id": job["id"],
        "status": job["status"],
        "message": job["message"],
        "frames_done": job["frames_done"],
        "frames_total": job["frames_total"],
        "counts": job["counts"],
        "labels": ACTION_LABELS,
        "events": job["events"],
        "playable": job["playable"],
        "preview_url": f"/jobs/{job['id']}/preview.jpg",
        "video_url": f"/jobs/{job['id']}/video" if done else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="监控动作识别网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    print(f"打开 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
