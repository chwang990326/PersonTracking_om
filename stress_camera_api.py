"""
Camera-stream stress test client for POST /api/v1/person/detect.

The client uses local videos as camera sources, samples frames at a target FPS,
posts them to the API, and increases the number of simulated camera streams
until the average successful processing speed drops below the threshold.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_API_URL = "http://127.0.0.1:8130/api/v1/person/detect"
DEFAULT_VIDEO_DIR = "video"
DEFAULT_CAMERA_IDS = "207"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".flv"}


@dataclass(frozen=True)
class EncodedVideo:
    path: Path
    fps: float
    frame_step: int
    frames: List[str]


@dataclass
class CameraMetrics:
    sent: int = 0
    success: int = 0
    failed: int = 0
    latencies_ms: List[float] = field(default_factory=list)


@dataclass
class StageMetrics:
    camera_count: int
    target_fps: float
    duration_seconds: float
    sent: int = 0
    success: int = 0
    failed: int = 0
    http_errors: int = 0
    api_errors: int = 0
    request_errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    per_camera: Dict[str, CameraMetrics] = field(default_factory=dict)

    @property
    def success_fps_total(self) -> float:
        return self.success / self.duration_seconds if self.duration_seconds > 0 else 0.0

    @property
    def success_fps_per_camera(self) -> float:
        if self.camera_count <= 0:
            return 0.0
        return self.success_fps_total / self.camera_count

    @property
    def success_rate(self) -> float:
        total = self.success + self.failed
        return self.success / total if total else 0.0

    @property
    def min_camera_fps(self) -> float:
        if not self.per_camera:
            return 0.0
        return min(item.success / self.duration_seconds for item in self.per_camera.values())

    @property
    def p50_latency_ms(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return percentile(self.latencies_ms, 95)


class MetricsRecorder:
    def __init__(self, camera_count: int, target_fps: float) -> None:
        self._lock = threading.Lock()
        self._collecting = False
        self._duration_seconds = 0.0
        self.metrics = StageMetrics(
            camera_count=camera_count,
            target_fps=target_fps,
            duration_seconds=0.0,
        )

    def start(self) -> None:
        with self._lock:
            self._collecting = True
            self._duration_seconds = 0.0
            self.metrics.sent = 0
            self.metrics.success = 0
            self.metrics.failed = 0
            self.metrics.http_errors = 0
            self.metrics.api_errors = 0
            self.metrics.request_errors = 0
            self.metrics.latencies_ms.clear()
            self.metrics.per_camera.clear()

    def finish(self, duration_seconds: float) -> StageMetrics:
        with self._lock:
            self._collecting = False
            self._duration_seconds = duration_seconds
            self.metrics.duration_seconds = duration_seconds
            return self.metrics

    def record(
        self,
        camera_name: str,
        sent: bool,
        success: bool,
        latency_ms: Optional[float],
        error_kind: Optional[str],
    ) -> None:
        with self._lock:
            if not self._collecting:
                return

            camera = self.metrics.per_camera.setdefault(camera_name, CameraMetrics())
            if sent:
                self.metrics.sent += 1
                camera.sent += 1

            if success:
                self.metrics.success += 1
                camera.success += 1
                if latency_ms is not None:
                    self.metrics.latencies_ms.append(latency_ms)
                    camera.latencies_ms.append(latency_ms)
                return

            self.metrics.failed += 1
            camera.failed += 1
            if error_kind == "http":
                self.metrics.http_errors += 1
            elif error_kind == "api":
                self.metrics.api_errors += 1
            elif error_kind == "request":
                self.metrics.request_errors += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Increase simulated camera streams until average successful FPS per "
            "camera is lower than the threshold."
        )
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Person detect API URL")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory containing source videos")
    parser.add_argument(
        "--video-path",
        action="append",
        default=[],
        help="Specific source video path. Can be passed multiple times. Defaults to all videos in --video-dir.",
    )
    parser.add_argument(
        "--camera-ids",
        default=DEFAULT_CAMERA_IDS,
        help="Comma-separated camera IDs. IDs are reused when camera count is larger than this list.",
    )
    parser.add_argument("--strict-camera-ids", action="store_true", help="Fail if there are fewer camera IDs than streams")
    parser.add_argument("--start-cameras", type=int, default=1, help="First camera count to test")
    parser.add_argument("--max-cameras", type=int, default=64, help="Maximum camera count to test")
    parser.add_argument("--step-cameras", type=int, default=1, help="Camera count increment per stage")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Send target FPS for each camera")
    parser.add_argument("--min-fps", type=float, default=5.0, help="Stop when average successful FPS per camera is below this")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.99,
        help="Stop when successful response rate is below this ratio",
    )
    parser.add_argument("--warmup-seconds", type=float, default=10.0, help="Warmup time before each measured stage")
    parser.add_argument("--duration-seconds", type=float, default=60.0, help="Measured time per stage")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument("--frame-step", type=int, default=0, help="Read one frame every N frames. 0 derives it from video FPS")
    parser.add_argument("--max-preload-frames", type=int, default=600, help="Maximum sampled frames to preload per video")
    parser.add_argument("--encode-format", default=".jpg", choices=[".jpg", ".png"], help="Frame encoding format")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality used when --encode-format=.jpg")
    parser.add_argument("--trust-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY environment variables")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV and JSON summaries")
    parser.add_argument("--save-json", action="store_true", help="Also save full JSON summary")
    parser.add_argument("--enable-face-recognition", action="store_true", default=True, help="Enable face recognition")
    parser.add_argument("--disable-face-recognition", action="store_false", dest="enable_face_recognition")
    parser.add_argument("--enable-behavior-detection", action="store_true", default=True, help="Enable behavior detection")
    parser.add_argument("--disable-behavior-detection", action="store_false", dest="enable_behavior_detection")
    parser.add_argument("--enable-uniformer-inference", action="store_true", default=False, help="Enable uniformer inference")
    parser.add_argument("--disable-uniformer-inference", action="store_false", dest="enable_uniformer_inference")
    parser.add_argument("--enable-spatial-positioning", action="store_true", default=True, help="Enable spatial positioning")
    parser.add_argument("--disable-spatial-positioning", action="store_false", dest="enable_spatial_positioning")
    parser.add_argument("--enable-target-tracking", action="store_true", default=True, help="Enable target tracking")
    parser.add_argument("--disable-target-tracking", action="store_false", dest="enable_target_tracking")
    return parser.parse_args()


def discover_videos(video_dir: Path, explicit_paths: Sequence[str]) -> List[Path]:
    if explicit_paths:
        videos = [Path(path).expanduser().resolve() for path in explicit_paths]
    else:
        videos = [
            item.resolve()
            for item in sorted(video_dir.expanduser().glob("*"))
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        ]

    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Video file not found: {missing[0]}")
    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    return videos


def parse_camera_ids(raw: str) -> List[str]:
    camera_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not camera_ids:
        raise ValueError("--camera-ids must contain at least one camera ID")
    return camera_ids


def derive_frame_step(video_fps: float, target_fps: float, explicit_frame_step: int) -> int:
    if explicit_frame_step > 0:
        return explicit_frame_step
    if video_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(video_fps / target_fps)))


def encode_frame(frame, encode_format: str, jpeg_quality: int) -> str:
    cv2 = import_cv2()
    params: List[int] = []
    if encode_format == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    success, buffer = cv2.imencode(encode_format, frame, params)
    if not success:
        raise RuntimeError(f"Failed to encode frame as {encode_format}")
    return base64.b64encode(buffer).decode("utf-8")


def preload_video(
    video_path: Path,
    target_fps: float,
    frame_step: int,
    max_preload_frames: int,
    encode_format: str,
    jpeg_quality: int,
) -> EncodedVideo:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    effective_step = derive_frame_step(source_fps, target_fps, frame_step)
    frames: List[str] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % effective_step != 0:
                continue
            frames.append(encode_frame(frame, encode_format, jpeg_quality))
            if max_preload_frames > 0 and len(frames) >= max_preload_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be loaded from video: {video_path}")

    return EncodedVideo(path=video_path, fps=source_fps, frame_step=effective_step, frames=frames)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def make_payload(args: argparse.Namespace, camera_id: str, frame_base64: str) -> Dict[str, object]:
    return {
        "image": frame_base64,
        "camera_id": camera_id,
        "associated_camera_ids": [],
        "timestamp": datetime.now().isoformat(),
        "enable_face_recognition": args.enable_face_recognition,
        "enable_behavior_detection": args.enable_behavior_detection,
        "enable_uniformer_inference": args.enable_uniformer_inference,
        "enable_spatial_positioning": args.enable_spatial_positioning,
        "enable_target_tracking": args.enable_target_tracking,
    }


def camera_worker(
    *,
    args: argparse.Namespace,
    camera_name: str,
    camera_id: str,
    encoded_video: EncodedVideo,
    frame_offset: int,
    start_event: threading.Event,
    stop_event: threading.Event,
    recorder: MetricsRecorder,
) -> None:
    requests = import_requests()
    session = requests.Session()
    session.trust_env = args.trust_env_proxy
    start_event.wait()

    frame_index = frame_offset % len(encoded_video.frames)
    interval_seconds = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    next_send_at = time.perf_counter()

    while not stop_event.is_set():
        if interval_seconds > 0:
            now = time.perf_counter()
            if now < next_send_at:
                if stop_event.wait(next_send_at - now):
                    break

        frame_base64 = encoded_video.frames[frame_index]
        frame_index = (frame_index + 1) % len(encoded_video.frames)
        payload = make_payload(args, camera_id, frame_base64)

        started_at = time.perf_counter()
        success = False
        error_kind: Optional[str] = None
        try:
            response = session.post(args.api_url, json=payload, timeout=args.timeout)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            if response.status_code >= 400:
                error_kind = "http"
            else:
                try:
                    body = response.json()
                    if body.get("code") == 0:
                        success = True
                    else:
                        error_kind = "api"
                except ValueError:
                    error_kind = "api"
            recorder.record(camera_name, True, success, latency_ms, error_kind)
        except requests.RequestException:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            recorder.record(camera_name, True, False, latency_ms, "request")

        if interval_seconds > 0:
            next_send_at += interval_seconds
            now = time.perf_counter()
            if next_send_at < now:
                next_send_at = now


def run_stage(
    args: argparse.Namespace,
    camera_count: int,
    videos: Sequence[EncodedVideo],
    camera_ids: Sequence[str],
) -> StageMetrics:
    if args.strict_camera_ids and camera_count > len(camera_ids):
        raise ValueError(
            f"--strict-camera-ids enabled, but {camera_count} streams need {camera_count} IDs "
            f"and only {len(camera_ids)} were provided"
        )

    recorder = MetricsRecorder(camera_count=camera_count, target_fps=args.target_fps)
    start_event = threading.Event()
    stop_event = threading.Event()
    threads: List[threading.Thread] = []

    for index in range(camera_count):
        camera_name = f"cam_{index + 1:03d}"
        camera_id = camera_ids[index % len(camera_ids)]
        encoded_video = videos[index % len(videos)]
        thread = threading.Thread(
            target=camera_worker,
            kwargs={
                "args": args,
                "camera_name": camera_name,
                "camera_id": camera_id,
                "encoded_video": encoded_video,
                "frame_offset": index * 7,
                "start_event": start_event,
                "stop_event": stop_event,
                "recorder": recorder,
            },
            name=f"stress-{camera_name}",
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    start_event.set()
    if args.warmup_seconds > 0:
        time.sleep(args.warmup_seconds)

    recorder.start()
    measured_start = time.perf_counter()
    time.sleep(args.duration_seconds)
    measured_duration = time.perf_counter() - measured_start
    metrics = recorder.finish(measured_duration)

    stop_event.set()
    for thread in threads:
        thread.join(timeout=args.timeout + 5.0)

    return metrics


def write_csv(path: Path, rows: Iterable[StageMetrics]) -> None:
    fieldnames = [
        "camera_count",
        "target_fps",
        "duration_seconds",
        "sent",
        "success",
        "failed",
        "success_rate",
        "success_fps_total",
        "success_fps_per_camera",
        "min_camera_fps",
        "p50_latency_ms",
        "p95_latency_ms",
        "http_errors",
        "api_errors",
        "request_errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "camera_count": item.camera_count,
                    "target_fps": f"{item.target_fps:.3f}",
                    "duration_seconds": f"{item.duration_seconds:.3f}",
                    "sent": item.sent,
                    "success": item.success,
                    "failed": item.failed,
                    "success_rate": f"{item.success_rate:.6f}",
                    "success_fps_total": f"{item.success_fps_total:.3f}",
                    "success_fps_per_camera": f"{item.success_fps_per_camera:.3f}",
                    "min_camera_fps": f"{item.min_camera_fps:.3f}",
                    "p50_latency_ms": f"{item.p50_latency_ms:.1f}",
                    "p95_latency_ms": f"{item.p95_latency_ms:.1f}",
                    "http_errors": item.http_errors,
                    "api_errors": item.api_errors,
                    "request_errors": item.request_errors,
                }
            )


def metrics_to_jsonable(metrics: StageMetrics) -> Dict[str, object]:
    return {
        "camera_count": metrics.camera_count,
        "target_fps": metrics.target_fps,
        "duration_seconds": metrics.duration_seconds,
        "sent": metrics.sent,
        "success": metrics.success,
        "failed": metrics.failed,
        "success_rate": metrics.success_rate,
        "success_fps_total": metrics.success_fps_total,
        "success_fps_per_camera": metrics.success_fps_per_camera,
        "min_camera_fps": metrics.min_camera_fps,
        "p50_latency_ms": metrics.p50_latency_ms,
        "p95_latency_ms": metrics.p95_latency_ms,
        "http_errors": metrics.http_errors,
        "api_errors": metrics.api_errors,
        "request_errors": metrics.request_errors,
        "per_camera": {
            name: {
                "sent": item.sent,
                "success": item.success,
                "failed": item.failed,
                "success_fps": item.success / metrics.duration_seconds if metrics.duration_seconds > 0 else 0.0,
                "p50_latency_ms": percentile(item.latencies_ms, 50),
                "p95_latency_ms": percentile(item.latencies_ms, 95),
            }
            for name, item in metrics.per_camera.items()
        },
    }


def print_stage(metrics: StageMetrics, status: str) -> None:
    print(
        f"[{status}] cameras={metrics.camera_count} "
        f"success_fps_total={metrics.success_fps_total:.2f} "
        f"avg_fps_per_camera={metrics.success_fps_per_camera:.2f} "
        f"min_camera_fps={metrics.min_camera_fps:.2f} "
        f"success_rate={metrics.success_rate:.3f} "
        f"latency_p50_ms={metrics.p50_latency_ms:.1f} "
        f"latency_p95_ms={metrics.p95_latency_ms:.1f} "
        f"errors=http:{metrics.http_errors},api:{metrics.api_errors},request:{metrics.request_errors}"
    )


def import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for this stress test. Install it with "
            "`pip install opencv-python` or run inside the project environment."
        ) from exc


def import_requests():
    try:
        import requests

        return requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for this stress test. Install it with "
            "`pip install requests` or run inside the project environment."
        ) from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.start_cameras <= 0:
        raise ValueError("--start-cameras must be positive")
    if args.max_cameras < args.start_cameras:
        raise ValueError("--max-cameras must be >= --start-cameras")
    if args.step_cameras <= 0:
        raise ValueError("--step-cameras must be positive")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.min_fps <= 0:
        raise ValueError("--min-fps must be positive")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.warmup_seconds < 0:
        raise ValueError("--warmup-seconds cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not 0 <= args.min_success_rate <= 1:
        raise ValueError("--min-success-rate must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")


def main() -> None:
    args = parse_args()
    validate_args(args)

    video_paths = discover_videos(Path(args.video_dir), args.video_path)
    camera_ids = parse_camera_ids(args.camera_ids)
    if len(camera_ids) == 1 and args.max_cameras > 1 and not args.strict_camera_ids:
        print(
            "warning: only one camera_id was provided; simulated streams will share server-side "
            "camera state. Pass multiple IDs for a closer multi-camera test."
        )
    elif len(camera_ids) < args.max_cameras and not args.strict_camera_ids:
        print("warning: camera IDs will be reused because fewer IDs were provided than max streams.")

    print(f"api_url={args.api_url}")
    print(f"target_fps_per_camera={args.target_fps}")
    print(f"stop_below_avg_fps_per_camera={args.min_fps}")
    print(f"min_success_rate={args.min_success_rate}")
    print(f"trust_env_proxy={args.trust_env_proxy}")
    print(f"http_proxy={os.getenv('HTTP_PROXY') or os.getenv('http_proxy')}")
    print(f"https_proxy={os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')}")

    encoded_videos: List[EncodedVideo] = []
    for path in video_paths:
        item = preload_video(
            path,
            target_fps=args.target_fps,
            frame_step=args.frame_step,
            max_preload_frames=args.max_preload_frames,
            encode_format=args.encode_format,
            jpeg_quality=args.jpeg_quality,
        )
        encoded_videos.append(item)
        print(
            f"loaded video={item.path} source_fps={item.fps:.2f} "
            f"frame_step={item.frame_step} preloaded_frames={len(item.frames)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("stress_%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{run_id}.csv"
    json_path = output_dir / f"{run_id}.json"

    results: List[StageMetrics] = []
    last_passing: Optional[StageMetrics] = None
    first_failing: Optional[StageMetrics] = None

    for camera_count in range(args.start_cameras, args.max_cameras + 1, args.step_cameras):
        print(
            f"\nstart stage cameras={camera_count} "
            f"warmup={args.warmup_seconds:.1f}s duration={args.duration_seconds:.1f}s"
        )
        metrics = run_stage(args, camera_count, encoded_videos, camera_ids)
        results.append(metrics)

        passed = (
            metrics.success_fps_per_camera >= args.min_fps
            and metrics.success_rate >= args.min_success_rate
        )
        print_stage(metrics, "PASS" if passed else "STOP")
        write_csv(csv_path, results)

        if passed:
            last_passing = metrics
        else:
            first_failing = metrics
            break

    summary = {
        "api_url": args.api_url,
        "target_fps_per_camera": args.target_fps,
        "min_fps_per_camera": args.min_fps,
        "min_success_rate": args.min_success_rate,
        "last_passing_camera_count": last_passing.camera_count if last_passing else 0,
        "first_failing_camera_count": first_failing.camera_count if first_failing else None,
        "csv_path": str(csv_path),
        "stages": [metrics_to_jsonable(item) for item in results],
    }

    if args.save_json:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"json_saved={json_path}")

    print(f"\ncsv_saved={csv_path}")
    print(
        "result: last_passing_camera_count="
        f"{summary['last_passing_camera_count']} "
        f"first_failing_camera_count={summary['first_failing_camera_count']}"
    )


if __name__ == "__main__":
    main()
