"""
Stress test multiple single-worker Docker instances with fixed camera routing.

The script reuses one source video for simulated cameras 201-210. It runs
stages from 1 camera up to 10 cameras. Each camera sends at a fixed cadence
without waiting for previous responses, and each camera is routed to a Docker
instance by a stable round-robin rule:

    camera_index % docker_count -> port 8130 + index

Example:
    python stress_multi_docker_5fps.py --docker-count 3
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_HOST = "http://192.168.100.64"
DEFAULT_START_PORT = 8130
DEFAULT_VIDEO_PATH = "video/test1.mp4"
DEFAULT_CAMERA_IDS = [str(camera_id) for camera_id in range(201, 211)]
DEFAULT_TARGET_FPS = 5.0
DEFAULT_STAGE_SECONDS = 60.0
DEFAULT_WARMUP_SECONDS = 0.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_REQUEST_WORKERS = 256
DEFAULT_MAX_IN_FLIGHT = 2000
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_JPEG_QUALITY = 85

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class EncodedFrame:
    sequence: int
    frame_index: int
    source_time_sec: float
    image_base64: str


@dataclass
class CameraRoute:
    camera_id: str
    docker_index: int
    url: str


@dataclass
class RequestRecord:
    camera_id: str
    docker_index: int
    url: str
    stage_camera_count: int
    sequence: int
    frame_index: int
    source_time_sec: float
    scheduled_offset_sec: float
    measured: bool
    success: bool
    status_code: Optional[int] = None
    api_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class StageMetrics:
    camera_count: int
    camera_ids: List[str]
    duration_seconds: float
    warmup_seconds: float
    sent: int = 0
    measured_sent: int = 0
    success: int = 0
    failed: int = 0
    dropped_api: int = 0
    request_errors: int = 0
    http_errors: int = 0
    api_errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_docker: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def success_rate(self) -> float:
        total = self.success + self.failed
        return self.success / total if total else 0.0

    @property
    def measured_qps(self) -> float:
        return self.measured_sent / self.duration_seconds if self.duration_seconds > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged 5 FPS stress tests against multiple Docker ports. "
            "Cameras 201-210 are added from 1 route up to 10 routes."
        )
    )
    parser.add_argument("--docker-count", type=int, required=True, help="Number of Docker instances/ports to route to")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Base host, e.g. http://192.168.100.64")
    parser.add_argument("--start-port", type=int, default=DEFAULT_START_PORT, help="First Docker host port")
    parser.add_argument("--api-path", default="/api/v1/person/detect", help="API path")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Source video reused by every camera")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS, help="Send FPS per camera")
    parser.add_argument("--start-cameras", type=int, default=1, help="First camera count")
    parser.add_argument("--max-cameras", type=int, default=10, help="Final camera count")
    parser.add_argument("--stage-seconds", type=float, default=DEFAULT_STAGE_SECONDS, help="Measured seconds per stage")
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS, help="Unmeasured warmup per stage")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP request timeout seconds")
    parser.add_argument("--request-workers", type=int, default=DEFAULT_REQUEST_WORKERS, help="Client request thread count")
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT, help="Client-side pending request cap")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--run-name", default="", help="Optional run directory name")
    parser.add_argument("--max-preload-frames", type=int, default=900, help="Max sampled frames to preload from video")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality")
    parser.add_argument("--trust-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY")
    parser.add_argument("--save-records", action="store_true", help="Save per-request records to JSONL")
    parser.add_argument("--enable-face-recognition", action="store_true", default=True)
    parser.add_argument("--disable-face-recognition", action="store_false", dest="enable_face_recognition")
    parser.add_argument("--enable-behavior-detection", action="store_true", default=True)
    parser.add_argument("--disable-behavior-detection", action="store_false", dest="enable_behavior_detection")
    parser.add_argument("--enable-uniformer-inference", action="store_true", default=False)
    parser.add_argument("--disable-uniformer-inference", action="store_false", dest="enable_uniformer_inference")
    parser.add_argument("--enable-spatial-positioning", action="store_true", default=True)
    parser.add_argument("--disable-spatial-positioning", action="store_false", dest="enable_spatial_positioning")
    parser.add_argument("--enable-target-tracking", action="store_true", default=True)
    parser.add_argument("--disable-target-tracking", action="store_false", dest="enable_target_tracking")
    return parser.parse_args()


def import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read and encode the test video") from exc


def import_requests():
    try:
        import requests

        return requests
    except ImportError as exc:
        raise RuntimeError("requests is required to run the HTTP stress test") from exc


def get_session(trust_env_proxy: bool):
    requests = import_requests()
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = trust_env_proxy
        _THREAD_LOCAL.session = session
    return session


def normalize_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def build_routes(args: argparse.Namespace) -> Dict[str, CameraRoute]:
    host = normalize_host(args.host)
    api_path = args.api_path if args.api_path.startswith("/") else "/" + args.api_path
    routes: Dict[str, CameraRoute] = {}
    for index, camera_id in enumerate(DEFAULT_CAMERA_IDS):
        docker_index = index % args.docker_count
        port = args.start_port + docker_index
        routes[camera_id] = CameraRoute(
            camera_id=camera_id,
            docker_index=docker_index,
            url=f"{host}:{port}{api_path}",
        )
    return routes


def preload_video_frames(video_path: Path, target_fps: float, max_frames: int, jpeg_quality: int) -> List[EncodedFrame]:
    cv2 = import_cv2()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or math.isnan(source_fps):
            source_fps = 25.0
        frame_step = max(1, int(round(source_fps / target_fps)))
        frames: List[EncodedFrame] = []
        frame_index = 0
        sample_sequence = 0

        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                ok_encode, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok_encode:
                    raise RuntimeError(f"Failed to JPEG encode frame {frame_index}")
                frames.append(
                    EncodedFrame(
                        sequence=sample_sequence,
                        frame_index=frame_index,
                        source_time_sec=frame_index / source_fps,
                        image_base64=base64.b64encode(buffer).decode("utf-8"),
                    )
                )
                sample_sequence += 1
            frame_index += 1

        if not frames:
            raise RuntimeError(f"No frames could be preloaded from {video_path}")
        return frames
    finally:
        cap.release()


def build_payload(args: argparse.Namespace, camera_id: str, image_base64: str) -> Dict[str, Any]:
    return {
        "image": image_base64,
        "camera_id": camera_id,
        "associated_camera_ids": [],
        "timestamp": datetime.now().isoformat(),
        "enable_face_recognition": args.enable_face_recognition,
        "enable_behavior_detection": args.enable_behavior_detection,
        "enable_uniformer_inference": args.enable_uniformer_inference,
        "enable_spatial_positioning": args.enable_spatial_positioning,
        "enable_target_tracking": args.enable_target_tracking,
    }


def post_frame(
    args: argparse.Namespace,
    route: CameraRoute,
    stage_camera_count: int,
    frame: EncodedFrame,
    camera_sequence: int,
    scheduled_offset_sec: float,
    measured: bool,
) -> RequestRecord:
    started_at = time.perf_counter()
    record = RequestRecord(
        camera_id=route.camera_id,
        docker_index=route.docker_index,
        url=route.url,
        stage_camera_count=stage_camera_count,
        sequence=camera_sequence,
        frame_index=frame.frame_index,
        source_time_sec=frame.source_time_sec,
        scheduled_offset_sec=scheduled_offset_sec,
        measured=measured,
        success=False,
    )

    try:
        session = get_session(args.trust_env_proxy)
        response = session.post(
            route.url,
            json=build_payload(args, route.camera_id, frame.image_base64),
            timeout=args.timeout,
        )
        record.latency_ms = (time.perf_counter() - started_at) * 1000.0
        record.status_code = response.status_code
        if response.status_code >= 400:
            record.error = response.text[:1000]
            return record

        body = response.json()
        record.api_code = body.get("code")
        record.success = record.api_code == 0
        if not record.success:
            record.error = body.get("message", "api code is not 0")
        return record
    except Exception as exc:
        record.latency_ms = (time.perf_counter() - started_at) * 1000.0
        record.error = str(exc)
        return record


def update_bucket(bucket: Dict[str, Any], record: RequestRecord) -> None:
    bucket["sent"] = bucket.get("sent", 0) + 1
    bucket["success"] = bucket.get("success", 0) + int(record.success)
    bucket["failed"] = bucket.get("failed", 0) + int(not record.success)
    if record.latency_ms is not None:
        bucket.setdefault("latencies_ms", []).append(record.latency_ms)


def record_metrics(metrics: StageMetrics, record: RequestRecord) -> None:
    if not record.measured:
        return

    metrics.measured_sent += 1
    if record.success:
        metrics.success += 1
        if record.latency_ms is not None:
            metrics.latencies_ms.append(record.latency_ms)
    else:
        metrics.failed += 1
        if record.status_code is None:
            metrics.request_errors += 1
        elif record.status_code >= 400:
            metrics.http_errors += 1
        else:
            metrics.api_errors += 1
        if record.api_code == 1203:
            metrics.dropped_api += 1

    update_bucket(metrics.per_camera.setdefault(record.camera_id, {}), record)
    update_bucket(metrics.per_docker.setdefault(str(record.docker_index), {}), record)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    latencies = bucket.get("latencies_ms", [])
    return {
        "sent": bucket.get("sent", 0),
        "success": bucket.get("success", 0),
        "failed": bucket.get("failed", 0),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
    }


def run_stage(
    args: argparse.Namespace,
    routes: Dict[str, CameraRoute],
    frames: List[EncodedFrame],
    camera_count: int,
    jsonl_handle,
) -> StageMetrics:
    active_camera_ids = DEFAULT_CAMERA_IDS[:camera_count]
    metrics = StageMetrics(
        camera_count=camera_count,
        camera_ids=active_camera_ids,
        duration_seconds=args.stage_seconds,
        warmup_seconds=args.warmup_seconds,
    )
    interval = 1.0 / args.target_fps
    run_seconds = args.warmup_seconds + args.stage_seconds
    stage_started_at = time.perf_counter()
    next_send = {camera_id: stage_started_at for camera_id in active_camera_ids}
    camera_sequences = {camera_id: 0 for camera_id in active_camera_ids}
    pending: set[Future] = set()

    def collect_done(done_futures: Sequence[Future]) -> None:
        for future in done_futures:
            record = future.result()
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            record_metrics(metrics, record)

    with ThreadPoolExecutor(max_workers=max(1, int(args.request_workers))) as executor:
        while True:
            now = time.perf_counter()
            elapsed = now - stage_started_at
            if elapsed >= run_seconds:
                break

            for camera_id in active_camera_ids:
                while now >= next_send[camera_id] and next_send[camera_id] - stage_started_at < run_seconds:
                    if len(pending) >= args.max_in_flight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        collect_done(list(done))

                    sequence = camera_sequences[camera_id]
                    frame = frames[sequence % len(frames)]
                    scheduled_offset_sec = next_send[camera_id] - stage_started_at
                    measured = scheduled_offset_sec >= args.warmup_seconds
                    future = executor.submit(
                        post_frame,
                        args,
                        routes[camera_id],
                        camera_count,
                        frame,
                        sequence,
                        scheduled_offset_sec,
                        measured,
                    )
                    pending.add(future)
                    metrics.sent += 1
                    camera_sequences[camera_id] += 1
                    next_send[camera_id] += interval

            done_now = [future for future in list(pending) if future.done()]
            for future in done_now:
                pending.remove(future)
            collect_done(done_now)

            time.sleep(0.002)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect_done(list(done))

    metrics.per_camera = {
        camera_id: summarize_bucket(bucket)
        for camera_id, bucket in metrics.per_camera.items()
    }
    metrics.per_docker = {
        docker_index: summarize_bucket(bucket)
        for docker_index, bucket in metrics.per_docker.items()
    }
    return metrics


def metrics_to_jsonable(metrics: StageMetrics) -> Dict[str, Any]:
    return {
        "camera_count": metrics.camera_count,
        "camera_ids": metrics.camera_ids,
        "duration_seconds": metrics.duration_seconds,
        "warmup_seconds": metrics.warmup_seconds,
        "sent_total": metrics.sent,
        "measured_sent": metrics.measured_sent,
        "success": metrics.success,
        "failed": metrics.failed,
        "success_rate": metrics.success_rate,
        "measured_qps": metrics.measured_qps,
        "avg_latency_ms": metrics.avg_latency_ms,
        "p50_latency_ms": metrics.p50_latency_ms,
        "p95_latency_ms": metrics.p95_latency_ms,
        "request_errors": metrics.request_errors,
        "http_errors": metrics.http_errors,
        "api_errors": metrics.api_errors,
        "dropped_api_1203": metrics.dropped_api,
        "per_camera": metrics.per_camera,
        "per_docker": metrics.per_docker,
    }


def print_stage(metrics: StageMetrics) -> None:
    print(
        "stage cameras={count} measured_sent={sent} success={success} failed={failed} "
        "success_rate={rate:.3f} avg_latency_ms={avg:.1f} p50_ms={p50:.1f} p95_ms={p95:.1f} "
        "measured_qps={qps:.2f} dropped_1203={dropped}".format(
            count=metrics.camera_count,
            sent=metrics.measured_sent,
            success=metrics.success,
            failed=metrics.failed,
            rate=metrics.success_rate,
            avg=metrics.avg_latency_ms,
            p50=metrics.p50_latency_ms,
            p95=metrics.p95_latency_ms,
            qps=metrics.measured_qps,
            dropped=metrics.dropped_api,
        )
    )


def make_run_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    run_name = args.run_name
    if not run_name:
        run_name = datetime.now().strftime("multi_docker_stress_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    if args.docker_count <= 0:
        raise ValueError("--docker-count must be positive")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.start_cameras < 1 or args.max_cameras > len(DEFAULT_CAMERA_IDS):
        raise ValueError("--start-cameras/--max-cameras must stay within 1..10")
    if args.start_cameras > args.max_cameras:
        raise ValueError("--start-cameras cannot be greater than --max-cameras")

    routes = build_routes(args)
    video_path = Path(args.video_path)
    frames = preload_video_frames(video_path, args.target_fps, args.max_preload_frames, args.jpeg_quality)
    run_dir = make_run_dir(args)
    records_path = run_dir / "requests.jsonl"
    summary_path = run_dir / "summary.json"

    print(f"host={normalize_host(args.host)} start_port={args.start_port} docker_count={args.docker_count}")
    print(f"video_path={video_path} preloaded_frames={len(frames)} target_fps={args.target_fps}")
    print(f"stage_seconds={args.stage_seconds} warmup_seconds={args.warmup_seconds}")
    print("camera routing:")
    for camera_id in DEFAULT_CAMERA_IDS:
        route = routes[camera_id]
        print(f"  camera_id={camera_id} -> docker_index={route.docker_index} url={route.url}")
    print(f"run_dir={run_dir}")

    results: List[StageMetrics] = []
    jsonl_handle = records_path.open("w", encoding="utf-8") if args.save_records else None
    try:
        for camera_count in range(args.start_cameras, args.max_cameras + 1):
            print(f"\nstart stage camera_count={camera_count}")
            metrics = run_stage(args, routes, frames, camera_count, jsonl_handle)
            results.append(metrics)
            print_stage(metrics)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    summary = {
        "host": normalize_host(args.host),
        "start_port": args.start_port,
        "docker_count": args.docker_count,
        "api_path": args.api_path,
        "video_path": str(video_path),
        "target_fps": args.target_fps,
        "stage_seconds": args.stage_seconds,
        "warmup_seconds": args.warmup_seconds,
        "camera_ids": DEFAULT_CAMERA_IDS,
        "routes": {camera_id: routes[camera_id].__dict__ for camera_id in DEFAULT_CAMERA_IDS},
        "stages": [metrics_to_jsonable(item) for item in results],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\ndone")
    print(f"saved_summary={summary_path}")
    if args.save_records:
        print(f"saved_records={records_path}")


if __name__ == "__main__":
    main()
