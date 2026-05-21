"""
Paced 5 FPS video client for /api/v1/person/detect.

The script samples video frames at 5 FPS and submits one request every
0.2 seconds in frame order. It does not wait for the previous response
before sending the next frame, so multiple requests can be in flight.
Full responses are saved under results/, and an annotated MP4 is written
similar to action_main.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import threading
import time
from concurrent.futures import Future, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests


DEFAULT_VIDEO_PATH = "video/test1.mp4"
DEFAULT_API_URL = "http://192.168.100.64:8130/api/v1/person/detect"
DEFAULT_CAMERA_ID = "207"
DEFAULT_TARGET_FPS = 5.0
DEFAULT_MAX_WORKERS = 32
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_ENCODE_FORMAT = ".jpg"
DEFAULT_JPEG_QUALITY = 85

_THREAD_LOCAL = threading.local()


@dataclass
class SampledFrame:
    sequence: int
    frame_index: int
    source_time_sec: float
    batch_second: int
    scheduled_send_time_sec: float
    client_submit_offset_sec: Optional[float]
    frame: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send video/test1.mp4 frames to the person detect API at a fixed "
            "5 FPS cadence without waiting for previous responses."
        )
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Person detect API URL")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID, help="Camera ID sent to the API")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS, help="Sample/send FPS")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum in-flight request workers. Keep this high enough to avoid client-side queuing.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--encode-format",
        default=DEFAULT_ENCODE_FORMAT,
        choices=[".jpg", ".png"],
        help="Frame encoding format before base64",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N source seconds; 0 means full video")
    parser.add_argument("--max-requests", type=int, default=0, help="Stop after N sent frames; 0 means no limit")
    parser.add_argument("--trust-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY")
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


def get_session(trust_env_proxy: bool) -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = trust_env_proxy
        _THREAD_LOCAL.session = session
    return session


def encode_frame(frame: Any, encode_format: str, jpeg_quality: int) -> str:
    if encode_format == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    else:
        params = []

    ok, buffer = cv2.imencode(encode_format, frame, params)
    if not ok:
        raise RuntimeError(f"Failed to encode frame as {encode_format}")
    return base64.b64encode(buffer).decode("utf-8")


def build_payload(args: argparse.Namespace, frame_base64: str) -> Dict[str, Any]:
    return {
        "image": frame_base64,
        "camera_id": args.camera_id,
        "associated_camera_ids": [],
        "timestamp": datetime.now().isoformat(),
        "enable_face_recognition": args.enable_face_recognition,
        "enable_behavior_detection": args.enable_behavior_detection,
        "enable_uniformer_inference": args.enable_uniformer_inference,
        "enable_spatial_positioning": args.enable_spatial_positioning,
        "enable_target_tracking": args.enable_target_tracking,
    }


def post_sample(args: argparse.Namespace, sample: SampledFrame) -> Dict[str, Any]:
    started_at = time.time()
    record: Dict[str, Any] = {
        "sequence": sample.sequence,
        "frame_index": sample.frame_index,
        "source_time_sec": sample.source_time_sec,
        "batch_second": sample.batch_second,
        "scheduled_send_time_sec": sample.scheduled_send_time_sec,
        "client_submit_offset_sec": sample.client_submit_offset_sec,
        "client_write_offset_sec": None,
        "camera_id": args.camera_id,
        "success": False,
        "status_code": None,
        "latency_ms": None,
        "client_response_offset_sec": None,
        "response": None,
        "error": None,
    }

    try:
        frame_base64 = encode_frame(sample.frame, args.encode_format, args.jpeg_quality)
        payload = build_payload(args, frame_base64)
        session = get_session(args.trust_env_proxy)
        response = session.post(args.api_url, json=payload, timeout=args.timeout)
        latency_ms = (time.time() - started_at) * 1000.0
        record["status_code"] = response.status_code
        record["latency_ms"] = latency_ms
        if sample.client_submit_offset_sec is not None:
            record["client_response_offset_sec"] = (
                sample.client_submit_offset_sec + latency_ms / 1000.0
            )

        if response.status_code >= 400:
            record["error"] = response.text[:1000]
            return record

        body = response.json()
        record["response"] = body
        record["success"] = body.get("code") == 0
        if not record["success"]:
            record["error"] = body.get("message", "api code is not 0")
        return record
    except Exception as exc:
        record["latency_ms"] = (time.time() - started_at) * 1000.0
        if sample.client_submit_offset_sec is not None:
            record["client_response_offset_sec"] = (
                sample.client_submit_offset_sec + record["latency_ms"] / 1000.0
            )
        record["error"] = str(exc)
        return record


def _color_from_identity(person_id: Any) -> tuple:
    if isinstance(person_id, int):
        color_idx = person_id
    else:
        try:
            color_idx = int(person_id)
        except (TypeError, ValueError):
            color_idx = abs(hash(str(person_id)))

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 128, 0),
        (0, 255, 255),
    ]
    return palette[color_idx % len(palette)]


def _draw_text_block(frame: Any, anchor_x: int, anchor_y: int, color: tuple, lines: List[str]) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max((size[0] for size in text_sizes), default=0)
    total_height = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1) + 10

    box_x1 = max(0, anchor_x)
    box_y2 = max(total_height + 2, anchor_y)
    box_y1 = max(0, box_y2 - total_height)
    box_x2 = min(frame.shape[1] - 1, box_x1 + max_width + 10)

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, -1)

    luminance = (color[2] * 299 + color[1] * 587 + color[0] * 114) / 1000
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    y = box_y1 + 16
    for line, size in zip(lines, text_sizes):
        cv2.putText(frame, line, (box_x1 + 5, y), font, font_scale, text_color, thickness)
        y += size[1] + line_gap


def draw_person(frame: Any, person: Dict[str, Any]) -> None:
    bbox = person.get("bounding_box") or []
    if len(bbox) != 4:
        return

    x, y, w, h = bbox
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    person_id = person.get("person_id") or "Unknown"
    color = _color_from_identity(person_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    behavior_events = person.get("behavior_events") or []
    behavior_label = "Behavior: None"
    if behavior_events:
        top_behavior = behavior_events[0]
        behavior_type = top_behavior.get("behavior_type", "unknown")
        behavior_conf = float(top_behavior.get("confidence", 0.0))
        behavior_label = f"Behavior: {behavior_type} ({behavior_conf:.2f})"

    coords = person.get("world_coordinates") or [0.0, 0.0, 0.0]
    if len(coords) == 3:
        coord_label = f"CAD: ({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f})"
    else:
        coord_label = "CAD: N/A"

    track_id = person.get("track_id")
    id_resource = person.get("id_resource", "unknown")
    switch_from = person.get("switch_from")
    det_conf = float(person.get("conf", 0.0))
    keypoint_count = int(person.get("keypoint_count", 0))

    info_lines = [
        f"ID: {person_id} [{id_resource}]",
        f"Track: {track_id if track_id is not None else 'N/A'}  Conf: {det_conf:.2f}",
        f"Keypoints: {keypoint_count}",
        behavior_label,
        coord_label,
    ]
    if switch_from is not None:
        info_lines.append(f"Switch From: {switch_from}")

    _draw_text_block(frame, x1, max(0, y1 - 6), color, info_lines)


def draw_header(frame: Any, sample: SampledFrame, record: Dict[str, Any]) -> None:
    latency = record.get("latency_ms")
    latency_text = f"{latency:.1f}ms" if isinstance(latency, (float, int)) else "N/A"
    status = "ok" if record.get("success") else "fail"
    header = (
        f"Seq {sample.sequence} | Frame {sample.frame_index} | "
        f"t={sample.source_time_sec:.2f}s | {status} | {latency_text}"
    )
    cv2.putText(
        frame,
        header,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if record.get("success") else (0, 0, 255),
        2,
    )

    if record.get("error"):
        cv2.putText(
            frame,
            str(record["error"])[:120],
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )


def annotate_frame(sample: SampledFrame, record: Dict[str, Any]) -> Any:
    frame = sample.frame.copy()
    response = record.get("response") or {}
    data = response.get("data") or {}
    persons = data.get("persons") or []
    for person in persons:
        draw_person(frame, person)
    draw_header(frame, sample, record)
    return frame


def summarize_record(record: Dict[str, Any]) -> str:
    response = record.get("response") or {}
    data = response.get("data") or {}
    persons = data.get("persons") or []
    if not persons:
        return "no_person"

    parts = []
    for person in persons:
        parts.append(
            "person_id={person_id} track_id={track_id} resource={resource} conf={conf} switch_from={switch_from}".format(
                person_id=person.get("person_id"),
                track_id=person.get("track_id"),
                resource=person.get("id_resource"),
                conf=person.get("conf"),
                switch_from=person.get("switch_from"),
            )
        )
    return " | ".join(parts)


def write_jsonl_line(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def write_completed_result(
    sample: SampledFrame,
    record: Dict[str, Any],
    writer: cv2.VideoWriter,
    jsonl_handle: Any,
    all_records: List[Dict[str, Any]],
    started_at: float,
) -> None:
    record["client_write_offset_sec"] = time.time() - started_at
    all_records.append(record)
    write_jsonl_line(jsonl_handle, record)
    annotated = annotate_frame(sample, record)
    writer.write(annotated)
    print(
        "result seq={seq} frame={frame} source_t={source_t:.2f}s "
        "submit_t={submit_t:.2f}s response_t={response_t} write_t={write_t:.2f}s "
        "status={status} code={code} latency_ms={latency} {summary}".format(
            seq=sample.sequence,
            frame=sample.frame_index,
            source_t=sample.source_time_sec,
            submit_t=sample.client_submit_offset_sec or 0.0,
            response_t=(
                f"{record['client_response_offset_sec']:.2f}s"
                if isinstance(record.get("client_response_offset_sec"), (float, int))
                else "N/A"
            ),
            write_t=record["client_write_offset_sec"],
            status="ok" if record.get("success") else "fail",
            code=record.get("status_code"),
            latency=(
                f"{record['latency_ms']:.1f}"
                if isinstance(record.get("latency_ms"), (float, int))
                else "N/A"
            ),
            summary=summarize_record(record),
        )
    )


def output_paths(output_dir: Path, video_path: Path, target_fps: float) -> Dict[str, Path]:
    stem = video_path.stem
    suffix = f"paced_{target_fps:g}fps"
    return {
        "video": output_dir / f"{stem}_{suffix}_annotated.mp4",
        "json": output_dir / f"{stem}_{suffix}_results.json",
        "jsonl": output_dir / f"{stem}_{suffix}_results.jsonl",
    }


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    paths = output_paths(output_dir, video_path, args.target_fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or math.isnan(source_fps):
        source_fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(paths["video"]), fourcc, float(args.target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {paths['video']}")

    print(f"video_path={video_path}")
    print(f"api_url={args.api_url}")
    print(f"camera_id={args.camera_id}")
    print(f"source_fps={source_fps:.3f} total_frames={total_frames}")
    print(f"target_fps={args.target_fps:g} max_workers={args.max_workers}")
    print("send_mode=paced_nonblocking")
    print("result_write_order=response_completion_order")
    print(f"results_json={paths['json']}")
    print(f"results_jsonl={paths['jsonl']}")
    print(f"output_video={paths['video']}")
    print(f"trust_env_proxy={args.trust_env_proxy}")
    print(f"http_proxy={os.getenv('HTTP_PROXY') or os.getenv('http_proxy')}")
    print(f"https_proxy={os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')}")

    future_to_sample: Dict[Future, SampledFrame] = {}
    pending_futures = set()
    all_records: List[Dict[str, Any]] = []
    sample_sequence = 0
    next_sample_time = 0.0
    sample_period = 1.0 / float(args.target_fps)
    frame_index = 0
    stop_requested = False

    started_at = time.time()
    try:
        with paths["jsonl"].open("w", encoding="utf-8") as jsonl_handle:
            with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    source_time_sec = frame_index / source_fps
                    if args.max_seconds > 0 and source_time_sec >= args.max_seconds:
                        stop_requested = True

                    if not stop_requested and source_time_sec + 1e-9 >= next_sample_time:
                        scheduled_send_time_sec = next_sample_time
                        target_wall_time = started_at + scheduled_send_time_sec
                        sleep_seconds = target_wall_time - time.time()
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)

                        sample = SampledFrame(
                            sequence=sample_sequence,
                            frame_index=frame_index,
                            source_time_sec=source_time_sec,
                            batch_second=int(scheduled_send_time_sec),
                            scheduled_send_time_sec=scheduled_send_time_sec,
                            client_submit_offset_sec=time.time() - started_at,
                            frame=frame.copy(),
                        )
                        future = executor.submit(post_sample, args, sample)
                        future_to_sample[future] = sample
                        pending_futures.add(future)
                        print(
                            "sent seq={seq} frame={frame} source_t={source_t:.2f}s "
                            "scheduled_t={scheduled_t:.2f}s submit_t={submit_t:.2f}s in_flight={in_flight}".format(
                                seq=sample.sequence,
                                frame=sample.frame_index,
                                source_t=sample.source_time_sec,
                                scheduled_t=sample.scheduled_send_time_sec,
                                submit_t=sample.client_submit_offset_sec or 0.0,
                                in_flight=len(pending_futures),
                            )
                        )
                        sample_sequence += 1
                        next_sample_time += sample_period

                        done_futures = [future for future in list(pending_futures) if future.done()]
                        done_futures.sort(
                            key=lambda item: (
                                item.result().get("client_response_offset_sec")
                                if isinstance(item.result().get("client_response_offset_sec"), (float, int))
                                else float("inf")
                            )
                        )
                        for done_future in done_futures:
                            pending_futures.remove(done_future)
                            done_sample = future_to_sample.pop(done_future)
                            done_record = done_future.result()
                            write_completed_result(
                                done_sample,
                                done_record,
                                writer,
                                jsonl_handle,
                                all_records,
                                started_at,
                            )

                        if args.max_requests > 0 and sample_sequence >= args.max_requests:
                            stop_requested = True

                    frame_index += 1
                    if stop_requested:
                        break

                print(f"all frames submitted, waiting for {len(pending_futures)} responses...")
                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures,
                        timeout=None,
                        return_when=FIRST_COMPLETED,
                    )
                    done_list = list(done)
                    done_list.sort(
                        key=lambda item: (
                            item.result().get("client_response_offset_sec")
                            if isinstance(item.result().get("client_response_offset_sec"), (float, int))
                            else float("inf")
                        )
                    )
                    for done_future in done_list:
                        done_sample = future_to_sample.pop(done_future)
                        done_record = done_future.result()
                        write_completed_result(
                            done_sample,
                            done_record,
                            writer,
                            jsonl_handle,
                            all_records,
                            started_at,
                        )
    finally:
        cap.release()
        writer.release()

    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_path": str(video_path),
                "api_url": args.api_url,
                "camera_id": args.camera_id,
                "target_fps": args.target_fps,
                "max_workers": args.max_workers,
                "send_mode": "paced_nonblocking",
                "result_write_order": "response_completion_order",
                "source_fps": source_fps,
                "total_records": len(all_records),
                "success_records": sum(1 for item in all_records if item.get("success")),
                "elapsed_seconds": time.time() - started_at,
                "records": all_records,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("done")
    print(f"saved_json={paths['json']}")
    print(f"saved_jsonl={paths['jsonl']}")
    print(f"saved_video={paths['video']}")


if __name__ == "__main__":
    main()
