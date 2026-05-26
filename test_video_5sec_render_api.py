"""
Send /video/test1.mp4 to the person-detect API at one frame every five seconds
and render returned person boxes onto an output video.

The HTTP API does not return the 17 pose keypoint coordinates or posture labels.
For visualization only, this script also uses the local service detector on the
same frame, matches local pose detections to API person boxes by IoU, and draws
the matched keypoints/posture without changing the API response contract.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests


DEFAULT_VIDEO_PATH = "/video/test3.mp4"
DEFAULT_FALLBACK_VIDEO_PATH = "video/test3.mp4"
DEFAULT_API_URL = "http://192.168.100.64:8130/api/v1/person/detect"
DEFAULT_CAMERA_ID = "203"
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_OUTPUT_FPS = 1.0
DEFAULT_JPEG_QUALITY = 90
DEFAULT_LOCAL_POSE_CONF = 0.5
DEFAULT_LOCAL_POSE_MATCH_IOU = 0.1

COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render person boxes from /api/v1/person/detect for one frame every five seconds."
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Person detect API URL")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID, help="camera_id sent to API")
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Sample and send one frame every N source seconds",
    )
    parser.add_argument(
        "--pace-real-time",
        action="store_true",
        help="Also wait N wall-clock seconds between requests",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout seconds")
    parser.add_argument("--max-requests", type=int, default=0, help="Stop after N requests; 0 means full video")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output video and JSON")
    parser.add_argument("--output-fps", type=float, default=DEFAULT_OUTPUT_FPS, help="FPS of annotated output video")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality for requests")
    parser.add_argument(
        "--local-pose-conf",
        type=float,
        default=DEFAULT_LOCAL_POSE_CONF,
        help="Confidence threshold for local YOLO26x-Pose visualization",
    )
    parser.add_argument(
        "--local-pose-match-iou",
        type=float,
        default=DEFAULT_LOCAL_POSE_MATCH_IOU,
        help="Minimum IoU when matching local pose detections to API boxes",
    )
    parser.add_argument(
        "--disable-local-pose-render",
        action="store_true",
        help="Do not load local service models for keypoint/posture rendering",
    )
    parser.add_argument("--trust-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY")
    parser.add_argument("--disable-face-recognition", action="store_true", help="Disable face recognition")
    parser.add_argument("--disable-behavior-detection", action="store_true", help="Disable behavior detection")
    parser.add_argument("--disable-spatial-positioning", action="store_true", help="Disable spatial positioning")
    parser.add_argument("--disable-target-tracking", action="store_true", help="Disable target tracking")
    return parser.parse_args()


def resolve_video_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    fallback = Path(DEFAULT_FALLBACK_VIDEO_PATH)
    if path_text == DEFAULT_VIDEO_PATH and fallback.exists():
        return fallback

    raise FileNotFoundError(f"Input video not found: {path_text}")


def encode_frame(frame: Any, jpeg_quality: int) -> str:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, buffer = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def build_payload(frame: Any, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "image": encode_frame(frame, args.jpeg_quality),
        "camera_id": args.camera_id,
        "associated_camera_ids": [],
        "timestamp": datetime.now().isoformat(),
        "enable_face_recognition": not args.disable_face_recognition,
        "enable_behavior_detection": not args.disable_behavior_detection,
        "enable_uniformer_inference": False,
        "enable_spatial_positioning": not args.disable_spatial_positioning,
        "enable_target_tracking": not args.disable_target_tracking,
    }


def create_local_pose_helper(camera_id: str):
    from models.posture_classifier import classify_posture_with_verification
    from service import VisionAnalysisService

    visual_service = VisionAnalysisService()
    camera_params = visual_service.get_camera_params(camera_id)
    detector = visual_service.get_camera_state(camera_id)["detector"]
    return {
        "service": visual_service,
        "camera_params": camera_params,
        "detector": detector,
        "posture_fn": classify_posture_with_verification,
    }


def color_for_index(index: int) -> Tuple[int, int, int]:
    palette = [
        (0, 220, 255),
        (80, 220, 80),
        (255, 160, 60),
        (230, 90, 255),
        (255, 220, 80),
        (90, 180, 255),
    ]
    return palette[index % len(palette)]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def xywh_to_xyxy(bbox: Any) -> Optional[np.ndarray]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def run_local_pose(frame: Any, helper: Optional[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if helper is None:
        return []

    detector = helper["detector"]
    camera_params = helper["camera_params"]
    posture_fn = helper["posture_fn"]

    results = detector.predict(
        frame,
        verbose=False,
        classes=[0],
        conf=float(args.local_pose_conf),
        device=getattr(helper["service"], "yolo_device", "cpu"),
    )
    if not results:
        return []

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = None
    if result.keypoints is not None:
        keypoints = result.keypoints.data.cpu().numpy()
    if keypoints is None or len(boxes) == 0:
        return []

    pose_items: List[Dict[str, Any]] = []
    for box, kpts in zip(boxes, keypoints):
        visible_count = int(np.sum(kpts[:, 2] > 0.5))
        posture = "Unknown"
        if visible_count > 0:
            try:
                posture, _ = posture_fn(kpts, camera_params, ratio_threshold=0.7)
            except Exception as exc:
                posture = f"posture_error:{exc}"
        pose_items.append(
            {
                "box": np.asarray(box, dtype=np.float32),
                "keypoints": kpts.astype(float).tolist(),
                "keypoint_count": visible_count,
                "posture": posture,
            }
        )
    return pose_items


def attach_local_pose_to_persons(
    persons: List[Dict[str, Any]],
    pose_items: List[Dict[str, Any]],
    min_iou: float,
) -> None:
    if not persons or not pose_items:
        return

    pose_boxes = np.stack([item["box"] for item in pose_items]).astype(np.float32)
    used_pose_indexes = set()
    for person in persons:
        person_box = xywh_to_xyxy(person.get("bounding_box"))
        if person_box is None:
            continue

        ious = box_iou(person_box, pose_boxes)
        if ious.size == 0:
            continue

        order = ious.argsort()[::-1]
        best_index = None
        for candidate in order:
            candidate_int = int(candidate)
            if candidate_int not in used_pose_indexes:
                best_index = candidate_int
                break
        if best_index is None or float(ious[best_index]) < min_iou:
            continue

        used_pose_indexes.add(best_index)
        matched_pose = pose_items[best_index]
        person["_local_pose_iou"] = float(ious[best_index])
        person["_local_keypoints"] = matched_pose["keypoints"]
        person["_local_keypoint_count"] = matched_pose["keypoint_count"]
        person["_local_posture"] = matched_pose["posture"]


def draw_text_block(frame: Any, x: int, y: int, lines: Iterable[str], color: Tuple[int, int, int]) -> None:
    lines = [line for line in lines if line]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    padding = 5
    gap = 5
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + padding * 2
    height = sum(size[1] for size in sizes) + gap * (len(lines) - 1) + padding * 2

    h, w = frame.shape[:2]
    x1 = max(0, min(x, w - width - 1))
    y1 = max(0, min(y, h - height - 1))
    x2 = min(w - 1, x1 + width)
    y2 = min(h - 1, y1 + height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    luminance = (color[2] * 299 + color[1] * 587 + color[0] * 114) / 1000
    text_color = (0, 0, 0) if luminance > 145 else (255, 255, 255)
    current_y = y1 + padding + sizes[0][1]
    for line, size in zip(lines, sizes):
        cv2.putText(frame, line, (x1 + padding, current_y), font, scale, text_color, thickness, cv2.LINE_AA)
        current_y += size[1] + gap


def normalize_keypoints(person: Dict[str, Any]) -> Optional[List[List[float]]]:
    local_keypoints = person.get("_local_keypoints")
    if isinstance(local_keypoints, list) and len(local_keypoints) >= 17:
        try:
            return [
                [float(item[0]), float(item[1]), float(item[2])]
                for item in local_keypoints[:17]
            ]
        except (TypeError, ValueError, IndexError):
            pass

    keypoints = person.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) < 17:
        return None

    normalized = []
    for item in keypoints[:17]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return None
        normalized.append([float(item[0]), float(item[1]), float(item[2])])
    return normalized


def draw_keypoints(frame: Any, keypoints: Optional[List[List[float]]], color: Tuple[int, int, int]) -> bool:
    if not keypoints:
        return False

    visible = [kp[2] > 0.5 for kp in keypoints]
    for start, end in COCO_SKELETON:
        if visible[start] and visible[end]:
            p1 = (safe_int(keypoints[start][0]), safe_int(keypoints[start][1]))
            p2 = (safe_int(keypoints[end][0]), safe_int(keypoints[end][1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    for idx, kp in enumerate(keypoints):
        if visible[idx]:
            center = (safe_int(kp[0]), safe_int(kp[1]))
            cv2.circle(frame, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return True


def draw_anchor_points(frame: Any, anchors: Dict[str, Any], color: Tuple[int, int, int]) -> None:
    if not isinstance(anchors, dict):
        return

    for name, point in anchors.items():
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        center = (safe_int(point[0]), safe_int(point[1]))
        cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)
        if name == "bottom_center":
            cv2.circle(frame, center, 6, color, 1, cv2.LINE_AA)


def draw_person(frame: Any, person: Dict[str, Any], index: int) -> None:
    bbox = person.get("bounding_box") or []
    if len(bbox) != 4:
        return

    x, y, width, height = bbox
    x1, y1 = safe_int(x), safe_int(y)
    x2, y2 = safe_int(float(x) + float(width)), safe_int(float(y) + float(height))
    color = color_for_index(index)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_anchor_points(frame, person.get("bbox_anchor_points") or {}, color)

    keypoints = normalize_keypoints(person)
    drew_keypoints = draw_keypoints(frame, keypoints, color)

    posture = (
        person.get("_local_posture")
        or person.get("posture")
        or person.get("posture_type")
        or person.get("pose")
        or "local pose not matched"
    )
    keypoint_status = "local service" if drew_keypoints else "local pose not matched"
    person_id = person.get("person_id") or "Unknown"
    track_id = person.get("track_id")
    keypoint_count = person.get("_local_keypoint_count", person.get("keypoint_count", 0))
    coords = person.get("world_coordinates") or []
    local_iou = person.get("_local_pose_iou")

    lines = [
        f"ID: {person_id}  Track: {track_id if track_id is not None else 'N/A'}",
        f"Box: {x2 - x1}x{y2 - y1}  KP count: {keypoint_count}",
        f"Keypoints: {keypoint_status}",
        f"Posture: {posture}",
    ]
    if local_iou is not None:
        lines.append(f"Local pose IoU: {float(local_iou):.2f}")
    if len(coords) >= 2:
        lines.append(f"CAD: {float(coords[0]):.0f}, {float(coords[1]):.0f}")

    label_y = y1 - 6
    if label_y < 70:
        label_y = y2 + 6
    draw_text_block(frame, x1, label_y, lines, color)


def draw_frame_header(
    frame: Any,
    sequence: int,
    frame_index: int,
    source_time_sec: float,
    status: str,
    latency_ms: Optional[float],
    message: str = "",
) -> None:
    latency_text = f"{latency_ms:.1f}ms" if latency_ms is not None else "N/A"
    header = (
        f"Seq {sequence} | Frame {frame_index} | Source {source_time_sec:.2f}s | "
        f"{status} | {latency_text}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(frame, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if message:
        cv2.putText(frame, message[:140], (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)


def extract_persons(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data") or {}
    persons = data.get("persons") or []
    return persons if isinstance(persons, list) else []


def main() -> None:
    args = parse_args()
    video_path = resolve_video_path(args.video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step = max(1, int(round(source_fps * args.interval_sec)))

    run_name = f"test1_5sec_render_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_dir / "annotated.mp4"
    output_json_path = run_dir / "responses.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, args.output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_video_path}")

    session = requests.Session()
    session.trust_env = args.trust_env_proxy

    local_pose_helper = None
    if not args.disable_local_pose_render:
        print("loading local VisionAnalysisService for keypoint/posture rendering...")
        local_pose_helper = create_local_pose_helper(args.camera_id)

    print(f"video_path={video_path}")
    print(f"api_url={args.api_url}")
    print(f"camera_id={args.camera_id}")
    print(f"source_fps={source_fps:.3f}")
    print(f"interval_sec={args.interval_sec}")
    print(f"frame_step={frame_step}")
    print(f"output_video={output_video_path}")
    print(f"output_json={output_json_path}")

    records: List[Dict[str, Any]] = []
    sequence = 0
    frame_index = 0
    last_send_started_at: Optional[float] = None

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break

            if args.pace_real_time and last_send_started_at is not None:
                wait_sec = args.interval_sec - (time.time() - last_send_started_at)
                if wait_sec > 0:
                    time.sleep(wait_sec)

            source_time_sec = frame_index / source_fps
            annotated = frame.copy()
            result: Dict[str, Any] = {}
            status = "fail"
            message = ""
            latency_ms: Optional[float] = None

            try:
                payload = build_payload(frame, args)
                last_send_started_at = time.time()
                response = session.post(args.api_url, json=payload, timeout=args.timeout)
                latency_ms = (time.time() - last_send_started_at) * 1000.0
                result = response.json()
                status = "ok" if response.status_code < 400 and result.get("code") == 0 else "fail"
                message = str(result.get("message") or response.reason or "")

                persons = extract_persons(result)
                pose_items = run_local_pose(frame, local_pose_helper, args)
                attach_local_pose_to_persons(persons, pose_items, float(args.local_pose_match_iou))

                for person_index, person in enumerate(persons):
                    draw_person(annotated, person, person_index)

                print(
                    f"seq={sequence} frame={frame_index} t={source_time_sec:.2f}s "
                    f"http={response.status_code} code={result.get('code')} "
                    f"persons={len(persons)} local_pose={len(pose_items)} "
                    f"latency_ms={latency_ms:.1f} {message}"
                )
            except Exception as exc:
                message = str(exc)
                print(f"seq={sequence} frame={frame_index} request_failed={exc}")

            draw_frame_header(annotated, sequence, frame_index, source_time_sec, status, latency_ms, message)
            writer.write(annotated)

            records.append(
                {
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "source_time_sec": source_time_sec,
                    "success": status == "ok",
                    "latency_ms": latency_ms,
                    "message": message,
                    "local_pose_render_enabled": local_pose_helper is not None,
                    "response": result,
                }
            )

            sequence += 1
            if args.max_requests > 0 and sequence >= args.max_requests:
                break
            frame_index += frame_step
    finally:
        cap.release()
        writer.release()

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": str(video_path),
                "api_url": args.api_url,
                "camera_id": args.camera_id,
                "interval_sec": args.interval_sec,
                "frame_step": frame_step,
                "output_video": str(output_video_path),
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"done. wrote {sequence} annotated frames")
    print(f"output_video={output_video_path}")
    print(f"output_json={output_json_path}")


if __name__ == "__main__":
    main()
