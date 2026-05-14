"""
Video-based test client for /api/v1/person/detect.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime

import cv2
import requests


DEFAULT_VIDEO_PATH = "/home/wangchenhao/video/PersonTracking_om/video/w1.mp4"
DEFAULT_API_URL = "http://127.0.0.1:8130/api/v1/person/detect"
DEFAULT_CAMERA_ID = "207"
DEFAULT_FRAME_STEP = 5
DEFAULT_TIMEOUT = 30
DEFAULT_ENCODE_FORMAT = ".jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Test person detect API with a local video file.")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Person detect API URL")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID, help="Camera ID")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Send one frame every N frames",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--encode-format",
        default=DEFAULT_ENCODE_FORMAT,
        choices=[".jpg", ".png"],
        help="Frame encoding format before base64",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after sending this many frames, 0 means no limit",
    )
    parser.add_argument(
        "--trust-env-proxy",
        action="store_true",
        help="Honor HTTP_PROXY/HTTPS_PROXY from the environment",
    )
    parser.add_argument(
        "--enable-face-recognition",
        action="store_true",
        default=True,
        help="Enable face recognition",
    )
    parser.add_argument(
        "--disable-face-recognition",
        action="store_false",
        dest="enable_face_recognition",
        help="Disable face recognition",
    )
    parser.add_argument(
        "--enable-behavior-detection",
        action="store_true",
        default=True,
        help="Enable behavior detection",
    )
    parser.add_argument(
        "--disable-behavior-detection",
        action="store_false",
        dest="enable_behavior_detection",
        help="Disable behavior detection",
    )
    parser.add_argument(
        "--enable-uniformer-inference",
        action="store_true",
        default=False,
        help="Enable uniformer inference",
    )
    parser.add_argument(
        "--disable-uniformer-inference",
        action="store_false",
        dest="enable_uniformer_inference",
        help="Disable uniformer inference",
    )
    parser.add_argument(
        "--enable-spatial-positioning",
        action="store_true",
        default=True,
        help="Enable spatial positioning",
    )
    parser.add_argument(
        "--disable-spatial-positioning",
        action="store_false",
        dest="enable_spatial_positioning",
        help="Disable spatial positioning",
    )
    parser.add_argument(
        "--enable-target-tracking",
        action="store_true",
        default=True,
        help="Enable target tracking",
    )
    parser.add_argument(
        "--disable-target-tracking",
        action="store_false",
        dest="enable_target_tracking",
        help="Disable target tracking",
    )
    return parser.parse_args()


def encode_frame(frame, encode_format):
    success, buffer = cv2.imencode(encode_format, frame)
    if not success:
        raise RuntimeError(f"Failed to encode frame as {encode_format}")
    return base64.b64encode(buffer).decode("utf-8")


def summarize_result(result):
    data = result.get("data") or {}
    persons = data.get("persons") or []
    if not persons:
        return "no persons"

    parts = []
    for person in persons:
        part = (
            f"person_id={person.get('person_id')} "
            f"track_id={person.get('track_id')} "
            f"id_resource={person.get('id_resource')} "
            f"switch_from={person.get('switch_from')} "
            f"conf={person.get('conf')}"
        )
        behaviors = person.get("behavior_events") or []
        if behaviors:
            behavior_names = ",".join(
                f"{item.get('behavior_type')}:{item.get('confidence')}" for item in behaviors
            )
            part += f" behaviors=[{behavior_names}]"
        parts.append(part)
    return " | ".join(parts)


def main():
    args = parse_args()
    session = requests.Session()
    session.trust_env = args.trust_env_proxy

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    print(f"video_path={args.video_path}")
    print(f"api_url={args.api_url}")
    print(f"camera_id={args.camera_id}")
    print(f"frame_step={args.frame_step}")
    print(f"encode_format={args.encode_format}")
    print(f"trust_env_proxy={args.trust_env_proxy}")
    print(f"http_proxy={os.getenv('HTTP_PROXY') or os.getenv('http_proxy')}")
    print(f"https_proxy={os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')}")

    frame_index = 0
    sent_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("video finished")
                break

            frame_index += 1
            if args.frame_step > 1 and (frame_index - 1) % args.frame_step != 0:
                continue

            image_base64 = encode_frame(frame, args.encode_format)
            payload = {
                "image": image_base64,
                "camera_id": args.camera_id,
                "associated_camera_ids": [],
                "timestamp": datetime.now().isoformat(),
                "enable_face_recognition": args.enable_face_recognition,
                "enable_behavior_detection": args.enable_behavior_detection,
                "enable_uniformer_inference": args.enable_uniformer_inference,
                "enable_spatial_positioning": args.enable_spatial_positioning,
                "enable_target_tracking": args.enable_target_tracking,
            }

            try:
                started_at = time.time()
                response = session.post(args.api_url, json=payload, timeout=args.timeout)
                elapsed_ms = (time.time() - started_at) * 1000.0
                if response.status_code >= 400:
                    print(
                        f"frame={frame_index} sent={sent_count + 1} "
                        f"status={response.status_code} cost_ms={elapsed_ms:.1f} "
                        f"body={response.text[:500]}"
                    )
                    break
                result = response.json()
                summary = summarize_result(result)
                print(
                    f"frame={frame_index} sent={sent_count + 1} "
                    f"status={response.status_code} cost_ms={elapsed_ms:.1f} {summary}"
                )
            except Exception as exc:
                print(f"frame={frame_index} request_failed={exc}")
                break

            sent_count += 1
            if args.max_frames > 0 and sent_count >= args.max_frames:
                print(f"stop after max_frames={args.max_frames}")
                break
    finally:
        cap.release()


if __name__ == "__main__":
    main()
