import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from service import CameraConfigError, VisionAnalysisService


CAMERA_ID = "203"
VIDEO_IN_PATH = (
    "./video/w1.mp4"
)
VIDEO_OUT_PATH = "results/output_203_long.mp4"
JSON_OUT_PATH = "results/location_203.json"

PROCESS_TARGET_FPS = 5.0
ENABLE_FACE_RECOGNITION = True
ENABLE_BEHAVIOR_DETECTION = True
ENABLE_UNIFORMER_INFERENCE = True
ENABLE_SPATIAL_POSITIONING = True
ENABLE_TARGET_TRACKING = True


def _open_video_io(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, writer, fps


def _color_from_identity(person_id):
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


def _draw_text_block(frame, anchor_x, anchor_y, color, lines):
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


def _draw_person(frame, person):
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


def _draw_frame_header(frame, frame_id, exist_person, processed):
    status = "processed" if processed else "reused"
    header = f"Frame {frame_id} | service={status} | exist_person={exist_person}"
    cv2.putText(
        frame,
        header,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def main():
    os.makedirs("results", exist_ok=True)

    service = VisionAnalysisService()
    cap, writer, fps = _open_video_io(VIDEO_IN_PATH, VIDEO_OUT_PATH)

    process_interval = max(1, int(round(fps / PROCESS_TARGET_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_persons_data = []
    frame_idx = 0
    last_result = {"exist_person": False, "persons": []}

    print(
        "Start processing video with service-aligned flow. "
        f"Source FPS: {fps:.2f}, process interval: {process_interval}"
    )

    try:
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                processed_this_frame = False
                if frame_idx % process_interval == 0:
                    processed_this_frame = True
                    exist_person, persons = service.detect_person_from_image(
                        frame,
                        camera_id=CAMERA_ID,
                        enable_face=ENABLE_FACE_RECOGNITION,
                        enable_behavior=ENABLE_BEHAVIOR_DETECTION,
                        enable_uniformer=ENABLE_UNIFORMER_INFERENCE,
                        enable_positioning=ENABLE_SPATIAL_POSITIONING,
                        enable_tracking=ENABLE_TARGET_TRACKING,
                    )

                    frame_result = {
                        "frame_id": frame_idx,
                        "exist_person": bool(exist_person),
                        "persons": persons,
                    }
                    all_persons_data.append(frame_result)
                    last_result = {
                        "exist_person": bool(exist_person),
                        "persons": persons,
                    }

                persons_to_draw = last_result.get("persons", [])
                for person in persons_to_draw:
                    _draw_person(frame, person)

                _draw_frame_header(
                    frame,
                    frame_idx,
                    last_result.get("exist_person", False),
                    processed_this_frame,
                )

                writer.write(frame)
                frame_idx += 1
                pbar.update(1)
    except CameraConfigError as exc:
        raise RuntimeError(
            f"Camera config load failed for camera_id={CAMERA_ID}: {exc}"
        ) from exc
    finally:
        cap.release()
        writer.release()

    print("Processing completed, saving outputs...")
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_persons_data, f, indent=2, ensure_ascii=False)

    print(f"Annotated video saved to: {VIDEO_OUT_PATH}")
    print(f"Service result JSON saved to: {JSON_OUT_PATH}")


if __name__ == "__main__":
    main()
