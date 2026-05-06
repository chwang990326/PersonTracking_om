import os

import cv2
from ultralytics import YOLO


# Manual configuration: change these paths before running.
# MODEL_PATH = "weights/yolov8n_crowdHuman.onnx"
MODEL_PATH = "weights/yolo11s.onnx"
VIDEO_IN_PATH = "video/0326测试.mp4"
VIDEO_OUT_PATH = "results/test_yolo_output_conf03.mp4"
CONF_THRESHOLD = 0.3
CLASS_IDS = [0]


def draw_detections(frame, boxes, confs):
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"person {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_top = max(0, y1 - text_h - baseline - 6)
        text_bottom = text_top + text_h + baseline + 6
        text_right = x1 + text_w + 8

        cv2.rectangle(
            frame,
            (x1, text_top),
            (text_right, text_bottom),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(VIDEO_IN_PATH):
        raise FileNotFoundError(f"Input video not found: {VIDEO_IN_PATH}")

    os.makedirs(os.path.dirname(VIDEO_OUT_PATH) or ".", exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Input video: {VIDEO_IN_PATH}")
    print(f"Output video: {VIDEO_OUT_PATH}")

    model = YOLO(MODEL_PATH, task="detect")

    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_IN_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {VIDEO_OUT_PATH}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame,
                verbose=False,
                conf=CONF_THRESHOLD,
                classes=CLASS_IDS,
            )

            boxes = []
            confs = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

            vis_frame = draw_detections(frame, boxes, confs)
            writer.write(vis_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
    finally:
        cap.release()
        writer.release()

    print("Finished processing video.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Input video: {VIDEO_IN_PATH}")
    print(f"Output video: {VIDEO_OUT_PATH}")
    print(f"Total frames written: {frame_idx}")


if __name__ == "__main__":
    main()
