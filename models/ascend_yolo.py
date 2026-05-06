from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from models.ascend_backend import AscendInferSession, is_om_path


class ArrayLikeTensor:
    """Minimal tensor-like wrapper for the fields used by the business code."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def int(self):
        return ArrayLikeTensor(self.array.astype(np.int64))

    def tolist(self):
        return self.array.tolist()

    def item(self):
        return self.array.item()

    def max(self):
        return self.array.max()

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        value = self.array[item]
        if isinstance(value, np.ndarray):
            return ArrayLikeTensor(value)
        return value

    @property
    def shape(self):
        return self.array.shape


class AscendBoxes:
    def __init__(self, xyxy, conf=None, cls=None, ids=None):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.xyxy = ArrayLikeTensor(xyxy)
        self.conf = ArrayLikeTensor(
            np.asarray(conf, dtype=np.float32).reshape(-1)
            if conf is not None
            else np.zeros((xyxy.shape[0],), dtype=np.float32)
        )
        self.cls = ArrayLikeTensor(
            np.asarray(cls, dtype=np.float32).reshape(-1)
            if cls is not None
            else np.zeros((xyxy.shape[0],), dtype=np.float32)
        )
        self.id = None if ids is None else ArrayLikeTensor(np.asarray(ids, dtype=np.int64).reshape(-1))


class AscendKeypoints:
    def __init__(self, data):
        self.data = ArrayLikeTensor(np.asarray(data, dtype=np.float32).reshape(-1, 17, 3))


class AscendResult:
    def __init__(self, boxes: AscendBoxes, keypoints: AscendKeypoints = None):
        self.boxes = boxes
        self.keypoints = keypoints


def create_yolo_model(model_path: str, task: str):
    if is_om_path(model_path):
        return AscendYOLO(model_path=model_path, task=task)

    from ultralytics import YOLO

    return YOLO(model_path, task=task)


def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / h, new_w / w)
    resized_w = int(round(w * ratio))
    resized_h = int(round(h * ratio))
    dw = (new_w - resized_w) / 2.0
    dh = (new_h - resized_h) / 2.0

    if (w, h) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (left, top)


def _xywh_to_xyxy(boxes):
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _scale_boxes(boxes, ratio, pad, original_shape):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    h, w = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def _scale_keypoints(kpts, ratio, pad, original_shape):
    kpts = kpts.copy()
    kpts[:, :, 0] = (kpts[:, :, 0] - pad[0]) / ratio
    kpts[:, :, 1] = (kpts[:, :, 1] - pad[1]) / ratio
    h, w = original_shape[:2]
    kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w)
    kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h)
    return kpts


def _box_iou(box, boxes):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-12)


def _nms(boxes, scores, classes, iou_thres):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[current], boxes[rest])
        same_class = classes[rest] == classes[current]
        suppress = (ious > iou_thres) & same_class
        order = rest[~suppress]
    return keep


def _normalize_classes(classes):
    if classes is None:
        return None
    if isinstance(classes, (int, np.integer)):
        return {int(classes)}
    return {int(cls) for cls in classes}


def _prepare_prediction(output):
    pred = np.asarray(output)
    pred = np.squeeze(pred)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[0], -1)
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
        pred = pred.T
    return pred.astype(np.float32, copy=False)


def _sigmoid_if_needed(values):
    if values.size == 0:
        return values
    if np.nanmin(values) < 0.0 or np.nanmax(values) > 1.0:
        return 1.0 / (1.0 + np.exp(-values))
    return values


class AscendYOLO:
    def __init__(self, model_path: str, task: str = "detect", input_size=(640, 640)):
        self.model_path = model_path
        self.task = task
        self.input_size = input_size
        self.session = AscendInferSession(model_path)
        self.tracker = None

    def __call__(self, source, *args, **kwargs):
        return self.predict(source, *args, **kwargs)

    def _preprocess(self, image):
        padded, ratio, pad = _letterbox(image, self.input_size)
        chw = padded[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(chw[None].astype(np.float32) / 255.0)
        return blob, ratio, pad

    def predict(
        self,
        source,
        verbose=False,
        classes=None,
        conf=0.25,
        iou=0.45,
        device=None,
        **kwargs,
    ):
        image = np.asarray(source)
        blob, ratio, pad = self._preprocess(image)
        outputs = self.session.infer(blob)
        pred = _prepare_prediction(outputs[0])
        class_filter = _normalize_classes(classes)

        if self.task == "pose":
            result = self._postprocess_pose(pred, image.shape, ratio, pad, float(conf), float(iou))
        else:
            result = self._postprocess_detect(
                pred,
                image.shape,
                ratio,
                pad,
                float(conf),
                float(iou),
                class_filter,
            )
        return [result]

    def track(self, source, persist=True, tracker=None, **kwargs):
        if not persist or self.tracker is None:
            self.tracker = self._create_tracker()

        result = self.predict(source, **kwargs)[0]
        boxes = result.boxes.xyxy.numpy()
        conf = result.boxes.conf.numpy()
        cls = result.boxes.cls.numpy()
        if len(boxes) == 0:
            tracked = self.tracker.update(np.empty((0, 5), dtype=np.float32))
            return [AscendResult(AscendBoxes(np.empty((0, 4), dtype=np.float32), ids=np.empty((0,), dtype=np.int64)))]

        dets = np.concatenate([boxes, conf.reshape(-1, 1)], axis=1).astype(np.float32)
        tracked = self.tracker.update(dets)
        if tracked.size == 0:
            return [AscendResult(AscendBoxes(np.empty((0, 4), dtype=np.float32), ids=np.empty((0,), dtype=np.int64)))]

        tracked_boxes = tracked[:, :4].astype(np.float32)
        tracked_ids = tracked[:, 4].astype(np.int64)
        matched_conf = np.zeros((len(tracked_boxes),), dtype=np.float32)
        matched_cls = np.zeros((len(tracked_boxes),), dtype=np.float32)
        for index, tracked_box in enumerate(tracked_boxes):
            ious = _box_iou(tracked_box, boxes)
            if ious.size > 0:
                best = int(np.argmax(ious))
                matched_conf[index] = conf[best]
                matched_cls[index] = cls[best]

        return [AscendResult(AscendBoxes(tracked_boxes, matched_conf, matched_cls, tracked_ids))]

    @staticmethod
    def _create_tracker():
        from models.ocsort import OCSort

        return OCSort(max_age=30, min_hits=0, iou_threshold=0.3)

    def _postprocess_detect(self, pred, original_shape, ratio, pad, conf_thres, iou_thres, class_filter):
        if pred.size == 0:
            return AscendResult(AscendBoxes(np.empty((0, 4), dtype=np.float32)))

        cols = pred.shape[1]
        if cols == 6:
            boxes = pred[:, :4]
            if boxes.size and np.nanmax(boxes) <= 2.0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] *= self.input_size[1]
                boxes[:, [1, 3]] *= self.input_size[0]
            scores = _sigmoid_if_needed(pred[:, 4])
            cls_ids = pred[:, 5].astype(np.int64)
            boxes = _scale_boxes(boxes, ratio, pad, original_shape)
        else:
            raw_boxes = pred[:, :4].copy()
            if raw_boxes.size and np.nanmax(raw_boxes) <= 2.0:
                raw_boxes[:, [0, 2]] *= self.input_size[1]
                raw_boxes[:, [1, 3]] *= self.input_size[0]
            boxes = _xywh_to_xyxy(raw_boxes)
            if cols == 85:
                obj = _sigmoid_if_needed(pred[:, 4])
                class_scores = _sigmoid_if_needed(pred[:, 5:])
                cls_ids = class_scores.argmax(axis=1)
                scores = obj * class_scores[np.arange(len(class_scores)), cls_ids]
            else:
                class_scores = _sigmoid_if_needed(pred[:, 4:])
                cls_ids = class_scores.argmax(axis=1)
                scores = class_scores[np.arange(len(class_scores)), cls_ids]
            boxes = _scale_boxes(boxes, ratio, pad, original_shape)

        mask = scores >= conf_thres
        if class_filter is not None:
            mask &= np.isin(cls_ids, list(class_filter))

        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        keep = _nms(boxes, scores, cls_ids, iou_thres)

        return AscendResult(
            AscendBoxes(
                boxes[keep] if keep else np.empty((0, 4), dtype=np.float32),
                scores[keep] if keep else np.empty((0,), dtype=np.float32),
                cls_ids[keep] if keep else np.empty((0,), dtype=np.float32),
            )
        )

    def _postprocess_pose(self, pred, original_shape, ratio, pad, conf_thres, iou_thres):
        if pred.size == 0 or pred.shape[1] < 56:
            return AscendResult(
                AscendBoxes(np.empty((0, 4), dtype=np.float32)),
                AscendKeypoints(np.empty((0, 17, 3), dtype=np.float32)),
            )

        num_keypoints = 17
        keypoint_values = num_keypoints * 3
        num_classes = max(1, pred.shape[1] - 4 - keypoint_values)
        class_scores = _sigmoid_if_needed(pred[:, 4 : 4 + num_classes])
        cls_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_scores)), cls_ids]
        kpt_start = 4 + num_classes

        raw_boxes = pred[:, :4].copy()
        if raw_boxes.size and np.nanmax(raw_boxes) <= 2.0:
            raw_boxes[:, [0, 2]] *= self.input_size[1]
            raw_boxes[:, [1, 3]] *= self.input_size[0]
        boxes = _scale_boxes(_xywh_to_xyxy(raw_boxes), ratio, pad, original_shape)
        keypoints = pred[:, kpt_start : kpt_start + keypoint_values].reshape(-1, num_keypoints, 3)
        keypoints[:, :, 2] = _sigmoid_if_needed(keypoints[:, :, 2])
        if keypoints.size and np.nanmax(keypoints[:, :, :2]) <= 2.0:
            keypoints = keypoints.copy()
            keypoints[:, :, 0] *= self.input_size[1]
            keypoints[:, :, 1] *= self.input_size[0]
        keypoints = _scale_keypoints(keypoints, ratio, pad, original_shape)

        mask = scores >= conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        keypoints = keypoints[mask]
        keep = _nms(boxes, scores, cls_ids, iou_thres)

        return AscendResult(
            AscendBoxes(
                boxes[keep] if keep else np.empty((0, 4), dtype=np.float32),
                scores[keep] if keep else np.empty((0,), dtype=np.float32),
                cls_ids[keep] if keep else np.empty((0,), dtype=np.float32),
            ),
            AscendKeypoints(keypoints[keep] if keep else np.empty((0, 17, 3), dtype=np.float32)),
        )
