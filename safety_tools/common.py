from __future__ import annotations

import json
import os
import math
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics_config"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402


from model_config import CLASS_NAMES, CLASS_IDS, INFERENCE_IMAGE_SIZE, NMS_IOU, normalize_class_name  # noqa: E402

Box = Sequence[float]
Point = Tuple[float, float]
GPU_INFO_CACHE: Optional[Dict[str, Any]] = None
try:
    MAX_HAZARD_EVENTS_IN_MEMORY = max(100, int(os.environ.get("SAFETY_MAX_RESULT_EVENTS", "5000")))
except ValueError:
    MAX_HAZARD_EVENTS_IN_MEMORY = 5000


class HazardIncidentTracker:
    """Report each rule/track combination only once during one monitoring task."""

    def __init__(self) -> None:
        self.active_keys: set[tuple[str, ...]] = set()
        self.reported_keys: set[tuple[str, ...]] = set()
        self.episode_counts: Dict[tuple[str, ...], int] = {}

    def reset(self) -> None:
        self.active_keys.clear()
        self.reported_keys.clear()
        self.episode_counts.clear()

    def update(self, instances: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_key: Dict[tuple[str, ...], Dict[str, Any]] = {}
        for instance in instances:
            raw_key = instance.get("_incident_key")
            if not isinstance(raw_key, (list, tuple)) or not raw_key:
                continue
            key = tuple(str(part) for part in raw_key)
            by_key.setdefault(key, instance)

        current_keys = set(by_key)
        # A missed frame or a temporary safe result must not create another
        # evidence image for a track that was already reported in this task.
        new_keys = current_keys - self.reported_keys
        self.active_keys = current_keys
        self.reported_keys.update(new_keys)
        new_incidents: List[Dict[str, Any]] = []
        for key in sorted(new_keys):
            episode = self.episode_counts.get(key, 0) + 1
            self.episode_counts[key] = episode
            item = dict(by_key[key])
            item.pop("_incident_key", None)
            item["incident_key"] = ":".join(key)
            item["incident_episode"] = episode
            item["incident_id"] = f"{item['incident_key']}:episode_{episode}"
            new_incidents.append(item)
        return new_incidents


def detect_gpu_info() -> Dict[str, Any]:
    global GPU_INFO_CACHE
    if GPU_INFO_CACHE is not None:
        return dict(GPU_INFO_CACHE)

    info: Dict[str, Any] = {"has_gpu": False, "gpu_count": 0, "error": ""}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            info["has_gpu"] = info["gpu_count"] > 0
    except Exception as exc:
        info["error"] = str(exc)
    GPU_INFO_CACHE = dict(info)
    return info


def default_device() -> str:
    return "0" if detect_gpu_info().get("has_gpu") else "cpu"


def resolve_device(requested: Optional[str]) -> str:
    raw = (requested or "").strip()
    gpu_info = detect_gpu_info()
    if not raw or raw.lower() in {"auto", "default"}:
        return default_device()
    normalized = raw.lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "0" if gpu_info.get("has_gpu") else "cpu"
    if normalized.startswith("cuda:"):
        return normalized.split(":", 1)[1] if gpu_info.get("has_gpu") else "cpu"
    if not gpu_info.get("has_gpu"):
        return "cpu"
    return raw


def ensure_dir(path: str | Path) -> None:
    target = Path(path)
    directory = target if target.suffix == "" else target.parent
    if directory:
        directory.mkdir(parents=True, exist_ok=True)


def load_model(weights: str | Path) -> YOLO:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    model = YOLO(str(weights_path))
    # Configure before predictor setup/fusion can discard the one-to-many head.
    model.overrides.update(end2end=False, agnostic_nms=False, iou=NMS_IOU, imgsz=INFERENCE_IMAGE_SIZE)
    return model


def model_class_names(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", {})
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(class_id): str(name).strip() for class_id, name in items}


def validate_model_classes(model: Any, required: Iterable[str]) -> List[int]:
    by_name = {normalize_class_name(name): class_id for class_id, name in model_class_names(model).items()}
    required = sorted(set(required))
    missing = [name for name in required if normalize_class_name(name) not in by_name]
    if missing:
        raise ValueError(f"模型缺少所选功能需要的类别: {', '.join(missing)}。")
    return [by_name[normalize_class_name(name)] for name in required]


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hazard_timestamp() -> Tuple[str, str]:
    """Return matching ISO and filename-safe timestamps for one hazard event."""
    now = datetime.now().astimezone()
    return now.isoformat(timespec="microseconds"), now.strftime("%Y%m%d_%H%M%S_%f")


def save_hazard_image(
    image: np.ndarray,
    output_hint: str | Path | None,
    algorithm: str,
    frame_id: int,
) -> Dict[str, Any]:
    """Persist one annotated hazard frame and return its common business fields.

    ``output_hint`` supplies the configured output directory and filename prefix;
    the actual evidence filename always contains a high-resolution timestamp.
    """
    timestamp, filename_timestamp = hazard_timestamp()
    hint = Path(output_hint) if output_hint is not None else PROJECT_ROOT / "outputs" / "hazards" / algorithm
    output_dir = hint if hint.suffix == "" else hint.parent
    prefix = hint.stem if hint.suffix else algorithm
    safe_algorithm = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in algorithm)
    filename = f"{prefix}_{safe_algorithm}_hazard_{filename_timestamp}_frame_{int(frame_id)}.jpg"
    image_path = (output_dir / filename).resolve()
    save_image(image, image_path)
    return {
        "hazard_timestamp": timestamp,
        "hazard_image_path": str(image_path),
        "hazard_frame_id": int(frame_id),
    }


def add_finished_at(data: Dict[str, Any]) -> Dict[str, Any]:
    data["finished_at"] = current_timestamp()
    return data


def format_yolo_detection_log(result: Any, frame_id: int) -> str:
    """Describe decoded model boxes before tracking or safety rules modify them."""
    obb = getattr(result, "obb", None)
    if obb is not None:
        count = len(obb)
        lines = [f"[YOLO OBB原始检测] frame={frame_id} count={count}（框架内置 NMS 后，跟踪前）"]
        if not count:
            lines.append("  未检测到目标")
            return "\n".join(lines)
        names = getattr(result, "names", {})
        obb = obb.cpu()
        for index, (points, box, confidence, class_id) in enumerate(
            zip(obb.xyxyxyxy.tolist(), obb.xyxy.tolist(), obb.conf.tolist(), obb.cls.tolist()), start=1
        ):
            class_id = int(class_id)
            name = names.get(class_id, str(class_id)) if isinstance(names, dict) else (
                names[class_id] if 0 <= class_id < len(names) else str(class_id)
            )
            coordinates = ", ".join(f"{float(value):.2f}" for value in box)
            polygon = "; ".join(", ".join(f"{float(value):.2f}" for value in point) for point in points)
            lines.append(
                f"  #{index} class={class_id} {name} conf={float(confidence):.4f} "
                f"xyxy=[{coordinates}] obb=[{polygon}]"
            )
        return "\n".join(lines)

    boxes = getattr(result, "boxes", None)
    count = len(boxes) if boxes is not None else 0
    lines = [f"[YOLO 原始检测] frame={frame_id} count={count}（框架内置 NMS 后，跟踪前）"]
    if not count:
        lines.append("  未检测到目标")
        return "\n".join(lines)
    names = getattr(result, "names", {})
    boxes = boxes.cpu()
    for index, (box, confidence, class_id) in enumerate(
        zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()), start=1
    ):
        class_id = int(class_id)
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else (
            names[class_id] if 0 <= class_id < len(names) else str(class_id)
        )
        coordinates = ", ".join(f"{float(value):.2f}" for value in box)
        lines.append(f"  #{index} class={class_id} {name} conf={float(confidence):.4f} xyxy=[{coordinates}]")
    return "\n".join(lines)


@contextmanager
def yolo_detection_logging(
    model: YOLO, frame_id: int, callback: Optional[Callable[[int, str], None]] = None
):
    """Capture each frame before Ultralytics' tracking callback changes Results."""
    if callback is None:
        yield
        return

    def capture(predictor: Any) -> None:
        for result in predictor.results:
            callback(frame_id, format_yolo_detection_log(result, frame_id))

    callbacks = model.callbacks.setdefault("on_predict_postprocess_end", [])
    callbacks.insert(0, capture)
    try:
        yield
    finally:
        callbacks.remove(capture)


def run_detection(model: YOLO, image: np.ndarray, conf: float = 0.4, iou: float = NMS_IOU, device: Optional[str] = None):
    predict_kwargs = {"conf": conf, "iou": iou, "imgsz": INFERENCE_IMAGE_SIZE,
                      "end2end": False, "agnostic_nms": False, "verbose": False}
    if device:
        predict_kwargs["device"] = device
    results = model.predict(source=image, **predict_kwargs)
    return results[0] if results else None


def parse_detections(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    obb = getattr(result, "obb", None)
    boxes = obb if obb is not None else getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = boxes.cpu()
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    classes = boxes.cls.tolist()
    track_ids = boxes.id.tolist() if getattr(boxes, "id", None) is not None else [None] * len(xyxy)
    points_rows = boxes.xyxyxyxy.tolist() if obb is not None else [None] * len(xyxy)
    names = model_class_names(result) or CLASS_NAMES

    detections = []
    for box, conf, class_id, track_id, points in zip(xyxy, confs, classes, track_ids, points_rows):
        normalized_box = [float(v) for v in box]
        confidence = float(conf)
        if not is_valid_box(normalized_box) or not math.isfinite(confidence):
            continue
        try:
            normalized_class_id = int(class_id)
            normalized_track_id = (
                int(track_id)
                if track_id is not None and math.isfinite(float(track_id))
                else None
            )
        except (TypeError, ValueError, OverflowError):
            continue
        detections.append(
            {
                "box": normalized_box,
                "confidence": confidence,
                "class_id": normalized_class_id,
                "class_name": names.get(normalized_class_id, str(normalized_class_id)),
                "track_id": normalized_track_id,
                **({"points": points} if points is not None else {}),
            }
        )
    return detections


def filter_detections(detections: Iterable[Dict[str, Any]], class_name: str) -> List[Dict[str, Any]]:
    return [
        det
        for det in detections
        if str(det.get("class_name", CLASS_NAMES.get(det.get("class_id"), ""))).casefold() == class_name.casefold()
        and is_valid_box(det.get("box"))
    ]


RAW_DETECTION_COLORS = {
    CLASS_IDS["person"]: (255, 120, 0),
    CLASS_IDS["helmet"]: (0, 180, 0),
    CLASS_IDS["head_no_helmet"]: (0, 0, 220),
    CLASS_IDS["forklift"]: (255, 0, 180),
    CLASS_IDS["fence"]: (0, 200, 255),
    CLASS_IDS["safety officer"]: (0, 210, 0),
    CLASS_IDS["crane"]: (255, 180, 0),
}


def draw_raw_detection_boxes(
    image: np.ndarray,
    detections: Iterable[Dict[str, Any]],
    thickness: int = 3,
) -> np.ndarray:
    """Draw model detection boxes only, without labels, text, or derived rule geometry."""
    annotated = image.copy()
    for detection in detections:
        box = int_box(detection.get("box"))
        if not box or not is_valid_box(box):
            continue
        x1, y1, x2, y2 = box
        name = detection.get("class_name", CLASS_NAMES.get(detection.get("class_id"), ""))
        color = RAW_DETECTION_COLORS.get(CLASS_IDS.get(name), (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(3, int(thickness)))
    return annotated


def is_valid_box(box: Any, min_size: float = 1e-6) -> bool:
    """Return whether a value is a finite, non-degenerate XYXY box."""
    if box is None:
        return False
    try:
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x2 - x1 > float(min_size)
        and y2 - y1 > float(min_size)
    )


def clip_box(box: Box, width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        float(max(0, min(width - 1, x1))),
        float(max(0, min(height - 1, y1))),
        float(max(0, min(width - 1, x2))),
        float(max(0, min(height - 1, y2))),
    ]


def int_box(box: Optional[Box]) -> Optional[List[int]]:
    if not is_valid_box(box):
        return None
    return [int(round(v)) for v in box]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    union = box_area(box_a) + box_area(box_b) - inter_area
    return 0.0 if union <= 0 else inter_area / union


def overlap_ratio(inner: Box, outer: Box) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter = box_area((max(ix1, ox1), max(iy1, oy1), min(ix2, ox2), min(iy2, oy2)))
    area = box_area(inner)
    return 0.0 if area <= 0 else inter / area


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


def bottom_center(box: Box) -> Point:
    x1, _, x2, y2 = box
    return (float((x1 + x2) * 0.5), float(y2))


def point_in_box(point: Point, box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_box(box: Box, scale: float, image_shape: Tuple[int, int, int] | Tuple[int, int]) -> List[float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    new_w = (x2 - x1) * scale
    new_h = (y2 - y1) * scale
    return clip_box((cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0), width, height)


def person_head_region(person_box: Box, image_shape: Tuple[int, int, int] | Tuple[int, int], upper_ratio: float = 0.4) -> List[float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = person_box
    region_h = (y2 - y1) * upper_ratio
    return clip_box((x1, y1, x2, y1 + region_h), width, height)


def draw_box(
    image: np.ndarray,
    box: Box,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    coords = int_box(box)
    if coords is None:
        return
    x1, y1, x2, y2 = coords
    thickness = max(3, int(thickness))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        text_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        ty = max(0, y1 - th - baseline - 4)
        cv2.rectangle(image, (x1, ty), (x1 + tw + 6, ty + th + baseline + 4), color, -1)
        cv2.putText(image, label, (x1 + 3, ty + th + 1), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (10, 24),
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, 0.65, 2)
        cv2.rectangle(image, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
        cv2.putText(image, line, (x, y), font, 0.65, color, 2, cv2.LINE_AA)
        y += th + baseline + 10


def draw_unsafe_banner(image: np.ndarray, violations: Sequence[str]) -> None:
    """Draw a prominent red safety banner without runtime font dependencies."""
    english_labels = {
        "人员未佩戴安全帽": "NO HELMET",
        "叉车司机未佩戴安全帽": "DRIVER NO HELMET",
        "叉车作业半径内有人": "PERSON NEAR FORKLIFT",
        "叉车作业区域未设置有效围蔽": "NO SAFETY FENCE",
        "叉车超速": "FORKLIFT OVERSPEED",
        "检测到安全违规": "SAFETY RULE VIOLATION",
    }
    normalized = []
    for item in violations:
        raw = str(item).strip()
        ascii_label = raw.encode("ascii", errors="ignore").decode("ascii").strip()
        normalized.append(english_labels.get(raw, ascii_label or "SAFETY RULE VIOLATION"))
    unique_violations = list(dict.fromkeys(normalized))
    message = "UNSAFE | VIOLATION: " + "; ".join(unique_violations)
    banner_height = min(image.shape[0], 64)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_height), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(image, message, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def save_json(data: Dict[str, Any], json_path: str | Path) -> None:
    target = Path(json_path)
    ensure_dir(target)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    ensure_dir(output_path)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {output_path}")


def distance(point_a: Point, point_b: Point) -> float:
    return float(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))
