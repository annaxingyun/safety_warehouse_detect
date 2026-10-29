"""Crane boom OBB geometry rules and temporal safety state."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from common import load_model


CRANE_CHECK = "crane_under_boom"
CRANE_CLASS_NAMES = {"boom", "person"}
DEFAULT_CRANE_OPTIONS: dict[str, Any] = {
    "boom_conf": 0.40,
    "person_conf": 0.40,
    "line_margin": 5.0,
    "bottom_margin": 5.0,
    "range_extend_ratio": 0.05,
    "min_axis_horizontal_ratio": 0.05,
    "enter_window": 5,
    "enter_hits": 3,
    "exit_safe_frames": 5,
    "track_ttl_frames": 30,
}


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        value = names.get(class_id, str(class_id))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = str(class_id)
    return str(value).strip().lower()


def validate_crane_model(model: Any) -> None:
    """Fail early when a selected model cannot supply the required OBB classes."""
    task = str(getattr(model, "task", "") or "").lower()
    if task != "obb":
        raise ValueError(f"吊臂模型必须是 OBB 模型，当前模型 task={task or 'unknown'}。")
    names = getattr(model, "names", {})
    available = {_class_name(names, int(class_id)) for class_id in names} if isinstance(names, dict) else {
        str(name).strip().lower() for name in names
    }
    missing = CRANE_CLASS_NAMES - available
    if missing:
        raise ValueError(f"吊臂 OBB 模型缺少类别: {', '.join(sorted(missing))}。")


def load_crane_model(weights: str | Path) -> Any:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"吊臂 OBB 权重文件不存在: {weights_path}")
    model = load_model(weights_path)
    validate_crane_model(model)
    return model


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    return list(tolist()) if callable(tolist) else list(value)


def parse_crane_obb_detections(
    result: Any,
    boom_conf: float = 0.40,
    person_conf: float = 0.40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse tracked Ultralytics OBB results without relying on class-number order."""
    obb = getattr(result, "obb", None) if result is not None else None
    if obb is None or len(obb) == 0:
        return [], []

    obb = obb.cpu()
    points_rows = _as_list(obb.xyxyxyxy)
    box_rows = _as_list(obb.xyxy)
    confidences = _as_list(obb.conf)
    classes = _as_list(obb.cls)
    raw_track_ids = _as_list(obb.id) if getattr(obb, "id", None) is not None else [None] * len(classes)
    names = getattr(result, "names", {})
    booms: list[dict[str, Any]] = []
    persons: list[dict[str, Any]] = []

    for index, (points, box, confidence, class_id, track_id) in enumerate(
        zip(points_rows, box_rows, confidences, classes, raw_track_ids)
    ):
        try:
            class_id = int(class_id)
            confidence = float(confidence)
            name = _class_name(names, class_id)
            threshold = float(boom_conf if name == "boom" else person_conf)
            normalized_points = [[float(point[0]), float(point[1])] for point in points]
            normalized_box = [float(value) for value in box]
            normalized_track_id = (
                int(track_id)
                if track_id is not None and math.isfinite(float(track_id))
                else None
            )
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        if name not in CRANE_CLASS_NAMES or confidence < threshold:
            continue
        if len(normalized_points) != 4 or len(normalized_box) != 4:
            continue
        flat_values = [value for point in normalized_points for value in point] + normalized_box + [confidence]
        if not all(math.isfinite(value) for value in flat_values):
            continue
        detection = {
            "index": index,
            "track_id": normalized_track_id,
            "target_id": normalized_track_id if normalized_track_id is not None else f"{name}_{index}",
            "class_id": class_id,
            "class_name": name,
            "confidence": confidence,
            "points": normalized_points,
            "box": normalized_box,
        }
        (booms if name == "boom" else persons).append(detection)
    return booms, persons


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _order_rectangle_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(array).all():
        raise ValueError("OBB contains non-finite coordinates")
    if len({(round(float(x), 6), round(float(y), 6)) for x, y in array}) != 4:
        raise ValueError("OBB contains duplicate points")
    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    ordered = array[np.argsort(angles)]
    if _polygon_area(ordered) < 1e-3:
        raise ValueError("OBB area is too small")
    crosses = []
    for index in range(4):
        edge_a = ordered[(index + 1) % 4] - ordered[index]
        edge_b = ordered[(index + 2) % 4] - ordered[(index + 1) % 4]
        crosses.append(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
    if any(abs(value) < 1e-6 for value in crosses) or not (
        all(value > 0 for value in crosses) or all(value < 0 for value in crosses)
    ):
        raise ValueError("OBB is not a convex rectangle")
    return ordered


def _point(point: np.ndarray | Sequence[float]) -> list[float]:
    return [float(point[0]), float(point[1])]


def build_boom_geometry(
    points: Sequence[Sequence[float]],
    min_axis_horizontal_ratio: float = 0.05,
) -> dict[str, Any]:
    """Build an order-independent boom geometry description from four OBB vertices."""
    ordered = _order_rectangle_points(points)
    edges = [(ordered[index], ordered[(index + 1) % 4]) for index in range(4)]
    lengths = [float(np.linalg.norm(end - start)) for start, end in edges]
    group_02 = (lengths[0] + lengths[2]) * 0.5
    group_13 = (lengths[1] + lengths[3]) * 0.5
    if max(group_02, group_13) < 1e-6:
        raise ValueError("OBB long edge is too short")
    if group_02 >= group_13:
        long_edges = [edges[0], edges[2]]
        short_edges = [edges[1], edges[3]]
        long_length, short_length = group_02, group_13
    else:
        long_edges = [edges[1], edges[3]]
        short_edges = [edges[0], edges[2]]
        long_length, short_length = group_13, group_02

    lower_edge = max(long_edges, key=lambda edge: float((edge[0][1] + edge[1][1]) * 0.5))
    axis_start = (short_edges[0][0] + short_edges[0][1]) * 0.5
    axis_end = (short_edges[1][0] + short_edges[1][1]) * 0.5
    axis_vector = axis_end - axis_start
    axis_length = float(np.linalg.norm(axis_vector))
    if axis_length < 1e-6:
        raise ValueError("OBB axis is too short")
    axis_horizontal_ratio = abs(float(axis_vector[0])) / axis_length
    if axis_horizontal_ratio < max(0.0, float(min_axis_horizontal_ratio)):
        raise ValueError("boom axis is too close to vertical for a stable 2D below-area rule")
    lowest_index = int(np.argmax(ordered[:, 1]))
    return {
        "points": [_point(point) for point in ordered],
        "lower_edge": [_point(lower_edge[0]), _point(lower_edge[1])],
        "axis": [_point(axis_start), _point(axis_end)],
        "lowest_point": _point(ordered[lowest_index]),
        "lowest_y": float(ordered[lowest_index][1]),
        "long_edge_length": long_length,
        "short_edge_length": short_length,
        "aspect_ratio": long_length / max(short_length, 1e-6),
        "axis_horizontal_ratio": axis_horizontal_ratio,
    }


def _signed_distance(a: Sequence[float], b: Sequence[float], point: Sequence[float]) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        raise ValueError("line edge is too short")
    return (dx * (float(point[1]) - float(a[1])) - dy * (float(point[0]) - float(a[0]))) / length


def evaluate_crane_pair(
    boom: dict[str, Any],
    person: dict[str, Any],
    image_height: int,
    line_margin: float,
    bottom_margin: float,
    range_extend_ratio: float,
) -> dict[str, Any]:
    geometry = boom.get("geometry") or {}
    box = person["box"]
    corners = [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
    ]
    lower_edge = geometry["lower_edge"]
    edge_a, edge_b = lower_edge
    reference = [(float(edge_a[0]) + float(edge_b[0])) * 0.5, float(max(0, image_height - 1))]
    reference_distance = _signed_distance(edge_a, edge_b, reference)
    if abs(reference_distance) < 1e-6:
        raise ValueError("image-bottom reference lies on the boom boundary")
    below_sign = 1.0 if reference_distance > 0 else -1.0
    corner_distances = [below_sign * _signed_distance(edge_a, edge_b, corner) for corner in corners]
    fully_below = all(distance >= float(line_margin) for distance in corner_distances)

    axis_start, axis_end = geometry["axis"]
    axis_x = float(axis_end[0]) - float(axis_start[0])
    axis_y = float(axis_end[1]) - float(axis_start[1])
    axis_length_squared = axis_x * axis_x + axis_y * axis_y
    if axis_length_squared < 1e-6:
        raise ValueError("boom axis is too short")
    person_center = [(float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5]
    projection_t = (
        (person_center[0] - float(axis_start[0])) * axis_x
        + (person_center[1] - float(axis_start[1])) * axis_y
    ) / axis_length_squared
    extend = max(0.0, float(range_extend_ratio))
    in_boom_range = -extend <= projection_t <= 1.0 + extend
    foot_below_lowest = float(box[3]) >= float(geometry["lowest_y"]) + float(bottom_margin)
    raw_danger = fully_below and in_boom_range and foot_below_lowest
    return {
        "boom_id": boom["target_id"],
        "boom_track_id": boom.get("track_id"),
        "person_id": person["target_id"],
        "person_track_id": person.get("track_id"),
        "boom_box": boom["box"],
        "boom_points": geometry["points"],
        "person_box": person["box"],
        "person_confidence": person["confidence"],
        "boom_confidence": boom["confidence"],
        "lower_edge": lower_edge,
        "lowest_y": geometry["lowest_y"],
        "person_bottom_y": float(box[3]),
        "person_center": person_center,
        "corner_signed_distances": corner_distances,
        "projection_t": projection_t,
        "fully_below": fully_below,
        "in_boom_range": in_boom_range,
        "foot_below_lowest": foot_below_lowest,
        "raw_danger": raw_danger,
        "confirmed_danger": raw_danger,
    }


class CraneDangerTemporalFilter:
    """Apply asymmetric entry/exit hysteresis to tracked boom-person pairs."""

    def __init__(
        self,
        window: int = 5,
        enter_hits: int = 3,
        exit_safe_frames: int = 5,
        ttl_frames: int = 30,
    ) -> None:
        self.window = max(1, int(window))
        self.enter_hits = min(self.window, max(1, int(enter_hits)))
        self.exit_safe_frames = max(1, int(exit_safe_frames))
        self.ttl_frames = max(self.exit_safe_frames, int(ttl_frames))
        self.states: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _key(pair: dict[str, Any]) -> tuple[str, str]:
        return str(pair["boom_id"]), str(pair["person_id"])

    def reset(self) -> None:
        self.states.clear()

    def update(self, pairs: list[dict[str, Any]], frame_id: int, advance_missing: bool = True) -> int:
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            key = self._key(pair)
            seen.add(key)
            state = self.states.setdefault(
                key,
                {
                    "history": deque(maxlen=self.window),
                    "dangerous": False,
                    "safe_streak": 0,
                    "last_seen": frame_id,
                },
            )
            raw_danger = bool(pair.get("raw_danger"))
            state["history"].append(raw_danger)
            state["safe_streak"] = 0 if raw_danger else int(state["safe_streak"]) + 1
            state["last_seen"] = frame_id
            if not state["dangerous"] and sum(state["history"]) >= self.enter_hits:
                state["dangerous"] = True
            elif state["dangerous"] and state["safe_streak"] >= self.exit_safe_frames:
                state["dangerous"] = False
            pair["confirmed_danger"] = bool(state["dangerous"])
            pair["temporal_hits"] = int(sum(state["history"]))
            pair["temporal_window_size"] = len(state["history"])
            pair["safe_streak"] = int(state["safe_streak"])

        for key, state in list(self.states.items()):
            if key in seen:
                continue
            age = int(frame_id) - int(state["last_seen"])
            if advance_missing:
                state["history"].append(False)
                state["safe_streak"] = int(state["safe_streak"]) + 1
                if state["dangerous"] and state["safe_streak"] >= self.exit_safe_frames:
                    state["dangerous"] = False
            if age > self.ttl_frames and not state["dangerous"]:
                self.states.pop(key, None)
        return sum(bool(state["dangerous"]) for state in self.states.values())


def analyze_crane_obb_result(
    result: Any,
    image_shape: Sequence[int],
    frame_id: int,
    options: Optional[dict[str, Any]] = None,
    temporal_filter: Optional[CraneDangerTemporalFilter] = None,
    temporal_applied: bool = True,
) -> dict[str, Any]:
    config = dict(DEFAULT_CRANE_OPTIONS)
    config.update(options or {})
    booms, persons = parse_crane_obb_detections(result, config["boom_conf"], config["person_conf"])
    valid_booms: list[dict[str, Any]] = []
    for boom in booms:
        try:
            boom["geometry"] = build_boom_geometry(boom["points"], config["min_axis_horizontal_ratio"])
            boom["geometry_valid"] = True
            valid_booms.append(boom)
        except (TypeError, ValueError) as exc:
            boom["geometry_valid"] = False
            boom["geometry_error"] = str(exc)

    pairs: list[dict[str, Any]] = []
    for boom in valid_booms:
        for person in persons:
            try:
                pairs.append(
                    evaluate_crane_pair(
                        boom,
                        person,
                        int(image_shape[0]),
                        config["line_margin"],
                        config["bottom_margin"],
                        config["range_extend_ratio"],
                    )
                )
            except (TypeError, ValueError) as exc:
                pairs.append(
                    {
                        "boom_id": boom["target_id"],
                        "person_id": person["target_id"],
                        "boom_box": boom["box"],
                        "person_box": person["box"],
                        "raw_danger": False,
                        "confirmed_danger": False,
                        "geometry_error": str(exc),
                    }
                )

    if temporal_applied and temporal_filter is not None:
        active_danger_count = temporal_filter.update(pairs, frame_id, advance_missing=bool(valid_booms))
    else:
        active_danger_count = sum(bool(pair.get("raw_danger")) for pair in pairs)
        for pair in pairs:
            pair["confirmed_danger"] = bool(pair.get("raw_danger"))

    current_danger_pairs = [pair for pair in pairs if pair.get("confirmed_danger")]
    raw_danger_pairs = [pair for pair in pairs if pair.get("raw_danger")]
    if active_danger_count:
        status = "unsafe"
    elif not valid_booms:
        status = "unknown"
    else:
        status = "safe"
    danger_person_ids = list(dict.fromkeys(pair["person_id"] for pair in current_danger_pairs))
    return {
        "result": status,
        "frame_id": int(frame_id),
        "boom_count": len(booms),
        "valid_boom_count": len(valid_booms),
        "person_count": len(persons),
        "raw_danger_pair_count": len(raw_danger_pairs),
        "danger_pair_count": len(current_danger_pairs),
        "active_danger_pair_count": int(active_danger_count),
        "danger_person_count": len(danger_person_ids),
        "danger_person_ids": danger_person_ids,
        "temporal_applied": bool(temporal_applied),
        "parameters": {
            key: config[key]
            for key in (
                "boom_conf",
                "person_conf",
                "line_margin",
                "bottom_margin",
                "range_extend_ratio",
                "min_axis_horizontal_ratio",
                "enter_window",
                "enter_hits",
                "exit_safe_frames",
            )
        },
        "booms": booms,
        "persons": persons,
        "pairs": pairs,
    }


def format_crane_detection_log(result: Any, frame_id: int) -> str:
    obb = getattr(result, "obb", None) if result is not None else None
    count = len(obb) if obb is not None else 0
    lines = [f"[YOLO OBB检测] frame={frame_id} count={count}（置信度过滤及跟踪后）"]
    if not count:
        lines.append("  未检测到 boom/person")
        return "\n".join(lines)
    obb = obb.cpu()
    names = getattr(result, "names", {})
    boxes = _as_list(obb.xyxy)
    confidences = _as_list(obb.conf)
    classes = _as_list(obb.cls)
    ids = _as_list(obb.id) if getattr(obb, "id", None) is not None else [None] * count
    for index, (box, confidence, class_id, track_id) in enumerate(zip(boxes, confidences, classes, ids), start=1):
        class_id = int(class_id)
        coordinates = ", ".join(f"{float(value):.2f}" for value in box)
        lines.append(
            f"  #{index} class={class_id} {_class_name(names, class_id)} "
            f"conf={float(confidence):.4f} track_id={track_id} xyxy=[{coordinates}]"
        )
    return "\n".join(lines)


def draw_crane_result(image: np.ndarray, result: Optional[dict[str, Any]]) -> np.ndarray:
    if not result:
        return image
    dangerous_person_ids = {
        str(pair.get("person_id")) for pair in result.get("pairs", []) if pair.get("confirmed_danger")
    }
    for boom in result.get("booms", []):
        geometry = boom.get("geometry") or {}
        points = geometry.get("points", boom.get("points", []))
        if len(points) == 4:
            polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            color = (0, 190, 0) if boom.get("geometry_valid") else (0, 165, 255)
            cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
        lower_edge = geometry.get("lower_edge") or []
        if len(lower_edge) == 2:
            cv2.line(image, tuple(map(int, lower_edge[0])), tuple(map(int, lower_edge[1])), (255, 80, 0), 4, cv2.LINE_AA)
        axis = geometry.get("axis") or []
        if len(axis) == 2:
            cv2.line(image, tuple(map(int, axis[0])), tuple(map(int, axis[1])), (255, 220, 0), 2, cv2.LINE_AA)
        lowest_point = geometry.get("lowest_point")
        if isinstance(lowest_point, list) and len(lowest_point) == 2:
            cv2.circle(image, tuple(map(int, lowest_point)), 6, (0, 255, 255), -1, cv2.LINE_AA)

    for person in result.get("persons", []):
        box = person.get("box") or []
        if len(box) != 4:
            continue
        dangerous = str(person.get("target_id")) in dangerous_person_ids
        color = (0, 0, 255) if dangerous else (0, 190, 0)
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4 if dangerous else 2, cv2.LINE_AA)
        label = "UNSAFE: UNDER BOOM" if dangerous else "PERSON"
        cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.circle(image, (int(round((x1 + x2) * 0.5)), y2), 5, (0, 255, 255), -1, cv2.LINE_AA)
    return image
