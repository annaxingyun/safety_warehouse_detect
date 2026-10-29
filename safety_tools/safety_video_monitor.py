"""Video and image-sequence safety monitoring with YOLO detection and BoT-SORT tracking."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np

from common import (
    HazardIncidentTracker,
    INFERENCE_IMAGE_SIZE,
    NMS_IOU,
    MAX_HAZARD_EVENTS_IN_MEMORY,
    add_finished_at,
    box_iou,
    draw_box,
    draw_raw_detection_boxes,
    draw_text_lines,
    draw_unsafe_banner,
    filter_detections,
    load_model,
    parse_detections,
    validate_model_classes,
    yolo_detection_logging,
    resolve_device,
    save_hazard_image,
    save_json,
)
from crane_safety_check import (
    CRANE_CHECK,
    DEFAULT_CRANE_OPTIONS,
    CraneDangerTemporalFilter,
    analyze_crane_obb_result,
    draw_crane_result,
    validate_crane_model,
)
from forklift_speed_check import (
    REFERENCE_SPEED_KMH,
    UNSAFE_SPEED_KMH,
    estimate_motion,
    speed_calibration_info,
    update_speeds,
)
from forklift_work_check import analyze_forklift_work_frame, parse_args as parse_work_args
from helmet_check import evaluate_persons
from material_count_cli import VIDEO_FALLBACK_FPS, get_image_files
from live_stream import LiveStreamReader, acquire_shared_live_reader, release_shared_live_reader
from lifting_operation_check import (
    LIFTING_CHECK,
    LIFTING_CLASS_NAMES,
    LIFTING_VIOLATIONS,
    analyze_lifting_operation,
)
from model_config import normalize_class_name


TRACKER_CONFIG = "botsort.yaml"
SAFETY_CHECK_ORDER = (
    "person_helmet",
    "driver_helmet",
    "forklift_proximity",
    "forklift_fence",
    "forklift_speed",
    CRANE_CHECK,
    LIFTING_CHECK,
)
WORK_CHECKS = {"driver_helmet", "forklift_proximity", "forklift_fence"}
CHECK_CLASS_REQUIREMENTS = {
    "person_helmet": {"person", "helmet", "head_no_helmet"},
    "driver_helmet": {"person", "helmet", "head_no_helmet", "forklift"},
    "forklift_proximity": {"person", "forklift"},
    "forklift_fence": {"person", "forklift", "fence"},
    "forklift_speed": {"forklift"},
    CRANE_CHECK: {"person", "boom"},
    LIFTING_CHECK: LIFTING_CLASS_NAMES,
}
HELMET_COLORS = {
    "helmet": (0, 180, 0),
    "no_helmet": (0, 0, 255),
    "unknown": (0, 190, 255),
}

HAZARD_MESSAGES = {
    "NO_HELMET": "人员未佩戴安全帽",
    "DRIVER_NO_HELMET": "叉车司机未佩戴安全帽",
    "PERSON_NEAR_FORKLIFT": "叉车作业半径内有人",
    "NO_FENCE_NEAR_FORKLIFT": "叉车作业区域未设置有效围蔽",
    "FORKLIFT_OVERSPEED": "叉车超速",
    "PERSON_UNDER_BOOM": "人员进入吊臂下方危险区域",
    "FORKLIFT_WORK_UNSAFE": "叉车作业存在安全隐患",
    **LIFTING_VIOLATIONS,
}

CHECK_VIOLATION_LABELS = {
    "person_helmet": "人员未佩戴安全帽",
    "driver_helmet": "叉车司机未佩戴安全帽",
    "forklift_proximity": "叉车作业半径内有人",
    "forklift_fence": "叉车作业区域未设置有效围蔽",
    "forklift_speed": "叉车超速",
    CRANE_CHECK: "吊臂下方有人",
    LIFTING_CHECK: "吊装作业缺少围栏或安全监护人员",
}


def unsafe_target_boxes(frame_result: dict[str, Any]) -> list[tuple[list[float], str]]:
    """Return de-duplicated unsafe target boxes for all unified safety checks."""
    targets: dict[tuple[int, int, int, int], tuple[list[float], str]] = {}

    def add(box: Any, label: str) -> None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return
        normalized = [float(value) for value in box]
        key = tuple(int(round(value)) for value in normalized)
        targets[key] = (normalized, label)

    results = frame_result.get("results") or {}
    for person in (results.get("person_helmet") or {}).get("persons", []) or []:
        if person.get("status") == "no_helmet":
            add(person.get("box"), "UNSAFE: NO HELMET")
    for driver in (results.get("driver_helmet") or {}).get("drivers", []) or []:
        if driver.get("status") == "no_helmet":
            add(driver.get("box"), "UNSAFE: DRIVER NO HELMET")
    for forklift in (results.get("forklift_proximity") or {}).get("forklifts", []) or []:
        if forklift.get("has_person_nearby"):
            add(forklift.get("forklift_box"), "UNSAFE: PERSON NEAR FORKLIFT")
            for person in forklift.get("nearby_persons", []) or []:
                add(person.get("box"), "UNSAFE: PERSON NEAR FORKLIFT")
    for forklift in (results.get("forklift_fence") or {}).get("forklifts", []) or []:
        if forklift.get("fence_check_applicable", True) and not forklift.get("has_fence_nearby"):
            add(forklift.get("forklift_box"), "UNSAFE: NO FENCE")
    for forklift in (results.get("forklift_speed") or {}).get("forklifts", []) or []:
        if forklift.get("overspeed"):
            add(forklift.get("box"), "UNSAFE: OVERSPEED")
    for pair in (results.get(CRANE_CHECK) or {}).get("pairs", []) or []:
        if pair.get("confirmed_danger"):
            add(pair.get("person_box"), "UNSAFE: UNDER BOOM")
    for crane in (results.get(LIFTING_CHECK) or {}).get("cranes", []) or []:
        if crane.get("result") == "unsafe":
            missing = []
            if not crane.get("has_safety_officer"):
                missing.append("NO SAFETY OFFICER")
            if not crane.get("has_fence"):
                missing.append("NO FENCE")
            add(crane.get("box"), f"UNSAFE: {' / '.join(missing)}")
    return list(targets.values())


def _tracked_target_id(item: dict[str, Any], fallback_type: str) -> str:
    for key in ("track_id", "person_id", "forklift_id", "crane_id"):
        if item.get(key) is not None:
            return str(item[key])
    box = item.get("box", item.get("forklift_box"))
    if isinstance(box, list) and box:
        return f"untracked_box_{'_'.join(str(int(value)) for value in box)}"
    return f"untracked_{fallback_type}"


def _hazard_instance(
    code: str,
    check: str,
    key_parts: tuple[Any, ...],
    details: Any,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "code": code,
        "check": check,
        "message": HAZARD_MESSAGES.get(code, code),
        "count": 1,
        "details": details,
        "targets": targets,
        "_incident_key": (code, *key_parts),
    }


def build_hazard_instances(algorithm: str, frame_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one tracked violation instance per unsafe target/rule combination."""
    instances: list[dict[str, Any]] = []
    if algorithm == "helmet":
        for person in frame_result.get("persons", []) or []:
            if person.get("status") != "no_helmet":
                continue
            person_id = _tracked_target_id(person, "person")
            instances.append(_hazard_instance(
                "NO_HELMET", "person_helmet", ("person", person_id), [person],
                [{"type": "person", "track_id": person.get("track_id"), "target_id": person_id}],
            ))
        return instances
    if algorithm == "forklift_speed":
        for forklift in frame_result.get("forklifts", []) or []:
            if not forklift.get("overspeed"):
                continue
            forklift_id = _tracked_target_id(forklift, "forklift")
            instances.append(_hazard_instance(
                "FORKLIFT_OVERSPEED", "forklift_speed", ("forklift", forklift_id), [forklift],
                [{"type": "forklift", "track_id": forklift.get("track_id"), "target_id": forklift_id}],
            ))
        return instances
    if algorithm == "forklift_work":
        for person in frame_result.get("persons", []) or []:
            if person.get("status") == "no_helmet":
                person_id = _tracked_target_id(person, "person")
                instances.append(_hazard_instance(
                    "NO_HELMET", "person_helmet", ("person", person_id), [person],
                    [{"type": "person", "track_id": person.get("track_id"), "target_id": person_id}],
                ))
        for forklift in frame_result.get("forklifts", []) or []:
            forklift_id = _tracked_target_id(forklift, "forklift")
            if forklift.get("fence_check_applicable", True) and not forklift.get("has_fence_nearby"):
                instances.append(_hazard_instance(
                    "NO_FENCE_NEAR_FORKLIFT", "forklift_fence", ("forklift", forklift_id), [forklift],
                    [{"type": "forklift", "track_id": forklift.get("forklift_id"), "target_id": forklift_id}],
                ))
            for person in forklift.get("nearby_persons", []) or []:
                person_id = _tracked_target_id(person, "person")
                instances.append(_hazard_instance(
                    "PERSON_NEAR_FORKLIFT", "forklift_proximity",
                    ("forklift", forklift_id, "person", person_id), [person],
                    [
                        {"type": "forklift", "track_id": forklift.get("forklift_id"), "target_id": forklift_id},
                        {"type": "person", "track_id": person.get("track_id"), "target_id": person_id},
                    ],
                ))
        return instances

    results = frame_result.get("results", {}) if isinstance(frame_result.get("results"), dict) else {}
    person_helmet = results.get("person_helmet", {})
    for person in person_helmet.get("persons", []) or []:
        if person.get("status") == "no_helmet":
            person_id = _tracked_target_id(person, "person")
            instances.append(_hazard_instance(
                "NO_HELMET", "person_helmet", ("person", person_id), [person],
                [{"type": "person", "track_id": person.get("track_id"), "target_id": person_id}],
            ))
    driver_helmet = results.get("driver_helmet", {})
    for driver in driver_helmet.get("drivers", []) or []:
        if driver.get("status") == "no_helmet":
            driver_id = _tracked_target_id(driver, "driver")
            forklift_id = str(driver.get("forklift_id", "unknown_forklift"))
            instances.append(_hazard_instance(
                "DRIVER_NO_HELMET", "driver_helmet", ("forklift", forklift_id, "person", driver_id), [driver],
                [
                    {"type": "forklift", "track_id": driver.get("forklift_id"), "target_id": forklift_id},
                    {"type": "person", "track_id": driver.get("track_id"), "target_id": driver_id},
                ],
            ))
    proximity = results.get("forklift_proximity", {})
    for forklift in proximity.get("forklifts", []) or []:
        forklift_id = _tracked_target_id(forklift, "forklift")
        for person in forklift.get("nearby_persons", []) or []:
            person_id = _tracked_target_id(person, "person")
            instances.append(_hazard_instance(
                "PERSON_NEAR_FORKLIFT", "forklift_proximity",
                ("forklift", forklift_id, "person", person_id), [person],
                [
                    {"type": "forklift", "track_id": forklift.get("forklift_id"), "target_id": forklift_id},
                    {"type": "person", "track_id": person.get("track_id"), "target_id": person_id},
                ],
            ))
    fence = results.get("forklift_fence", {})
    for forklift in fence.get("forklifts", []) or []:
        if forklift.get("fence_check_applicable", True) and not forklift.get("has_fence_nearby"):
            forklift_id = _tracked_target_id(forklift, "forklift")
            instances.append(_hazard_instance(
                "NO_FENCE_NEAR_FORKLIFT", "forklift_fence", ("forklift", forklift_id), [forklift],
                [{"type": "forklift", "track_id": forklift.get("forklift_id"), "target_id": forklift_id}],
            ))
    speed = results.get("forklift_speed", {})
    for forklift in speed.get("forklifts", []) or []:
        if forklift.get("overspeed"):
            forklift_id = _tracked_target_id(forklift, "forklift")
            instances.append(_hazard_instance(
                "FORKLIFT_OVERSPEED", "forklift_speed", ("forklift", forklift_id), [forklift],
                [{"type": "forklift", "track_id": forklift.get("track_id"), "target_id": forklift_id}],
            ))
    crane = results.get(CRANE_CHECK, {})
    for pair in crane.get("pairs", []) or []:
        if not pair.get("confirmed_danger"):
            continue
        boom_id = str(pair.get("boom_id", "unknown_boom"))
        person_id = str(pair.get("person_id", "unknown_person"))
        instances.append(_hazard_instance(
            "PERSON_UNDER_BOOM", CRANE_CHECK, ("boom", boom_id, "person", person_id), [pair],
            [
                {"type": "boom", "track_id": pair.get("boom_track_id"), "target_id": boom_id},
                {"type": "person", "track_id": pair.get("person_track_id"), "target_id": person_id},
            ],
        ))
    for crane in (results.get(LIFTING_CHECK) or {}).get("cranes", []) or []:
        crane_id = _tracked_target_id(crane, "crane")
        for code in crane.get("violations", []) or []:
            instances.append(_hazard_instance(
                code, LIFTING_CHECK, ("crane", crane_id), [crane],
                [{"type": "crane", "track_id": crane.get("track_id"), "target_id": crane_id}],
            ))
    return instances


def attach_hazard_event(
    frame_result: dict[str, Any],
    annotated_frame: np.ndarray,
    output_hint: Optional[Path],
    algorithm: str,
    hazards: list[dict[str, Any]],
    is_first_hazard: bool,
    event_sequence: int,
) -> bool:
    """Persist one image for the alert frame and attach all new violations to it."""
    if not hazards:
        return False
    evidence_label = (
        f"{algorithm}_{hazards[0].get('code', 'hazard')}"
        if len(hazards) == 1
        else f"{algorithm}_event"
    )
    evidence = save_hazard_image(
        annotated_frame,
        output_hint,
        evidence_label,
        int(frame_result.get("frame_id", 0)),
    )
    persisted_hazards: list[dict[str, Any]] = []
    for hazard in hazards:
        item = dict(hazard)
        item.update(
            hazard_timestamp=evidence["hazard_timestamp"],
            hazard_image_path=evidence["hazard_image_path"],
        )
        persisted_hazards.append(item)
    event = {
        "event_id": f"{algorithm}_event_{int(event_sequence)}_frame_{int(frame_result.get('frame_id', 0))}",
        "event_sequence": int(event_sequence),
        "algorithm": algorithm,
        "source_type": frame_result.get("source_type"),
        "frame_id": int(frame_result.get("frame_id", 0)),
        "is_first_hazard": bool(is_first_hazard),
        "hazard_timestamp": evidence["hazard_timestamp"],
        "hazard_image_path": evidence["hazard_image_path"],
        "hazard_image_paths": [evidence["hazard_image_path"]],
        "hazard_count": len(persisted_hazards),
        "hazards": persisted_hazards,
    }
    frame_result.update(
        hazard_timestamp=event["hazard_timestamp"],
        hazard_image_path=event["hazard_image_path"],
        hazard_image_paths=event["hazard_image_paths"],
        hazard_count=event["hazard_count"],
        hazards=persisted_hazards,
        is_first_hazard=bool(is_first_hazard),
        hazard_event=event,
    )
    return True


def _reset_model_trackers(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def _track_frame(model, frame: np.ndarray, tracker: str, conf: float, iou: float, device: str, classes=None):
    kwargs: dict[str, Any] = {
        "source": frame,
        "tracker": tracker,
        "persist": True,
        "conf": conf,
        "iou": iou,
        "imgsz": INFERENCE_IMAGE_SIZE,
        "end2end": False,
        "agnostic_nms": False,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    if classes is not None:
        kwargs["classes"] = classes
    results = model.track(**kwargs)
    return results[0] if results else None


class HelmetTrackSmoother:
    """Confirm no-helmet state after consecutive processed frames of the same track."""

    def __init__(self, window: int = 5, ttl_frames: int = 30):
        self.window = max(1, int(window))
        self.ttl_frames = max(self.window, int(ttl_frames))
        self.history: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=self.window))
        self.last_seen: dict[int, int] = {}
        self.last_frame_id: Optional[int] = None

    def update(self, persons: list[dict[str, Any]], frame_id: int) -> None:
        for person in persons:
            raw_status = str(person.get("status", "unknown"))
            person["raw_status"] = raw_status
            track_id = person.get("track_id")
            if track_id is None:
                # An untracked detection cannot establish continuity across frames.
                person["status"] = "unknown" if raw_status == "no_helmet" and self.window > 1 else raw_status
                continue
            track_id = int(track_id)
            if self.last_seen.get(track_id) not in {self.last_frame_id, frame_id}:
                self.history[track_id].clear()
            if self.last_seen.get(track_id) != frame_id:
                self.history[track_id].append(raw_status)
            self.last_seen[track_id] = frame_id
            if raw_status == "no_helmet":
                history = self.history[track_id]
                confirmed = len(history) == self.window and all(value == "no_helmet" for value in history)
                person["status"] = "no_helmet" if confirmed else "unknown"
        self.last_frame_id = frame_id
        expired = [track_id for track_id, seen in self.last_seen.items() if frame_id - seen > self.ttl_frames]
        for track_id in expired:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)


def analyze_helmet_frame(
    model,
    frame: np.ndarray,
    frame_id: int,
    conf: float,
    iou: float,
    device: str,
    tracker: str,
    smoother: HelmetTrackSmoother,
    no_helmet_overlap: float = 0.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = _track_frame(model, frame, tracker, conf, iou, device)
    # Framework NMS runs before tracking and safety evaluation.
    detections = parse_detections(result)
    persons = evaluate_persons(detections, frame.shape, no_helmet_overlap)
    smoother.update(persons, frame_id)

    annotated = draw_raw_detection_boxes(frame, detections)
    public_persons: list[dict[str, Any]] = []
    for person in persons:
        status = str(person.get("status", "unknown"))
        item = dict(person)
        item.pop("_head_region", None)
        public_persons.append(item)

    total = len(public_persons)
    helmet_count = sum(1 for person in public_persons if person["status"] == "helmet")
    no_helmet_count = sum(1 for person in public_persons if person["status"] == "no_helmet")
    unknown_count = total - helmet_count - no_helmet_count
    frame_result = {
        "frame_id": frame_id,
        "total_persons": total,
        "person_count": total,
        "helmet_worn_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "unknown_count": unknown_count,
        "persons": public_persons,
        "result": "unsafe" if no_helmet_count else ("unknown" if not total or unknown_count else "safe"),
    }
    for person in public_persons:
        if person.get("status") == "no_helmet":
            draw_box(annotated, person.get("box"), "UNSAFE: NO HELMET", (0, 0, 255), 5)
    return annotated, frame_result


def make_work_args(weights: str, conf: float, device: str, options: dict[str, Any]):
    args = parse_work_args(["--weights", weights, "--image", "frame", "--output", "frame.jpg"])
    args.conf = float(conf)
    args.device = resolve_device(device)
    for name, value in options.items():
        if hasattr(args, name):
            setattr(args, name, value)
    return args


def _public_person(person: dict[str, Any]) -> dict[str, Any]:
    item = dict(person)
    item.pop("_head_region", None)
    return item


def _result_from_statuses(statuses: list[str]) -> str:
    if any(status == "no_helmet" for status in statuses):
        return "unsafe"
    if not statuses or any(status == "unknown" for status in statuses):
        return "unknown"
    return "safe"


def build_selected_frame_result(
    enabled_checks: set[str],
    frame_id: int,
    persons: list[dict[str, Any]],
    work_result: Optional[dict[str, Any]],
    speed_result: Optional[dict[str, Any]],
    crane_result: Optional[dict[str, Any]] = None,
    lifting_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Expose only selected monitoring capabilities, even when internal dependency data was calculated."""
    selected = [name for name in SAFETY_CHECK_ORDER if name in enabled_checks]
    results: dict[str, Any] = {}
    driver_by_box: dict[tuple[int, ...], dict[str, Any]] = {}
    if work_result:
        for forklift in work_result.get("forklifts", []) or []:
            for driver in forklift.get("excluded_persons", []) or []:
                key = tuple(driver.get("box") or [])
                if key:
                    driver_by_box[key] = {**driver, "forklift_id": forklift.get("forklift_id")}

    if "person_helmet" in enabled_checks:
        public_persons = [_public_person(person) for person in persons if tuple(person.get("box") or []) not in driver_by_box]
        statuses = [str(person.get("status", "unknown")) for person in public_persons]
        results["person_helmet"] = {
            "result": _result_from_statuses(statuses),
            "total_persons": len(public_persons),
            "helmet_worn_count": statuses.count("helmet"),
            "no_helmet_count": statuses.count("no_helmet"),
            "unknown_count": statuses.count("unknown"),
            "persons": public_persons,
        }

    if "driver_helmet" in enabled_checks:
        drivers = []
        for driver in driver_by_box.values():
            item = dict(driver)
            item["status"] = str(item.pop("helmet_status", item.get("status", "unknown")))
            drivers.append(item)
        statuses = [str(driver.get("status", "unknown")) for driver in drivers]
        results["driver_helmet"] = {
            "result": _result_from_statuses(statuses),
            "driver_count": len(drivers),
            "helmet_worn_count": statuses.count("helmet"),
            "no_helmet_count": statuses.count("no_helmet"),
            "unknown_count": statuses.count("unknown"),
            "drivers": drivers,
        }

    if "forklift_proximity" in enabled_checks:
        forklifts = []
        for forklift in (work_result or {}).get("forklifts", []) or []:
            forklifts.append(
                {
                    "forklift_id": forklift.get("forklift_id"),
                    "forklift_box": forklift.get("forklift_box"),
                    "has_person_nearby": bool(forklift.get("has_person_nearby")),
                    "nearby_persons": forklift.get("nearby_persons", []),
                    "work_zone": forklift.get("work_zone"),
                    "circle_work_zone": forklift.get("circle_work_zone"),
                }
            )
        has_nearby = any(item["has_person_nearby"] for item in forklifts)
        results["forklift_proximity"] = {
            "result": "unknown" if not forklifts else ("unsafe" if has_nearby else "safe"),
            "forklift_count": len(forklifts),
            "has_person_nearby": has_nearby,
            "nearby_person_count": sum(len(item["nearby_persons"]) for item in forklifts),
            "forklifts": forklifts,
        }

    if "forklift_fence" in enabled_checks:
        forklifts = []
        movement_threshold = float((speed_result or {}).get("movement_threshold_kmh", 1.0))
        speed_items = list((speed_result or {}).get("forklifts", []) or [])
        matched_speed_indexes: set[int] = set()

        def match_speed_item(forklift: dict[str, Any]) -> Optional[dict[str, Any]]:
            """Associate every work-result forklift with its own speed result.

            Both results originate from the same frame, so box overlap is more
            reliable than an ID alone when a tracker temporarily drops or
            reassigns IDs in a multi-forklift scene. Each speed item can be used
            only once.
            """
            forklift_box = forklift.get("forklift_box")
            best_index: Optional[int] = None
            best_iou = 0.0
            if isinstance(forklift_box, (list, tuple)) and len(forklift_box) == 4:
                for index, item in enumerate(speed_items):
                    if index in matched_speed_indexes:
                        continue
                    speed_box = item.get("box")
                    if not isinstance(speed_box, (list, tuple)) or len(speed_box) != 4:
                        continue
                    overlap = box_iou(forklift_box, speed_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_index = index
            if best_index is None or best_iou < 0.5:
                forklift_id = forklift.get("forklift_id")
                best_index = next(
                    (
                        index
                        for index, item in enumerate(speed_items)
                        if index not in matched_speed_indexes and item.get("track_id") == forklift_id
                    ),
                    None,
                )
            if best_index is None:
                return None
            matched_speed_indexes.add(best_index)
            return speed_items[best_index]

        for forklift in (work_result or {}).get("forklifts", []) or []:
            speed_item = match_speed_item(forklift)
            movement_speed = speed_item.get("smoothed_speed_kmh") if speed_item else None
            movement_known = speed_result is None or movement_speed is not None
            is_moving = (
                None
                if not movement_known
                else bool(movement_speed is not None and float(movement_speed) > movement_threshold)
            )
            fence_check_applicable = speed_result is None or is_moving is False
            forklifts.append(
                {
                    "forklift_id": forklift.get("forklift_id"),
                    "forklift_box": forklift.get("forklift_box"),
                    "has_fence_nearby": bool(forklift.get("has_fence_nearby")),
                    "nearby_fences": forklift.get("nearby_fences", []),
                    "is_moving": is_moving,
                    "fence_check_applicable": fence_check_applicable,
                    "fence_check_skip_reason": (
                        "forklift_moving"
                        if is_moving is True
                        else ("movement_unknown" if not fence_check_applicable else None)
                    ),
                }
            )
        checked_forklifts = [item for item in forklifts if item["fence_check_applicable"]]
        all_fenced = bool(checked_forklifts) and all(item["has_fence_nearby"] for item in checked_forklifts)
        results["forklift_fence"] = {
            "result": "unknown" if not checked_forklifts else ("safe" if all_fenced else "unsafe"),
            "forklift_count": len(forklifts),
            "fence_count": int((work_result or {}).get("fence_count", 0) or 0),
            "all_forklifts_fenced": all_fenced,
            "checked_forklift_count": len(checked_forklifts),
            "moving_forklift_count": sum(item["is_moving"] is True for item in forklifts),
            "movement_unknown_forklift_count": sum(item["is_moving"] is None for item in forklifts),
            "movement_threshold_kmh": movement_threshold,
            "forklifts": forklifts,
        }

    if "forklift_speed" in enabled_checks:
        results["forklift_speed"] = dict(speed_result or {"result": "unknown", "forklifts": []})

    if CRANE_CHECK in enabled_checks:
        results[CRANE_CHECK] = dict(
            crane_result
            or {
                "result": "unknown",
                "boom_count": 0,
                "person_count": 0,
                "danger_person_count": 0,
                "pairs": [],
            }
        )

    if LIFTING_CHECK in enabled_checks:
        results[LIFTING_CHECK] = dict(lifting_result or analyze_lifting_operation([]))

    statuses = [str(item.get("result", "unknown")) for item in results.values()]
    overall = "unsafe" if "unsafe" in statuses else ("safe" if statuses and all(x == "safe" for x in statuses) else "unknown")
    payload: dict[str, Any] = {
        "frame_id": frame_id,
        "selected_checks": selected,
        "results": results,
        "result": overall,
    }
    helmet = results.get("person_helmet", {})
    driver = results.get("driver_helmet", {})
    proximity = results.get("forklift_proximity", {})
    fence = results.get("forklift_fence", {})
    speed = results.get("forklift_speed", {})
    crane = results.get(CRANE_CHECK, {})
    payload.update(
        total_persons=helmet.get("total_persons"),
        helmet_worn_count=helmet.get("helmet_worn_count"),
        no_helmet_count=helmet.get("no_helmet_count"),
        driver_count=driver.get("driver_count"),
        driver_no_helmet_count=driver.get("no_helmet_count"),
        forklift_count=max(
            int(proximity.get("forklift_count", 0) or 0),
            int(fence.get("forklift_count", 0) or 0),
            int(speed.get("forklift_count", 0) or 0),
        ),
        current_speed_px_per_frame=speed.get("current_speed_px_per_frame"),
        current_speed_kmh=speed.get("current_speed_kmh"),
        smoothed_speed_px_per_frame=speed.get("smoothed_speed_px_per_frame"),
        smoothed_speed_kmh=speed.get("smoothed_speed_kmh"),
        speed_limit_kmh=speed.get("speed_limit_kmh"),
        overspeed_frame_count=speed.get("overspeed_frame_count"),
        speed_status=speed.get("speed_status"),
        crane_boom_count=crane.get("boom_count"),
        crane_person_count=crane.get("person_count"),
        crane_danger_person_count=crane.get("danger_person_count"),
    )
    return {key: value for key, value in payload.items() if value is not None}


def draw_selected_frame(
    frame: np.ndarray,
    enabled_checks: set[str],
    frame_result: dict[str, Any],
    detections: list[dict[str, Any]],
    draw_circle_work_zone: bool = True,
    crane_result: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Draw enlarged detection boxes and an explicit unsafe banner when needed."""
    annotated = frame.copy()
    if "forklift_proximity" in enabled_checks and draw_circle_work_zone:
        for forklift in ((frame_result.get("results") or {}).get("forklift_proximity") or {}).get("forklifts", []) or []:
            circle = forklift.get("circle_work_zone") or {}
            polygon = circle.get("polygon") or []
            if not circle.get("enabled") or len(polygon) < 3:
                continue
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            has_person_nearby = bool(forklift.get("has_person_nearby"))
            fill_color = (0, 80, 255) if has_person_nearby else (0, 210, 255)
            line_color = (0, 0, 255) if has_person_nearby else (0, 140, 255)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [points], fill_color)
            cv2.addWeighted(overlay, 0.24, annotated, 0.76, 0, annotated)
            cv2.polylines(annotated, [points], True, line_color, 4, cv2.LINE_AA)
    annotated = draw_raw_detection_boxes(annotated, detections, thickness=3)
    if CRANE_CHECK in enabled_checks:
        annotated = draw_crane_result(annotated, crane_result)
    for crane in ((frame_result.get("results") or {}).get(LIFTING_CHECK) or {}).get("cranes", []):
        if crane.get("result") == "safe":
            draw_box(annotated, crane["box"], "LIFTING: OK", (0, 180, 0), 3)
    for box, label in unsafe_target_boxes(frame_result):
        draw_box(annotated, box, label, (0, 0, 255), 5)
    if frame_result.get("result") == "unsafe":
        violations = [
            (
                "、".join(dict.fromkeys(
                    LIFTING_VIOLATIONS[code]
                    for crane in item.get("cranes", [])
                    for code in crane.get("violations", [])
                ))
                if name == LIFTING_CHECK else CHECK_VIOLATION_LABELS.get(name, name)
            )
            for name, item in (frame_result.get("results") or {}).items()
            if item.get("result") == "unsafe"
        ]
        draw_unsafe_banner(annotated, violations or ["检测到安全违规"])
    return annotated


def run_unified_safety_monitor(
    enabled_checks: set[str],
    source_type: str,
    source_path: Union[Path, str],
    weights_path: str,
    output_media: Optional[Path],
    conf: float = 0.4,
    iou: float = NMS_IOU,
    device: str = "cpu",
    tracker: str = TRACKER_CONFIG,
    work_options: Optional[dict[str, Any]] = None,
    speed_options: Optional[dict[str, Any]] = None,
    crane_options: Optional[dict[str, Any]] = None,
    preview_callback: Optional[Callable[[int, np.ndarray, dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    sample_interval: int = 5,
    raw_preview_callback: Optional[Callable[[int, np.ndarray, dict[str, Any]], None]] = None,
    detection_log_callback: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    enabled_checks = set(enabled_checks)
    invalid = enabled_checks.difference(SAFETY_CHECK_ORDER)
    if invalid or not enabled_checks:
        raise ValueError(f"invalid or empty safety checks: {sorted(invalid)}")
    if source_type not in {"single_image", "image_sequence", "video", "live_stream"}:
        raise ValueError(f"unsupported source type: {source_type}")
    if source_type == "single_image" and "forklift_speed" in enabled_checks:
        raise ValueError("叉车速度检测需要视频、直播流或连续图片序列，不能使用单张图片。")

    sample_interval = max(1, int(sample_interval))
    source_path = str(source_path) if source_type == "live_stream" else Path(source_path)
    model = load_model(weights_path)
    required_classes = set().union(*(CHECK_CLASS_REQUIREMENTS[check] for check in enabled_checks))
    target_class_ids = validate_model_classes(model, required_classes)
    if CRANE_CHECK in enabled_checks:
        validate_crane_model(model)
    device = resolve_device(device)
    smoother = HelmetTrackSmoother(window=1 if source_type == "single_image" else 5)
    crane_config = dict(DEFAULT_CRANE_OPTIONS)
    crane_config.update(crane_options or {})
    thresholds = [float(conf)] if enabled_checks - {CRANE_CHECK} else []
    if CRANE_CHECK in enabled_checks:
        thresholds.extend([float(crane_config["boom_conf"]), float(crane_config["person_conf"])])
    inference_conf = min(thresholds)
    box_class_names = {"person", "helmet", "head_no_helmet", "forklift", "fence"}
    if LIFTING_CHECK in enabled_checks:
        box_class_names.update(LIFTING_CLASS_NAMES)
    crane_temporal = CraneDangerTemporalFilter(
        window=crane_config["enter_window"],
        enter_hits=crane_config["enter_hits"],
        exit_safe_frames=crane_config["exit_safe_frames"],
        ttl_frames=crane_config["track_ttl_frames"],
    )
    work_options = dict(work_options or {})
    speed_options = dict(speed_options or {})
    camera_mode = str(work_options.get("camera_mode", speed_options.get("camera_mode", "oblique")))
    if camera_mode not in {"oblique", "overhead"}:
        raise ValueError("camera_mode must be one of oblique, overhead")
    work_options["camera_mode"] = camera_mode
    speed_options["camera_mode"] = camera_mode
    work_args = make_work_args(weights_path, conf, device, work_options) if enabled_checks.intersection(WORK_CHECKS) else None
    speed_limit = float(speed_options.get("speed_limit", REFERENCE_SPEED_KMH))
    unsafe_speed_limit = float(speed_options.get("unsafe_speed_limit", UNSAFE_SPEED_KMH))
    movement_threshold_kmh = max(0.0, float(speed_options.get("movement_threshold", 1.0)))
    speed_window = max(1, int(speed_options.get("speed_window", 30)))
    min_bg_matches = max(1, int(speed_options.get("min_bg_matches", 20)))
    motion_model = str(speed_options.get("motion_model", "affine"))

    cap: Optional[cv2.VideoCapture] = None
    live_reader: Optional[LiveStreamReader] = None
    live_frame_callback: Optional[Callable[[Any], None]] = None
    image_files: list[Path] = []
    fps = VIDEO_FALLBACK_FPS
    total_frames = 1
    if source_type == "single_image":
        image_files = [source_path]
    elif source_type == "image_sequence":
        image_files = get_image_files(source_path)
        if not image_files:
            raise ValueError("输入文件夹中没有可用图片。")
        total_frames = len(image_files)
        fps = 10.0
    elif source_type == "video":
        if not isinstance(source_path, Path) or not source_path.is_file():
            raise ValueError("视频模式下，输入必须是视频文件。")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {source_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or VIDEO_FALLBACK_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    else:
        def on_raw(packet) -> None:
            if raw_preview_callback is not None:
                raw_preview_callback(packet.frame_id, packet.frame, {"fps": packet.fps, "generation": packet.generation})
        live_frame_callback = on_raw
        live_reader = acquire_shared_live_reader(str(source_path), frame_callback=live_frame_callback)
        total_frames = 0

    processed = 0
    source_frame_id = 0
    last_seen_source_id = -1
    previous_processed_source_id: Optional[int] = None
    live_generation: Optional[int] = None
    input_size: Optional[tuple[int, int]] = None
    stopped = False
    unsafe_frames = 0
    latest_result: dict[str, Any] = {}
    worst_result: dict[str, Any] = {}
    hazard_events: deque[dict[str, Any]] = deque(maxlen=MAX_HAZARD_EVENTS_IN_MEMORY)
    hazard_event_count = 0
    hazard_incident_count = 0
    incident_tracker = HazardIncidentTracker()
    worst_score = -1
    started_at = time.monotonic()
    prev_gray: Optional[np.ndarray] = None
    prev_boxes: list[list[float]] = []
    prev_points: dict[int, tuple[float, float]] = {}
    speed_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=speed_window))
    overspeed_frame_count = 0
    max_speed: Optional[float] = None

    try:
        while True:
            if stop_flag is not None and stop_flag():
                stopped = True
                break
            if image_files:
                if processed >= len(image_files):
                    break
                frame = cv2.imread(str(image_files[processed]))
                if frame is None:
                    processed += 1
                    continue
                frame_name = image_files[processed].name
            elif live_reader is not None:
                packet = live_reader.wait_for_frame(last_seen_source_id, timeout=1.0)
                if packet is None:
                    if (
                        not live_reader.opened_once
                        and time.monotonic() - started_at > live_reader.startup_timeout_seconds
                    ):
                        raise RuntimeError(live_reader.last_failure or live_reader.error or "直播流连接超时。")
                    continue
                last_seen_source_id = packet.frame_id
                if live_generation is not None and packet.generation != live_generation:
                    _reset_model_trackers(model)
                    smoother = HelmetTrackSmoother()
                    crane_temporal.reset()
                    incident_tracker.reset()
                    previous_processed_source_id = None
                    prev_gray = None
                    prev_boxes = []
                    prev_points.clear()
                    speed_history.clear()
                live_generation = packet.generation
                if previous_processed_source_id is not None and packet.frame_id - previous_processed_source_id < sample_interval:
                    continue
                source_frame_id = packet.frame_id
                fps = packet.fps
                frame = packet.frame
                frame_name = f"stream_frame_{source_frame_id}"
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok:
                    break
                source_frame_id += 1
                if raw_preview_callback is not None:
                    raw_preview_callback(source_frame_id, frame, {"fps": fps, "generation": 1})
                if (source_frame_id - 1) % sample_interval != 0:
                    continue
                frame_name = f"frame_{source_frame_id}"

            height, width = frame.shape[:2]
            if input_size is None:
                input_size = (width, height)
            elif input_size != (width, height):
                frame = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
            frame_id = source_frame_id if source_type in {"video", "live_stream"} else processed + 1
            source_delta = max(1, frame_id - previous_processed_source_id) if previous_processed_source_id is not None else 1

            with yolo_detection_logging(model, frame_id, detection_log_callback):
                detection_result = _track_frame(model, frame, tracker, inference_conf, iou, device, target_class_ids)
            # Keep the shared result at the lowest required threshold for crane rules.
            detections = [det for det in parse_detections(detection_result)
                          if det["confidence"] >= conf and normalize_class_name(det["class_name"]) in box_class_names]
            persons: list[dict[str, Any]] = []
            if enabled_checks.intersection({"person_helmet", "driver_helmet"}):
                persons = evaluate_persons(detections, frame.shape, 0.2)
                smoother.update(persons, frame_id)

            work_result = None
            if work_args is not None:
                _, work_result = analyze_forklift_work_frame(
                    model,
                    frame,
                    work_args,
                    frame_name=frame_name,
                    detection_result=detection_result,
                    draw_output=False,
                    detections=detections,
                )
                smoothed_by_box = {tuple(person.get("box") or []): person.get("status", "unknown") for person in persons}
                for forklift in work_result.get("forklifts", []) or []:
                    for driver in forklift.get("excluded_persons", []) or []:
                        driver["helmet_status"] = smoothed_by_box.get(tuple(driver.get("box") or []), driver.get("helmet_status", "unknown"))

            speed_result = None
            needs_motion_state = (
                "forklift_speed" in enabled_checks
                or ("forklift_fence" in enabled_checks and source_type != "single_image")
            )
            if needs_motion_state:
                forklifts = [det for det in filter_detections(detections, "forklift") if det.get("track_id") is not None]
                curr_boxes = [det["box"] for det in forklifts]
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    motion_success, matrix = False, None
                else:
                    motion_success, matrix, _ = estimate_motion(prev_gray, curr_gray, prev_boxes, curr_boxes, min_bg_matches, motion_model)
                speed_items = update_speeds(
                    forklifts, prev_points, speed_history, matrix, motion_success, motion_model,
                    speed_limit, source_delta, fps, unsafe_speed_limit,
                    camera_mode=camera_mode,
                )
                has_overspeed = any(item.get("overspeed") for item in speed_items)
                if has_overspeed:
                    overspeed_frame_count += 1
                known = [item for item in speed_items if item.get("decision_speed_kmh") is not None]
                fastest = max(known, key=lambda item: float(item["decision_speed_kmh"]), default=None)
                if fastest is not None:
                    current_max = float(fastest["smoothed_speed_kmh"])
                    max_speed = current_max if max_speed is None else max(max_speed, current_max)
                speed_result = {
                    "result": "unknown" if not known else ("unsafe" if has_overspeed else "safe"),
                    "speed_status": "unknown" if not known else ("overspeed" if has_overspeed else "ok"),
                    "camera_mode": camera_mode,
                    "speed_calibration": speed_calibration_info(camera_mode),
                    "forklift_count": len(speed_items),
                    "current_speed_px_per_frame": fastest.get("current_speed_px_per_frame") if fastest else None,
                    "smoothed_speed_px_per_frame": fastest.get("smoothed_speed_px_per_frame") if fastest else None,
                    "current_speed_kmh": fastest.get("current_speed_kmh") if fastest else None,
                    "smoothed_speed_kmh": fastest.get("smoothed_speed_kmh") if fastest else None,
                    "movement_threshold_kmh": movement_threshold_kmh,
                    "speed_limit_kmh": speed_limit,
                    "unsafe_speed_kmh": unsafe_speed_limit,
                    "speed_average_window_frames": speed_window,
                    "speed_decision_basis": "up_to_window_rolling_average_speed_kmh",
                    "max_speed_kmh": round(max_speed, 4) if max_speed is not None else None,
                    "overspeed_frame_count": overspeed_frame_count,
                    "motion_compensation_success": bool(motion_success),
                    "source_frame_delta": source_delta,
                    "forklifts": speed_items,
                }
                prev_gray = curr_gray
                prev_boxes = curr_boxes

            crane_result = None
            if CRANE_CHECK in enabled_checks:
                crane_result = analyze_crane_obb_result(
                    detection_result,
                    frame.shape,
                    frame_id,
                    options=crane_config,
                    temporal_filter=crane_temporal,
                    temporal_applied=source_type != "single_image",
                )

            lifting_result = analyze_lifting_operation(detections) if LIFTING_CHECK in enabled_checks else None
            frame_result = build_selected_frame_result(
                enabled_checks, frame_id, persons, work_result, speed_result, crane_result, lifting_result
            )
            frame_result.update(source_type=source_type, sample_interval=sample_interval, source_frame_delta=source_delta, image=frame_name)
            annotated = draw_selected_frame(
                frame,
                enabled_checks,
                frame_result,
                detections,
                draw_circle_work_zone=bool(getattr(work_args, "draw_circle_work_zone", True)),
                crane_result=crane_result,
            )
            current_instances = build_hazard_instances("safety_combined", frame_result)
            new_incidents = incident_tracker.update(current_instances)
            current_has_hazard = bool(current_instances)
            frame_result["has_hazard"] = current_has_hazard
            if new_incidents and attach_hazard_event(
                frame_result,
                annotated,
                output_media,
                "safety_combined",
                new_incidents,
                hazard_event_count == 0,
                hazard_event_count + 1,
            ):
                event = copy.deepcopy(frame_result["hazard_event"])
                hazard_events.append(event)
                hazard_event_count += 1
                hazard_incident_count += int(event.get("hazard_count", 0))
            processed += 1
            previous_processed_source_id = frame_id
            latest_result = frame_result
            if frame_result.get("result") == "unsafe":
                unsafe_frames += 1
                score = sum(1 for item in frame_result.get("results", {}).values() if item.get("result") == "unsafe")
                if score > worst_score:
                    worst_score = score
                    worst_result = copy.deepcopy(frame_result)

            if preview_callback is not None:
                preview_callback(frame_id, annotated.copy(), frame_result)
            if progress_callback is not None:
                progress_callback(processed, total_frames)
    finally:
        if cap is not None:
            cap.release()
        if live_reader is not None:
            release_shared_live_reader(str(source_path), live_reader, frame_callback=live_frame_callback)

    if not latest_result:
        latest_result = {"selected_checks": [x for x in SAFETY_CHECK_ORDER if x in enabled_checks], "results": {}, "result": "unknown"}
    final_result = copy.deepcopy(worst_result if unsafe_frames and worst_result else latest_result)
    if hazard_events:
        latest_event = copy.deepcopy(hazard_events[-1])
        final_result.update(
            hazard_event=latest_event,
            hazard_timestamp=latest_event["hazard_timestamp"],
            hazard_image_path=latest_event["hazard_image_path"],
            hazard_image_paths=latest_event["hazard_image_paths"],
            hazard_count=latest_event["hazard_count"],
            hazards=latest_event["hazards"],
        )
    final_result.update(
        source=str(source_path),
        source_type=source_type,
        sample_interval=sample_interval,
        processed_frames=processed,
        unsafe_frame_count=unsafe_frames,
        hazard_event_count=hazard_event_count,
        hazard_incident_count=hazard_incident_count,
        hazard_events=list(hazard_events),
        stopped_by_user=stopped,
        result="unsafe" if unsafe_frames else str(latest_result.get("result", "unknown")),
    )
    return add_finished_at(final_result)


def run_safety_monitor(
    mode: str,
    source_type: str,
    source_path: Union[Path, str],
    weights_path: str,
    output_media: Optional[Path],
    conf: float = 0.4,
    iou: float = NMS_IOU,
    device: str = "cpu",
    tracker: str = TRACKER_CONFIG,
    work_options: Optional[dict[str, Any]] = None,
    preview_callback: Optional[Callable[[int, np.ndarray, dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    sample_interval: int = 5,
    raw_preview_callback: Optional[Callable[[int, np.ndarray, dict[str, Any]], None]] = None,
    detection_log_callback: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    if mode not in {"helmet", "forklift_work"}:
        raise ValueError(f"unsupported monitor mode: {mode}")
    if source_type not in {"single_image", "image_sequence", "video", "live_stream"}:
        raise ValueError(f"unsupported source type: {source_type}")
    sample_interval = max(1, int(sample_interval))
    source_path = str(source_path) if source_type == "live_stream" else Path(source_path)

    model = load_model(weights_path)
    device = resolve_device(device)
    smoother = HelmetTrackSmoother(window=1 if source_type == "single_image" else 5)
    work_args = make_work_args(weights_path, conf, device, work_options or {}) if mode == "forklift_work" else None
    cap: Optional[cv2.VideoCapture] = None
    live_reader: Optional[LiveStreamReader] = None
    live_frame_callback: Optional[Callable[[Any], None]] = None
    image_files: list[Path] = []
    fps = VIDEO_FALLBACK_FPS
    total_frames = 1
    stopped = False

    if source_type == "single_image":
        image_files = [source_path]
    elif source_type == "image_sequence":
        image_files = get_image_files(source_path)
        if not image_files:
            raise ValueError("输入文件夹中没有可用图片。")
        total_frames = len(image_files)
        fps = 10.0
    elif source_type == "video":
        assert isinstance(source_path, Path)
        if not source_path.is_file():
            raise ValueError("视频模式下，输入必须是视频文件。")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {source_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or VIDEO_FALLBACK_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    else:
        assert isinstance(source_path, str)

        def on_raw(packet) -> None:
            if raw_preview_callback is not None:
                raw_preview_callback(packet.frame_id, packet.frame, {"fps": packet.fps, "generation": packet.generation})

        live_frame_callback = on_raw
        live_reader = acquire_shared_live_reader(source_path, frame_callback=live_frame_callback)
        total_frames = 0

    processed = 0
    unsafe_frames = 0
    max_person_count = 0
    max_no_helmet_count = 0
    latest_result: dict[str, Any] = {}
    worst_result: dict[str, Any] = {}
    hazard_events: deque[dict[str, Any]] = deque(maxlen=MAX_HAZARD_EVENTS_IN_MEMORY)
    hazard_event_count = 0
    hazard_incident_count = 0
    incident_tracker = HazardIncidentTracker()
    worst_score = -1
    input_size: Optional[tuple[int, int]] = None
    source_frame_id = 0
    last_seen_source_id = -1
    previous_processed_source_id: Optional[int] = None
    live_generation: Optional[int] = None
    started_at = time.monotonic()

    try:
        while True:
            if stop_flag is not None and stop_flag():
                stopped = True
                break
            if image_files:
                if processed >= len(image_files):
                    break
                frame = cv2.imread(str(image_files[processed]))
                if frame is None:
                    processed += 1
                    continue
                frame_name = image_files[processed].name
            elif live_reader is not None:
                packet = live_reader.wait_for_frame(last_seen_source_id, timeout=1.0)
                if packet is None:
                    if (
                        not live_reader.opened_once
                        and time.monotonic() - started_at > live_reader.startup_timeout_seconds
                    ):
                        raise RuntimeError(live_reader.last_failure or live_reader.error or "直播流连接超时。")
                    continue
                last_seen_source_id = packet.frame_id
                if live_generation is not None and packet.generation != live_generation:
                    _reset_model_trackers(model)
                    smoother = HelmetTrackSmoother()
                    incident_tracker.reset()
                    previous_processed_source_id = None
                live_generation = packet.generation
                if previous_processed_source_id is not None and packet.frame_id - previous_processed_source_id < sample_interval:
                    continue
                source_frame_id = packet.frame_id
                fps = packet.fps
                frame = packet.frame
                frame_name = f"stream_frame_{source_frame_id}"
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok:
                    break
                source_frame_id += 1
                if raw_preview_callback is not None:
                    raw_preview_callback(source_frame_id, frame, {"fps": fps, "generation": 1})
                frame_name = f"frame_{processed + 1}"

            height, width = frame.shape[:2]
            if input_size is None:
                input_size = (width, height)
            elif input_size != (width, height):
                frame = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)

            frame_id = source_frame_id if source_type in {"video", "live_stream"} else processed + 1
            with yolo_detection_logging(model, frame_id, detection_log_callback):
                if mode == "helmet":
                    annotated, frame_result = analyze_helmet_frame(
                        model, frame, frame_id, conf, iou, device, tracker, smoother
                    )
                else:
                    assert work_args is not None
                    annotated, frame_result = analyze_forklift_work_frame(
                        model, frame, work_args, frame_name=frame_name, tracker=tracker,
                        helmet_smoother=smoother, frame_id=frame_id,
                    )
                    frame_result["frame_id"] = frame_id
            frame_result.setdefault("image", frame_name)
            frame_result["source_type"] = source_type
            frame_result["sample_interval"] = sample_interval
            current_instances = build_hazard_instances(mode, frame_result)
            new_incidents = incident_tracker.update(current_instances)
            current_has_hazard = bool(current_instances)
            frame_result["has_hazard"] = current_has_hazard
            if new_incidents and attach_hazard_event(
                frame_result,
                annotated,
                output_media,
                mode,
                new_incidents,
                hazard_event_count == 0,
                hazard_event_count + 1,
            ):
                event = copy.deepcopy(frame_result["hazard_event"])
                hazard_events.append(event)
                hazard_event_count += 1
                hazard_incident_count += int(event.get("hazard_count", 0))

            processed += 1
            previous_processed_source_id = frame_id
            latest_result = frame_result
            person_count = int(frame_result.get("total_persons", frame_result.get("person_count", 0)) or 0)
            no_helmet_count = int(frame_result.get("no_helmet_count", 0) or 0)
            max_person_count = max(max_person_count, person_count)
            max_no_helmet_count = max(max_no_helmet_count, no_helmet_count)
            if str(frame_result.get("result", "unknown")) == "unsafe":
                unsafe_frames += 1
                violation_score = no_helmet_count + len(frame_result.get("violations", []) or [])
                if violation_score > worst_score:
                    worst_score = violation_score
                    worst_result = copy.deepcopy(frame_result)

            if preview_callback is not None:
                preview_callback(frame_id, annotated.copy(), frame_result)
            if progress_callback is not None:
                progress_callback(processed, total_frames)

    finally:
        if cap is not None:
            cap.release()
        if live_reader is not None:
            release_shared_live_reader(source_path, live_reader, frame_callback=live_frame_callback)

    if not latest_result:
        latest_result = {"result": "unknown", "persons": [], "total_persons": 0, "no_helmet_count": 0}
    final_result = copy.deepcopy(worst_result if unsafe_frames and worst_result else latest_result)
    if hazard_events:
        latest_event = copy.deepcopy(hazard_events[-1])
        final_result.update(
            hazard_event=latest_event,
            hazard_timestamp=latest_event["hazard_timestamp"],
            hazard_image_path=latest_event["hazard_image_path"],
            hazard_image_paths=latest_event["hazard_image_paths"],
            hazard_count=latest_event["hazard_count"],
            hazards=latest_event["hazards"],
        )
    final_result.update(
        {
            "source": str(source_path),
            "source_type": source_type,
            "sample_interval": sample_interval,
            "processed_frames": processed,
            "unsafe_frame_count": unsafe_frames,
            "hazard_event_count": hazard_event_count,
            "hazard_incident_count": hazard_incident_count,
            "hazard_events": list(hazard_events),
            "max_person_count": max_person_count,
            "max_no_helmet_count": max_no_helmet_count,
            "stopped_by_user": stopped,
            "result": "unsafe" if unsafe_frames else str(latest_result.get("result", "unknown")),
        }
    )
    return add_finished_at(final_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety monitoring for images, image sequences, and videos with BoT-SORT.")
    parser.add_argument("--mode", choices=["helmet", "forklift_work"])
    parser.add_argument("--checks", help="Comma-separated unified checks")
    parser.add_argument("--source-type", choices=["single_image", "image_sequence", "video", "live_stream"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=NMS_IOU, help="Framework NMS threshold (backend/CLI only)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nearby-mode", choices=["original", "circle", "both"], default="circle")
    parser.add_argument("--camera-mode", choices=["oblique", "overhead"], default="oblique")
    parser.add_argument("--forklift-radius-scale", type=float, default=1.2)
    parser.add_argument(
        "--min-forklift-radius-px",
        type=float,
        default=180.0,
        help="Deprecated compatibility option; ignored by work-zone radius calculation",
    )
    parser.add_argument(
        "--max-forklift-radius-px",
        type=float,
        default=450.0,
        help="Deprecated compatibility option; ignored by work-zone radius calculation",
    )
    parser.add_argument("--fence-iou-thres", type=float, default=0.01)
    parser.add_argument("--speed-limit", type=float, default=REFERENCE_SPEED_KMH)
    parser.add_argument("--unsafe-speed-limit", type=float, default=UNSAFE_SPEED_KMH)
    parser.add_argument("--movement-threshold", type=float, default=1.0, help="Moving/stationary threshold in km/h, applied to rolling-average speed")
    parser.add_argument("--speed-window", type=int, default=30)
    parser.add_argument("--min-bg-matches", type=int, default=20)
    parser.add_argument("--motion-model", choices=["affine", "homography"], default="affine")
    parser.add_argument("--boom-conf", type=float, default=0.40)
    parser.add_argument("--crane-person-conf", type=float, default=0.40)
    parser.add_argument("--crane-line-margin", type=float, default=5.0)
    parser.add_argument("--crane-bottom-margin", type=float, default=5.0)
    parser.add_argument("--crane-range-extend-ratio", type=float, default=0.05)
    parser.add_argument("--crane-enter-window", type=int, default=5)
    parser.add_argument("--crane-enter-hits", type=int, default=3)
    parser.add_argument("--crane-exit-safe-frames", type=int, default=5)
    parser.add_argument("--crane-track-ttl-frames", type=int, default=30)
    parser.add_argument("--sample-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_options = {
        "nearby_mode": args.nearby_mode,
        "camera_mode": args.camera_mode,
        "forklift_radius_scale": args.forklift_radius_scale,
        "min_forklift_radius_px": args.min_forklift_radius_px,
        "max_forklift_radius_px": args.max_forklift_radius_px,
        "fence_iou_thres": args.fence_iou_thres,
    }
    source = args.source if args.source_type == "live_stream" else Path(args.source)
    if args.checks:
        checks = {item.strip() for item in args.checks.split(",") if item.strip()}
        result = run_unified_safety_monitor(
            enabled_checks=checks,
            source_type=args.source_type,
            source_path=source,
            weights_path=args.weights,
            output_media=Path(args.output),
            conf=args.conf,
            iou=args.iou,
            device=resolve_device(args.device),
            work_options=work_options,
            speed_options={
                "speed_limit": args.speed_limit,
                "unsafe_speed_limit": args.unsafe_speed_limit,
                "movement_threshold": args.movement_threshold,
                "speed_window": args.speed_window,
                "min_bg_matches": args.min_bg_matches,
                "motion_model": args.motion_model,
                "camera_mode": args.camera_mode,
            },
            crane_options={
                "boom_conf": args.boom_conf,
                "person_conf": args.crane_person_conf,
                "line_margin": args.crane_line_margin,
                "bottom_margin": args.crane_bottom_margin,
                "range_extend_ratio": args.crane_range_extend_ratio,
                "enter_window": args.crane_enter_window,
                "enter_hits": args.crane_enter_hits,
                "exit_safe_frames": args.crane_exit_safe_frames,
                "track_ttl_frames": args.crane_track_ttl_frames,
            },
            sample_interval=args.sample_interval,
        )
    else:
        if not args.mode:
            raise ValueError("either --mode or --checks is required")
        result = run_safety_monitor(
            mode=args.mode,
            source_type=args.source_type,
            source_path=source,
            weights_path=args.weights,
            output_media=Path(args.output),
            conf=args.conf,
            iou=args.iou,
            device=resolve_device(args.device),
            work_options=work_options,
            sample_interval=args.sample_interval,
        )
    json_path = Path(args.json)
    save_json(result, json_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
