import argparse
import sys
import time
from collections import defaultdict, deque
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from common import (
    NMS_IOU,
    INFERENCE_IMAGE_SIZE,
    HazardIncidentTracker,
    MAX_HAZARD_EVENTS_IN_MEMORY,
    bottom_center,
    distance,
    draw_box,
    draw_text_lines,
    draw_unsafe_banner,
    draw_raw_detection_boxes,
    ensure_dir,
    int_box,
    load_model,
    parse_detections,
    validate_model_classes,
    yolo_detection_logging,
    print_json,
    resolve_device,
    save_hazard_image,
    save_json,
    add_finished_at,
)


Point = Tuple[float, float]
# Known forklift ground footprint used for approximate single-camera scale estimation.
FORKLIFT_LENGTH_M = 2.5
FORKLIFT_WIDTH_M = 1.5
FORKLIFT_HEIGHT_M = 2.0
REFERENCE_SPEED_KMH = 5.0
UNSAFE_SPEED_KMH = 6.0


def log_status(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def format_speed_status(frame_items: List[Dict[str, Any]]) -> str:
    if not frame_items:
        return "speeds=none"
    parts = []
    for item in frame_items:
        current_speed = item["current_speed_kmh"]
        smoothed_speed = item["smoothed_speed_kmh"]
        if current_speed is None:
            speed_text = "unknown"
        else:
            speed_text = f"{current_speed:.2f} km/h"
        if smoothed_speed is None:
            avg_text = "unknown"
        else:
            avg_text = f"{smoothed_speed:.2f}"
        parts.append(f'ID{item["track_id"]}: {speed_text} (avg {avg_text}) {item["speed_status"]}')
    return "speeds=" + "; ".join(parts)


def speed_calibration_info(camera_mode: str = "oblique") -> Dict[str, Any]:
    """Describe the known-size diagonal used for local speed calibration."""
    if camera_mode == "overhead":
        axes = ("length", "width")
        dimensions = (FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
        method = "known_forklift_footprint_diagonal_local_scale"
    elif camera_mode == "oblique":
        axes = ("length", "height")
        dimensions = (FORKLIFT_LENGTH_M, FORKLIFT_HEIGHT_M)
        method = "known_forklift_side_diagonal_local_scale"
    else:
        raise ValueError("camera_mode must be one of oblique, overhead")
    axis_1, axis_2 = axes
    return {
        "camera_mode": camera_mode,
        "method": method,
        "calibration_axes": [axis_1, axis_2],
        "forklift_dimensions_m": {
            "length": FORKLIFT_LENGTH_M,
            "width": FORKLIFT_WIDTH_M,
            "height": FORKLIFT_HEIGHT_M,
        },
        "reference_diagonal_m": float(np.hypot(*dimensions)),
        "formula": f"meters_per_pixel = sqrt({axis_1}^2 + {axis_2}^2) / bbox_diagonal_px",
        "speed_formula": "speed_kmh = pixel_speed_per_frame * meters_per_pixel * fps * 3.6",
    }


def estimate_meters_per_pixel(box: List[float], camera_mode: str = "oblique") -> Optional[float]:
    """Estimate a local scale using the visible dimensions for the camera mode.

    A 2D detection box is not a full camera calibration target, so this is an
    intentionally approximate self-calibration. Overhead views use the known
    length/width diagonal; oblique views use the known length/height diagonal.
    Re-estimating it every frame makes the scale adapt as a forklift moves.
    """
    calibration = speed_calibration_info(camera_mode)
    coords = int_box(box)
    if coords is None:
        return None
    x1, y1, x2, y2 = coords
    pixel_diagonal = float(np.hypot(x2 - x1, y2 - y1))
    if pixel_diagonal < 8.0:
        return None
    return float(calibration["reference_diagonal_m"]) / pixel_diagonal


def rounded_kmh(speed_kmh: Optional[float]) -> Optional[float]:
    return round(float(speed_kmh), 4) if speed_kmh is not None else None


def make_background_mask(shape: Tuple[int, int], boxes: List[List[float]]) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    height, width = shape
    for box in boxes:
        x1, y1, x2, y2 = int_box(box) or [0, 0, 0, 0]
        pad_x = max(4, int((x2 - x1) * 0.08))
        pad_y = max(4, int((y2 - y1) * 0.08))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width - 1, x2 + pad_x)
        y2 = min(height - 1, y2 + pad_y)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    return mask


def estimate_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_boxes: List[List[float]],
    curr_boxes: List[List[float]],
    min_matches: int,
    motion_model: str,
) -> Tuple[bool, Optional[np.ndarray], int]:
    orb = cv2.ORB_create(nfeatures=2500)
    prev_mask = make_background_mask(prev_gray.shape, prev_boxes)
    curr_mask = make_background_mask(curr_gray.shape, curr_boxes)

    kp1, des1 = orb.detectAndCompute(prev_gray, prev_mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, curr_mask)
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return False, None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < min_matches:
        return False, None, len(matches)

    matches = sorted(matches, key=lambda m: m.distance)[: max(min_matches, int(len(matches) * 0.7))]
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if motion_model == "homography":
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if matrix is None or inliers is None or int(inliers.sum()) < min_matches:
        return False, None, len(matches)
    return True, matrix, len(matches)


def transform_point(point: Point, matrix: Optional[np.ndarray], motion_model: str) -> Point:
    if matrix is None:
        return point
    pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
    if motion_model == "homography":
        mapped = cv2.perspectiveTransform(pts, matrix)[0, 0]
    else:
        mapped = cv2.transform(pts, matrix)[0, 0]
    return (float(mapped[0]), float(mapped[1]))


def track_forklifts(model, frame: np.ndarray, conf: float, device: Optional[str] = None) -> List[Dict[str, Any]]:
    track_kwargs = {
        "conf": conf,
        "imgsz": INFERENCE_IMAGE_SIZE,
        "classes": validate_model_classes(model, {"forklift"}),
        "end2end": False,
        "agnostic_nms": False,
        "iou": NMS_IOU,
        "tracker": "botsort.yaml",
        "persist": True,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device
    results = model.track(source=frame, **track_kwargs)
    detections = parse_detections(results[0] if results else None)
    return [det for det in detections if det["class_name"].casefold() == "forklift" and det["track_id"] is not None]


def update_speeds(
    forklifts: List[Dict[str, Any]],
    prev_points: Dict[int, Point],
    speed_history: Dict[int, Deque[float]],
    matrix: Optional[np.ndarray],
    motion_success: bool,
    motion_model: str,
    speed_limit_kmh: float,
    source_frame_delta: int = 1,
    fps: float = 25.0,
    unsafe_speed_kmh: float = UNSAFE_SPEED_KMH,
    camera_mode: str = "oblique",
) -> List[Dict[str, Any]]:
    calibration = speed_calibration_info(camera_mode)
    frame_items = []
    current_points: Dict[int, Point] = {}

    for det in forklifts:
        track_id = int(det["track_id"])
        ground_point = bottom_center(det["box"])
        current_points[track_id] = ground_point
        meters_per_pixel = estimate_meters_per_pixel(det["box"], camera_mode)

        if track_id in prev_points:
            previous_point = prev_points[track_id]
            compensated = transform_point(previous_point, matrix, motion_model) if motion_success else previous_point
            displacement = distance(ground_point, compensated)
            frame_delta = max(1, int(source_frame_delta))
            current_speed_px = displacement / frame_delta
            if meters_per_pixel is not None and fps > 0:
                current_speed_kmh = current_speed_px * meters_per_pixel * float(fps) * 3.6
                speed_history[track_id].append(current_speed_kmh)
                smoothed = float(np.mean(speed_history[track_id])) if speed_history[track_id] else None
                smoothed_speed_px = smoothed / (meters_per_pixel * float(fps) * 3.6) if smoothed is not None else None
            else:
                current_speed_kmh = None
                smoothed = None
                smoothed_speed_px = None
            # Use exactly the same two-decimal rolling-average value that is
            # shown in the UI for every threshold decision.  The deque keeps at
            # most 30 samples; during startup all samples collected so far are
            # averaged, as requested.
            decision_speed = round(smoothed, 2) if smoothed is not None else None
            over_speed_limit = bool(decision_speed is not None and decision_speed > speed_limit_kmh)
            overspeed = bool(decision_speed is not None and decision_speed > unsafe_speed_kmh)
            speed_status = "overspeed" if overspeed else "ok"
            speed_source = (
                "motion_compensated_ground_pixel_speed"
                if motion_success
                else "image_displacement_fallback"
            )
        else:
            current_speed_px = None
            current_speed_kmh = None
            smoothed = None
            smoothed_speed_px = None
            decision_speed = None
            over_speed_limit = False
            overspeed = False
            speed_status = "unknown"
            speed_source = "unavailable_no_previous_track"

        frame_items.append(
            {
                "track_id": track_id,
                "box": int_box(det["box"]),
                "ground_point": [round(ground_point[0], 2), round(ground_point[1], 2)],
                "current_speed_px_per_frame": round(current_speed_px, 4) if current_speed_px is not None else None,
                "sampled_displacement_px": round(displacement, 4) if current_speed_px is not None else None,
                "source_frame_delta": max(1, int(source_frame_delta)),
                "smoothed_speed_px_per_frame": round(smoothed_speed_px, 4) if smoothed_speed_px is not None else None,
                "meters_per_pixel": round(meters_per_pixel, 6) if meters_per_pixel is not None else None,
                "current_speed_kmh": rounded_kmh(current_speed_kmh),
                # This is both the displayed rolling average and the decision
                # value, so the existing API only needs to expose one result.
                "smoothed_speed_kmh": decision_speed,
                "decision_speed_kmh": decision_speed,
                "speed_average_sample_count": len(speed_history[track_id]),
                "speed_average_window_frames": int(speed_history[track_id].maxlen or 0),
                "speed_limit_kmh": float(speed_limit_kmh),
                "unsafe_speed_kmh": float(unsafe_speed_kmh),
                "speed_calibration": {
                    **calibration,
                    "decision_formula": "decision_speed_kmh = round(mean(last up-to-window speed samples), 2); over_limit > speed_limit_kmh; overspeed > unsafe_speed_kmh",
                },
                "overspeed": bool(overspeed),
                "speed_status": speed_status,
                "motion_compensation_success": bool(motion_success) if current_speed_px is not None else False,
                "speed_source": speed_source,
            }
        )

    prev_points.clear()
    prev_points.update(current_points)
    return frame_items


def draw_warning_banner(frame: np.ndarray, text: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 54), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    forklifts: List[Dict[str, Any]],
    frame_items: List[Dict[str, Any]],
    motion_success: bool,
    camera_mode: str = "oblique",
) -> None:
    by_track = {item["track_id"]: item for item in frame_items}
    for det in forklifts:
        track_id = int(det["track_id"])
        item = by_track[track_id]
        speed_kmh = item["current_speed_kmh"]
        average_kmh = item["decision_speed_kmh"]
        average_window = item.get("speed_average_window_frames", 30)
        sample_count = item.get("speed_average_sample_count", 0)
        if speed_kmh is None:
            label = f"ID {track_id}: speed calibrating"
            color, thickness = (0, 165, 255), 3
        elif item["overspeed"]:
            label = f"UNSAFE | AVG {average_kmh:.2f} km/h (>6) [{sample_count}/{average_window}]"
            color, thickness = (0, 0, 255), 5
        elif average_kmh is not None and average_kmh > item["speed_limit_kmh"]:
            label = f"OVER LIMIT | AVG {average_kmh:.2f} km/h (>5, no alarm) [{sample_count}/{average_window}]"
            color, thickness = (0, 165, 255), 4
        else:
            label = f"ID {track_id}: AVG {average_kmh:.2f} km/h (limit 5) [{sample_count}/{average_window}]"
            color, thickness = (0, 180, 0), 3
        draw_box(frame, det["box"], label, color, thickness)
        gp = item["ground_point"]
        cv2.circle(frame, (int(round(gp[0])), int(round(gp[1]))), 5, (0, 255, 255), -1)
    if any(item["overspeed"] for item in frame_items):
        draw_unsafe_banner(frame, ["叉车超速"])
    axes_text = "length x width" if camera_mode == "overhead" else "length x height"
    draw_text_lines(
        frame,
        [
            f"Calibration ({camera_mode}): {axes_text}",
            f"Rolling AVG up to 30 frames | Limit: 5 km/h | Alarm: > 6 km/h | MC: {'OK' if motion_success else 'FAIL'}",
        ],
        origin=(10, max(32, frame.shape[0] - 68)),
        bg_color=(0, 0, 120) if any(item["overspeed"] for item in frame_items) else (0, 90, 0),
    )


def process_video(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.time()
    camera_mode = str(getattr(args, "camera_mode", "oblique"))
    calibration = speed_calibration_info(camera_mode)
    preview_callback = getattr(args, "preview_callback", None)
    raw_preview_callback = getattr(args, "raw_preview_callback", None)
    progress_callback = getattr(args, "progress_callback", None)
    stop_flag = getattr(args, "stop_flag", None)
    source_type = getattr(args, "source_type", "video")
    sample_interval = max(1, int(getattr(args, "sample_interval", 5)))
    stopped_by_user = False
    cap: Optional[cv2.VideoCapture] = None
    live_reader = None
    live_frame_callback = None
    fps = 25.0
    total_frames = 0
    source_frame_id = 0
    last_seen_source_id = -1
    log_status(f"[start] opening {source_type}: {args.video}", args.quiet)

    if source_type == "live_stream":
        from live_stream import acquire_shared_live_reader, release_shared_live_reader

        def on_raw(packet) -> None:
            if raw_preview_callback is not None:
                raw_preview_callback(packet.frame_id, packet.frame, {"fps": packet.fps, "generation": packet.generation})

        live_frame_callback = on_raw
        live_reader = acquire_shared_live_reader(str(args.video), frame_callback=live_frame_callback)
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"failed to open video: {args.video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ensure_dir(args.output)
    log_status(f"[start] loading model: {args.weights}", args.quiet)
    model = load_model(args.weights)
    log_status("[ready] model loaded, processing frames...", args.quiet)

    frames_json = deque(maxlen=5000) if source_type == "live_stream" else []
    prev_gray: Optional[np.ndarray] = None
    prev_boxes: List[List[float]] = []
    prev_points: Dict[int, Point] = {}
    speed_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=args.speed_window))
    processed_count = 0
    previous_processed_source_id: Optional[int] = None
    live_generation: Optional[int] = None
    motion_ok_count = 0
    forklift_seen_count = 0
    overspeed_frame_count = 0
    overspeed_track_ids = set()
    hazard_events: Deque[Dict[str, Any]] = deque(maxlen=MAX_HAZARD_EVENTS_IN_MEMORY)
    hazard_event_count = 0
    hazard_incident_count = 0
    incident_tracker = HazardIncidentTracker()
    max_speed_kmh: Optional[float] = None
    calculated_speed_count = 0

    try:
        while True:
            if stop_flag is not None and stop_flag():
                stopped_by_user = True
                break
            if live_reader is not None:
                packet = live_reader.wait_for_frame(last_seen_source_id, timeout=1.0)
                if packet is None:
                    if (
                        not live_reader.opened_once
                        and time.time() - start_time > live_reader.startup_timeout_seconds
                    ):
                        raise RuntimeError(live_reader.last_failure or live_reader.error or "直播流连接超时。")
                    continue
                last_seen_source_id = packet.frame_id
                if live_generation is not None and packet.generation != live_generation:
                    predictor = getattr(model, "predictor", None)
                    for tracker in getattr(predictor, "trackers", []) or []:
                        reset = getattr(tracker, "reset", None)
                        if callable(reset):
                            reset()
                    prev_gray = None
                    prev_boxes = []
                    prev_points.clear()
                    speed_history.clear()
                    previous_processed_source_id = None
                    incident_tracker.reset()
                live_generation = packet.generation
                if previous_processed_source_id is not None and packet.frame_id - previous_processed_source_id < sample_interval:
                    continue
                source_frame_id = packet.frame_id
                fps = packet.fps
                frame = packet.frame
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

            source_frame_delta = (
                max(1, source_frame_id - previous_processed_source_id)
                if previous_processed_source_id is not None
                else 1
            )

            with yolo_detection_logging(model, source_frame_id, getattr(args, "detection_log_callback", None)):
                forklifts = track_forklifts(model, frame, args.conf, getattr(args, "device", None))
            curr_boxes = [det["box"] for det in forklifts]
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                motion_success, matrix = False, None
            else:
                motion_success, matrix, _ = estimate_motion(
                    prev_gray, curr_gray, prev_boxes, curr_boxes, args.min_bg_matches, args.motion_model
                )
            if motion_success:
                motion_ok_count += 1

            frame_items = update_speeds(
                forklifts, prev_points, speed_history, matrix, motion_success, args.motion_model,
                args.speed_limit_kmh, source_frame_delta, fps, args.unsafe_speed_kmh,
                camera_mode=camera_mode,
            )
            forklift_seen_count += len(frame_items)
            frame_has_overspeed = False
            for item in frame_items:
                current_speed = item["smoothed_speed_kmh"]
                if current_speed is not None:
                    calculated_speed_count += 1
                    max_speed_kmh = current_speed if max_speed_kmh is None else max(max_speed_kmh, current_speed)
                if item["overspeed"]:
                    frame_has_overspeed = True
                    overspeed_track_ids.add(item["track_id"])
            if frame_has_overspeed:
                overspeed_frame_count += 1
            draw_frame(frame, forklifts, frame_items, motion_success, camera_mode)
            known_items = [item for item in frame_items if item["decision_speed_kmh"] is not None]
            fastest_item = max(known_items, key=lambda item: item["decision_speed_kmh"], default=None)
            has_overspeed = any(item["overspeed"] for item in frame_items)
            speed_status = "unknown" if not frame_items or fastest_item is None else ("overspeed" if has_overspeed else "ok")
            frame_status = {
            "frame_id": source_frame_id,
            "source_type": source_type,
            "processed_frame_count": processed_count + 1,
            "sample_interval": sample_interval,
            "source_frame_delta": source_frame_delta,
            "current_speed_px_per_frame": fastest_item["current_speed_px_per_frame"] if fastest_item else None,
            "smoothed_speed_px_per_frame": fastest_item["smoothed_speed_px_per_frame"] if fastest_item else None,
            "current_speed_kmh": fastest_item["current_speed_kmh"] if fastest_item else None,
            "smoothed_speed_kmh": fastest_item["smoothed_speed_kmh"] if fastest_item else None,
            "speed_limit_kmh": float(args.speed_limit_kmh),
            "unsafe_speed_kmh": float(args.unsafe_speed_kmh),
            "overspeed": bool(has_overspeed),
            "speed_status": speed_status,
            "track_id": fastest_item["track_id"] if fastest_item else None,
            "forklift_count": len(frame_items),
            "max_speed_kmh": rounded_kmh(max_speed_kmh),
            "overspeed_frame_count": overspeed_frame_count,
            "total_frames": total_frames,
            "motion_compensation_success": bool(motion_success),
            "forklifts": frame_items,
            "result": "unsafe" if has_overspeed else ("unknown" if fastest_item is None else "safe"),
            "has_hazard": bool(has_overspeed),
            }
            current_incidents = [
                {
                    "code": "FORKLIFT_OVERSPEED",
                    "check": "forklift_speed",
                    "message": "叉车超速",
                    "count": 1,
                    "details": [item],
                    "targets": [{"type": "forklift", "track_id": item["track_id"], "target_id": str(item["track_id"])}],
                    "_incident_key": ("FORKLIFT_OVERSPEED", "forklift", str(item["track_id"])),
                }
                for item in frame_items
                if item.get("overspeed")
            ]
            new_incidents = incident_tracker.update(current_incidents)
            if new_incidents:
                evidence = save_hazard_image(
                    frame,
                    Path(args.output),
                    "forklift_speed_FORKLIFT_OVERSPEED",
                    source_frame_id,
                )
                persisted_incidents = [
                    {
                        **incident,
                        "hazard_timestamp": evidence["hazard_timestamp"],
                        "hazard_image_path": evidence["hazard_image_path"],
                    }
                    for incident in new_incidents
                ]
                event = {
                    "event_id": f"forklift_speed_event_{hazard_event_count + 1}_frame_{source_frame_id}",
                    "event_sequence": hazard_event_count + 1,
                    "algorithm": "forklift_speed",
                    "source_type": source_type,
                    "frame_id": source_frame_id,
                    "is_first_hazard": hazard_event_count == 0,
                    "hazard_timestamp": evidence["hazard_timestamp"],
                    "hazard_image_path": evidence["hazard_image_path"],
                    "hazard_image_paths": [evidence["hazard_image_path"]],
                    "hazard_count": len(persisted_incidents),
                    "hazards": persisted_incidents,
                }
                hazard_events.append(event)
                hazard_event_count += 1
                hazard_incident_count += int(event["hazard_count"])
                frame_status.update(
                    is_first_hazard=event["is_first_hazard"],
                    hazard_timestamp=event["hazard_timestamp"],
                    hazard_image_path=event["hazard_image_path"],
                    hazard_image_paths=event["hazard_image_paths"],
                    hazard_count=event["hazard_count"],
                    hazards=event["hazards"],
                    hazard_event=event,
                )
            if preview_callback is not None:
                preview_callback(source_frame_id, frame.copy(), frame_status)
            if progress_callback is not None:
                progress_callback(source_frame_id, total_frames)

            frames_json.append(
                {
                "frame_id": source_frame_id,
                "source_frame_delta": source_frame_delta,
                "motion_compensation_success": bool(motion_success),
                "forklifts": frame_items,
                }
            )

            prev_gray = curr_gray
            prev_boxes = curr_boxes
            previous_processed_source_id = source_frame_id
            processed_count += 1
            if processed_count == 1 or processed_count % args.progress_interval == 0:
                elapsed = max(1e-6, time.time() - start_time)
                log_status(f"[progress] source_frame={source_frame_id}, processed={processed_count}, speed={processed_count / elapsed:.2f} frames/s, {format_speed_status(frame_items)}", args.quiet)
    finally:
        if cap is not None:
            cap.release()
        if live_reader is not None:
            release_shared_live_reader(str(args.video), live_reader, frame_callback=live_frame_callback)
    elapsed = max(1e-6, time.time() - start_time)
    log_status(
        (
            f"[done] processed {processed_count} frames in {elapsed:.1f}s "
            f"({processed_count / elapsed:.2f} frames/s), motion_ok={motion_ok_count}, "
            f"forklift_detections={forklift_seen_count}"
        ),
        args.quiet,
    )
    if hazard_events:
        log_status(f"[done] hazard images: {sum(len(event['hazard_image_paths']) for event in hazard_events)}", args.quiet)
    if calculated_speed_count == 0:
        result_status = "unknown"
    elif overspeed_track_ids:
        result_status = "unsafe"
    else:
        result_status = "safe"
    summary = {
        "video": str(args.video) if source_type == "live_stream" else str(Path(args.video).name),
        "source_type": source_type,
        "sample_interval": sample_interval,
        "camera_mode": camera_mode,
        "speed_limit_kmh": float(args.speed_limit_kmh),
        "unsafe_speed_kmh": float(args.unsafe_speed_kmh),
        "speed_average_window_frames": int(args.speed_window),
        "speed_decision_basis": "up_to_30_frame_rolling_average_speed_kmh",
        "speed_calibration": calibration,
        "total_frames": processed_count,
        "overspeed_frame_count": overspeed_frame_count,
        "overspeed_track_ids": sorted(overspeed_track_ids),
        "max_speed_kmh": rounded_kmh(max_speed_kmh),
        "result": result_status,
        "stopped_by_user": bool(stopped_by_user),
        "hazard_event_count": hazard_event_count,
        "hazard_incident_count": hazard_incident_count,
    }
    result = {
        "video": str(args.video) if source_type == "live_stream" else str(Path(args.video).name),
        "source_type": source_type,
        "sample_interval": sample_interval,
        "camera_mode": camera_mode,
        "fps": fps,
        "speed_limit_kmh": float(args.speed_limit_kmh),
        "unsafe_speed_kmh": float(args.unsafe_speed_kmh),
        "speed_average_window_frames": int(args.speed_window),
        "speed_decision_basis": "up_to_30_frame_rolling_average_speed_kmh",
        "summary": summary,
        "frames": list(frames_json),
        "hazard_events": list(hazard_events),
    }
    if hazard_events:
        latest_event = hazard_events[-1]
        result.update(
            has_hazard=True,
            is_first_hazard=latest_event["is_first_hazard"],
            hazard_timestamp=latest_event["hazard_timestamp"],
            hazard_image_path=latest_event["hazard_image_path"],
            hazard_image_paths=latest_event["hazard_image_paths"],
            hazard_count=latest_event["hazard_count"],
            hazards=latest_event["hazards"],
            hazard_event=latest_event,
        )
    else:
        result["has_hazard"] = False
    return add_finished_at(result)


def process_forklift_speed_video(
    weights_path: str,
    video_path: str,
    output_path: str,
    json_path: Optional[str] = None,
    speed_limit_kmh: float = REFERENCE_SPEED_KMH,
    unsafe_speed_kmh: float = UNSAFE_SPEED_KMH,
    preview_callback: Optional[Callable[[int, np.ndarray, Dict[str, Any]], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    conf: float = 0.4,
    speed_window: int = 30,
    min_bg_matches: int = 20,
    motion_model: str = "affine",
    device: Optional[str] = None,
    progress_interval: int = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    source_type: str = "video",
    sample_interval: int = 5,
    raw_preview_callback: Optional[Callable[[int, np.ndarray, Dict[str, Any]], None]] = None,
    camera_mode: str = "oblique",
    detection_log_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    device = resolve_device(device)
    args = SimpleNamespace(
        weights=str(weights_path),
        video=str(video_path),
        output=str(output_path),
        json=str(json_path) if json_path else None,
        conf=float(conf),
        speed_window=int(speed_window),
        speed_limit_kmh=float(speed_limit_kmh),
        unsafe_speed_kmh=float(unsafe_speed_kmh),
        min_bg_matches=int(min_bg_matches),
        motion_model=str(motion_model),
        device=device,
        progress_interval=int(progress_interval),
        quiet=bool(quiet),
        preview_callback=preview_callback,
        stop_flag=stop_flag,
        progress_callback=progress_callback,
        source_type=source_type,
        sample_interval=max(1, int(sample_interval)),
        raw_preview_callback=raw_preview_callback,
        detection_log_callback=detection_log_callback,
        camera_mode=str(camera_mode),
    )
    if args.speed_window < 1:
        raise ValueError("speed_window must be >= 1")
    if args.speed_limit_kmh < 0 or args.unsafe_speed_kmh < args.speed_limit_kmh:
        raise ValueError("unsafe_speed_kmh must be >= speed_limit_kmh >= 0")
    if args.progress_interval < 1:
        raise ValueError("progress_interval must be >= 1")
    result = process_video(args)
    if json_path:
        save_json(result, json_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forklift self-calibrated speed estimation with background motion compensation.")
    parser.add_argument("--weights", required=True, help="YOLO weights path, e.g. safety_best.pt")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--source-type", choices=["video", "live_stream"], default="video")
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--camera-mode", choices=["oblique", "overhead"], default="oblique")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--json", help="Output JSON path. If omitted, JSON is printed to terminal")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--speed-window", type=int, default=30, help="Rolling-average window in frames for overspeed decisions")
    parser.add_argument("--speed-limit-kmh", type=float, default=REFERENCE_SPEED_KMH, help="Displayed speed limit in km/h")
    parser.add_argument("--unsafe-speed-kmh", type=float, default=UNSAFE_SPEED_KMH, help="Unsafe alarm threshold in km/h")
    parser.add_argument("--min-bg-matches", type=int, default=20, help="Minimum background feature matches")
    parser.add_argument("--motion-model", choices=["affine", "homography"], default="affine", help="Global motion model")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 for GPU or cpu for CPU")
    parser.add_argument("--progress-interval", type=int, default=30, help="Print progress every N frames")
    parser.add_argument("--quiet", action="store_true", help="Disable console progress messages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.speed_window < 1:
        raise ValueError("--speed-window must be >= 1")
    if args.speed_limit_kmh < 0 or args.unsafe_speed_kmh < args.speed_limit_kmh:
        raise ValueError("--unsafe-speed-kmh must be >= --speed-limit-kmh >= 0")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be >= 1")
    result = process_video(args)
    if args.json:
        log_status(f"[done] writing JSON: {args.json}", args.quiet)
        save_json(result, args.json)
    else:
        print_json(result)


if __name__ == "__main__":
    main()
