import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from common import (
    INFERENCE_IMAGE_SIZE,
    NMS_IOU,
    box_center,
    box_iou,
    bottom_center,
    clip_box,
    distance,
    draw_box,
    draw_text_lines,
    draw_raw_detection_boxes,
    draw_unsafe_banner,
    filter_detections,
    int_box,
    is_valid_box,
    load_model,
    parse_detections,
    point_in_box,
    resolve_device,
    overlap_ratio,
    run_detection,
    save_hazard_image,
    save_json,
    add_finished_at,
)
from helmet_check import evaluate_persons


HELMET_STATUS_COLORS = {
    "helmet": (0, 180, 0),
    "no_helmet": (0, 0, 255),
    "unknown": (0, 215, 255),
}

FORKLIFT_LENGTH_M = 2.5
FORKLIFT_WIDTH_M = 1.5
FORKLIFT_FOOTPRINT_MINOR_MAJOR_RATIO = FORKLIFT_WIDTH_M / FORKLIFT_LENGTH_M


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def make_forklift_ground_zone(
    forklift_box: Sequence[float],
    image_shape,
    left_scale: float,
    right_scale: float,
    front_scale: float,
    back_scale: float,
    height_depth_ratio: float,
    width_depth_ratio: float,
) -> Tuple[List[float], float]:
    x1, y1, x2, y2 = forklift_box
    ground_x, ground_y = bottom_center(forklift_box)
    forklift_width = x2 - x1
    forklift_height = y2 - y1
    ground_depth_px = min(forklift_height * height_depth_ratio, forklift_width * width_depth_ratio)
    zone = (
        ground_x - forklift_width * left_scale,
        ground_y - ground_depth_px * front_scale,
        ground_x + forklift_width * right_scale,
        ground_y + ground_depth_px * back_scale,
    )
    height, width = image_shape[:2]
    return clip_box(zone, width, height), float(ground_depth_px)


def is_box_fully_inside(inner_box: Sequence[float], outer_box: Sequence[float]) -> bool:
    """Return True when every edge of ``inner_box`` is inside or on ``outer_box``."""
    inner_x1, inner_y1, inner_x2, inner_y2 = (float(value) for value in inner_box)
    outer_x1, outer_y1, outer_x2, outer_y2 = (float(value) for value in outer_box)
    return bool(
        inner_x1 >= outer_x1
        and inner_y1 >= outer_y1
        and inner_x2 <= outer_x2
        and inner_y2 <= outer_y2
    )


def is_driver_or_excluded_person(
    person_box: Sequence[float],
    person_ground_point: Tuple[float, float],
    forklift_box: Sequence[float],
    forklift_ground_point: Tuple[float, float],
    driver_iou_thres: float,
    driver_inside_ratio_thres: float,
    driver_dx_norm_thres: float,
    driver_min_dy_up_norm: float,
    driver_max_dy_up_norm: float,
) -> bool:
    # A person whose entire detection box is covered by a forklift box is unambiguously treated as its driver.
    if is_box_fully_inside(person_box, forklift_box):
        return True
    x1, _, x2, _ = forklift_box
    px, _ = person_ground_point
    metrics = ground_relative_position(person_ground_point, forklift_box, forklift_ground_point)
    relative_driver_position = (
        x1 <= px <= x2
        and metrics["dx_norm"] < driver_dx_norm_thres
        and metrics["dy_up_norm"] > driver_min_dy_up_norm
        and metrics["dy_up_norm"] < driver_max_dy_up_norm
    )
    person_forklift_iou = box_iou(person_box, forklift_box)
    person_inside_forklift_ratio = overlap_ratio(person_box, forklift_box)
    overlaps_forklift = (
        person_forklift_iou >= driver_iou_thres
        or person_inside_forklift_ratio >= driver_inside_ratio_thres
    )
    return bool(relative_driver_position and overlaps_forklift)


def person_foot_points(person_box: Sequence[float]) -> List[Tuple[float, float]]:
    x1, _, x2, y2 = person_box
    width = max(1.0, float(x2 - x1))
    return [(float(x1 + width * ratio), float(y2)) for ratio in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]


def min_distance_to_points(origin: Tuple[float, float], points: Sequence[Tuple[float, float]]) -> Tuple[float, Tuple[float, float]]:
    if not points:
        return 0.0, origin
    best = min(points, key=lambda point: distance(origin, point))
    return distance(origin, best), best


def any_point_in_box(points: Sequence[Tuple[float, float]], box: Sequence[float]) -> bool:
    return any(point_in_box(point, box) for point in points)


def first_point_in_mask(points: Sequence[Tuple[float, float]], mask: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    if mask is None:
        return None
    height, width = mask.shape[:2]
    for px, py in points:
        x = int(round(px))
        y = int(round(py))
        if 0 <= x < width and 0 <= y < height and mask[y, x] != 0:
            return (float(px), float(py))
    return None


def first_point_in_polygon(points: Sequence[Tuple[float, float]], polygon: Optional[List[List[int]]]) -> Optional[Tuple[float, float]]:
    if not polygon or len(polygon) < 3:
        return None
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    for point in points:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return point
    return None


def build_circle_work_zone(
    forklift_box: Sequence[float],
    work_zone: Sequence[float],
    forklift_radius_scale: float,
    min_forklift_radius_px: float,
    max_forklift_radius_px: float,
    road_direction_angle: Optional[float] = None,
    road_direction_confidence: float = 0.0,
    camera_mode: str = "oblique",
) -> Dict[str, Any]:
    # ``min_forklift_radius_px`` and ``max_forklift_radius_px`` are retained in
    # the call signature for backward compatibility only.  A fixed pixel clamp
    # makes the physical work zone change with camera resolution and forklift
    # distance. ``work_zone`` is also retained for compatibility, but its
    # left/right/front/back expansion must not be applied again here.  The
    # circle/ellipse axes are derived directly from the forklift detection box.
    x1, y1, x2, y2 = forklift_box
    forklift_ground_point = bottom_center(forklift_box)
    forklift_width = max(1.0, float(x2 - x1))
    forklift_height = max(1.0, float(y2 - y1))
    radius_scale = max(0.01, float(forklift_radius_scale))
    box_axis_x_px = forklift_width * 0.5 * radius_scale
    box_axis_y_px = forklift_height * 0.5 * radius_scale

    if camera_mode == "overhead":
        center = box_center(forklift_box)
        radius_px = max(box_axis_x_px, box_axis_y_px)
        polygon = ellipse_polygon(center, radius_px, radius_px, 0.0, num_points=72)
        return {
            "enabled": True,
            "center": [round(center[0], 2), round(center[1], 2)],
            "radius_px": round(float(radius_px), 2),
            "radius_x_px": round(float(radius_px), 2),
            "radius_y_px": round(float(radius_px), 2),
            "radius_scale": radius_scale,
            "min_radius_px": float(min_forklift_radius_px),
            "max_radius_px": float(max_forklift_radius_px),
            "shape": "circle",
            "rotation_angle_deg": 0.0,
            "road_direction_confidence": 0.0,
            "scale_source": "forklift_box_overhead",
            "polygon": polygon,
        }

    # In an oblique view a near-square detector box must not collapse the work
    # zone back into a circle.  Use the known 2.5 m x 1.5 m footprint ratio:
    # the detected box supplies local pixel scale, while physical length/width
    # supplies a stable ellipse aspect ratio.  Height is intentionally unused.
    axis_x_px = max(forklift_width, forklift_height) * 0.5 * radius_scale
    axis_y_px = axis_x_px * FORKLIFT_FOOTPRINT_MINOR_MAJOR_RATIO
    rotation_angle_deg = float(road_direction_angle) if road_direction_angle is not None else 0.0

    polygon = ellipse_polygon(
        forklift_ground_point,
        axis_x_px,
        axis_y_px,
        rotation_angle_deg,
        num_points=72,
    )
    return {
        "enabled": True,
        "center": [round(forklift_ground_point[0], 2), round(forklift_ground_point[1], 2)],
        "radius_px": round(float(max(axis_x_px, axis_y_px)), 2),
        "radius_x_px": round(float(axis_x_px), 2),
        "radius_y_px": round(float(axis_y_px), 2),
        "radius_scale": radius_scale,
        "min_radius_px": float(min_forklift_radius_px),
        "max_radius_px": float(max_forklift_radius_px),
        "shape": "ellipse",
        "rotation_angle_deg": round(float(rotation_angle_deg), 2),
        "road_direction_confidence": round(float(road_direction_confidence), 4),
        "scale_source": "forklift_box_known_length_width_ratio",
        "polygon": polygon,
    }


def ellipse_polygon(
    center: Tuple[float, float],
    axis_x_px: float,
    axis_y_px: float,
    rotation_angle_deg: float,
    num_points: int = 72,
) -> List[List[int]]:
    cx, cy = center
    theta = np.radians(float(rotation_angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    points = []
    for i in range(max(12, int(num_points))):
        ang = 2.0 * np.pi * i / max(12, int(num_points))
        x = float(axis_x_px) * float(np.cos(ang))
        y = float(axis_y_px) * float(np.sin(ang))
        rx = cx + x * cos_t - y * sin_t
        ry = cy + x * sin_t + y * cos_t
        points.append([int(round(rx)), int(round(ry))])
    return points


def point_in_rotated_ellipse(
    point: Tuple[float, float],
    center: Tuple[float, float],
    axis_x_px: float,
    axis_y_px: float,
    rotation_angle_deg: float,
) -> bool:
    if axis_x_px <= 0 or axis_y_px <= 0:
        return False
    px, py = point
    cx, cy = center
    theta = -np.radians(float(rotation_angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    dx = px - cx
    dy = py - cy
    xr = dx * cos_t - dy * sin_t
    yr = dx * sin_t + dy * cos_t
    value = (xr / float(axis_x_px)) ** 2 + (yr / float(axis_y_px)) ** 2
    return value <= 1.0


def check_person_nearby_original_rules(
    person: Dict[str, Any],
    person_id: int,
    forklift: Dict[str, Any],
    forklift_id: int,
    forklift_ground_point: Tuple[float, float],
    work_zone: Sequence[float],
    fence_based_work_zone: Optional[Sequence[Sequence[int]]],
    ground_point_work_area_mask: Optional[np.ndarray],
    road_constrained_work_area_mask: Optional[np.ndarray],
    road_aligned_work_zone: Dict[str, Any],
    near_threshold_px: float,
) -> Dict[str, Any]:
    person_ground_point = bottom_center(person["box"])
    foot_points = person_foot_points(person["box"])
    distance_px, closest_foot_point = min_distance_to_points(forklift_ground_point, foot_points)
    in_rect_work_zone = any_point_in_box(foot_points, work_zone)
    near_by_distance = distance_px <= near_threshold_px
    in_fence_based_work_zone = first_point_in_polygon(foot_points, fence_based_work_zone) is not None
    in_ground_point_work_area = is_person_in_ground_point_work_area(person["box"], ground_point_work_area_mask)
    in_road_constrained_work_area = bool(
        road_constrained_work_area_mask is not None
        and is_person_in_mask_area(person["box"], road_constrained_work_area_mask)
    )
    in_road_aligned_forklift_work_zone = is_person_in_mask_area(person["box"], road_aligned_work_zone.get("mask"))
    in_fence_clipped_forklift_work_zone = bool(
        road_aligned_work_zone.get("fence_clipped_work_zone", {}).get("success")
        and in_road_aligned_forklift_work_zone
    )

    reasons = []
    if near_by_distance:
        reasons.append("near_forklift_by_ground_point_distance")
    if in_rect_work_zone:
        reasons.append("in_rect_work_zone")
    if in_fence_based_work_zone:
        reasons.append("in_fence_based_work_zone")
    if in_ground_point_work_area:
        reasons.append("in_ground_point_work_area")
    if in_road_constrained_work_area:
        reasons.append("in_road_constrained_work_area")
    if in_fence_clipped_forklift_work_zone:
        reasons.append("in_fence_clipped_forklift_work_zone")
    if in_road_aligned_forklift_work_zone:
        reasons.append("in_road_aligned_forklift_work_zone")

    return {
        "matched": bool(reasons),
        "reasons": reasons,
        "debug_info": {
            "person_id": person_id,
            "forklift_id": forklift_id,
            "distance_to_forklift_px": round(float(distance_px), 2),
            "near_threshold_px": round(float(near_threshold_px), 2),
            "person_ground_point": [round(person_ground_point[0], 2), round(person_ground_point[1], 2)],
            "person_foot_points": [[round(px, 2), round(py, 2)] for px, py in foot_points],
            "closest_foot_point": [round(closest_foot_point[0], 2), round(closest_foot_point[1], 2)],
            "in_rect_work_zone": bool(in_rect_work_zone),
            "near_by_distance": bool(near_by_distance),
            "in_fence_based_work_zone": bool(in_fence_based_work_zone),
            "in_ground_point_work_area": bool(in_ground_point_work_area),
            "in_road_constrained_work_area": bool(in_road_constrained_work_area),
            "in_road_aligned_forklift_work_zone": bool(in_road_aligned_forklift_work_zone),
            "in_fence_clipped_forklift_work_zone": bool(in_fence_clipped_forklift_work_zone),
        },
    }


def check_person_nearby_circle_rule(
    person: Dict[str, Any],
    forklift_box: Sequence[float],
    forklift_ground_point: Tuple[float, float],
    circle_work_zone: Dict[str, Any],
) -> Dict[str, Any]:
    person_ground_point = bottom_center(person["box"])
    foot_points = person_foot_points(person["box"])
    distance_px, closest_foot_point = min_distance_to_points(forklift_ground_point, foot_points)
    radius_px = float(circle_work_zone.get("radius_px", 0.0) or 0.0)
    axis_x_px = float(circle_work_zone.get("radius_x_px", radius_px) or radius_px)
    axis_y_px = float(circle_work_zone.get("radius_y_px", radius_px) or radius_px)
    rotation_angle_deg = float(circle_work_zone.get("rotation_angle_deg", 0.0) or 0.0)
    center = tuple(circle_work_zone.get("center") or [forklift_ground_point[0], forklift_ground_point[1]])
    shape = str(circle_work_zone.get("shape", "circle"))
    if shape == "ellipse":
        matched = any(point_in_rotated_ellipse(point, center, axis_x_px, axis_y_px, rotation_angle_deg) for point in foot_points)
    else:
        matched = any(distance(point, center) <= radius_px for point in foot_points)
    return {
        "matched": matched,
        "reasons": ["in_circle_work_zone"] if matched else [],
        "debug_info": {
            "distance_to_forklift_px": round(float(distance_px), 2),
            "radius_px": round(radius_px, 2),
            "radius_x_px": round(axis_x_px, 2),
            "radius_y_px": round(axis_y_px, 2),
            "rotation_angle_deg": round(rotation_angle_deg, 2),
            "shape": shape,
            "center": circle_work_zone.get("center"),
            "person_ground_point": [round(person_ground_point[0], 2), round(person_ground_point[1], 2)],
            "person_foot_points": [[round(px, 2), round(py, 2)] for px, py in foot_points],
            "closest_foot_point": [round(closest_foot_point[0], 2), round(closest_foot_point[1], 2)],
        },
    }


def ground_relative_position(
    person_ground_point: Tuple[float, float],
    forklift_box: Sequence[float],
    forklift_ground_point: Tuple[float, float],
) -> Dict[str, float]:
    _, y1, _, y2 = forklift_box
    fx, fy = forklift_ground_point
    px, py = person_ground_point
    forklift_width = max(1.0, float(forklift_box[2] - forklift_box[0]))
    forklift_height = max(1.0, float(y2 - y1))
    dx = abs(px - fx)
    dy_up = fy - py
    return {
        "dx_norm": float(dx / forklift_width),
        "dy_up_norm": float(dy_up / forklift_height),
    }


def make_fence_based_work_zone(
    forklift_box: Sequence[float],
    nearby_fences: List[Dict[str, Any]],
    image_shape,
    padding: float,
) -> Optional[List[List[int]]]:
    if not nearby_fences:
        return None

    fx1, _, fx2, fy2 = forklift_box
    points = [[fx1, fy2], [fx2, fy2]]
    for fence in nearby_fences:
        x1, y1, x2, y2 = fence["box"]
        cx, cy = box_center(fence["box"])
        points.extend([[x1, y2], [x2, y2], [cx, cy]])

    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return None

    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    centroid = hull.mean(axis=0)
    padded = []
    height, width = image_shape[:2]
    for point in hull:
        vector = point - centroid
        length = float(np.linalg.norm(vector))
        if length > 1e-6:
            point = point + vector / length * padding
        x = int(round(max(0, min(width - 1, point[0]))))
        y = int(round(max(0, min(height - 1, point[1]))))
        padded.append([x, y])
    return padded if len(padded) >= 3 else None


def contour_to_polygon(contour: np.ndarray, epsilon_ratio: float = 0.01) -> List[List[int]]:
    if contour is None or len(contour) == 0:
        return []
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(point[0][0]), int(point[0][1])] for point in approx]


def build_ground_point_metadata(
    forklift_boxes: Sequence[Sequence[float]],
    fence_boxes: Sequence[Sequence[float]],
) -> Dict[str, List[Dict[str, Any]]]:
    forklift_boxes = [box for box in forklift_boxes if is_valid_box(box)]
    fence_boxes = [box for box in fence_boxes if is_valid_box(box)]
    return {
        "forklifts": [
            {
                "forklift_id": idx,
                "box": int_box(box),
                "ground_point": [round(bottom_center(box)[0], 2), round(bottom_center(box)[1], 2)],
            }
            for idx, box in enumerate(forklift_boxes, start=1)
        ],
        "fences": [
            {
                "fence_id": idx,
                "box": int_box(box),
                "ground_point": [round(bottom_center(box)[0], 2), round(bottom_center(box)[1], 2)],
            }
            for idx, box in enumerate(fence_boxes, start=1)
        ],
    }


def generate_ground_point_work_area(
    image_shape,
    forklift_boxes,
    fence_boxes,
    padding_px=80,
    fallback_zone_radius=220,
) -> Optional[Dict[str, Any]]:
    try:
        height, width = image_shape[:2]
    except (TypeError, ValueError, IndexError):
        return None
    if int(height) <= 0 or int(width) <= 0:
        return None
    forklift_candidates = [] if forklift_boxes is None else forklift_boxes
    fence_candidates = [] if fence_boxes is None else fence_boxes
    forklift_boxes = [box for box in forklift_candidates if is_valid_box(box)]
    fence_boxes = [box for box in fence_candidates if is_valid_box(box)]
    # A forklift work area has no business meaning without at least one valid forklift.
    if not forklift_boxes:
        return None
    ground_points = build_ground_point_metadata(forklift_boxes, fence_boxes)
    points = [item["ground_point"] for item in ground_points["forklifts"] + ground_points["fences"]]

    mask = np.zeros((height, width), dtype=np.uint8)
    area_type = "forklift_fence_ground_point_convex_hull"

    if len(points) >= 3:
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull.astype(np.int32)], 255)
        kernel_size = max(1, int(round(padding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        center = bottom_center(forklift_boxes[0])
        radius = max(1, int(round(fallback_zone_radius)))
        cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), radius, 255, -1)
        area_type = "forklift_ground_point_fallback_circle"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    polygon = contour_to_polygon(contour)
    if len(polygon) < 3:
        return None

    return {
        "mask": mask,
        "enabled": True,
        "type": area_type,
        "padding_px": int(round(padding_px)),
        "fallback_zone_radius": int(round(fallback_zone_radius)),
        "polygon": polygon,
        "ground_points": ground_points,
    }


def is_person_in_ground_point_work_area(person_box, work_area_mask) -> bool:
    return first_point_in_mask(person_foot_points(person_box), work_area_mask) is not None


def mask_to_polygons(mask: Optional[np.ndarray], min_area: float = 20.0) -> List[List[List[int]]]:
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            polygon = contour_to_polygon(contour)
            if len(polygon) >= 3:
                polygons.append(polygon)
    return polygons


def generate_road_surface_mask(
    image,
    forklift_boxes,
    fence_boxes,
    method="auto",
    min_area_ratio=0.03,
    road_sat_max=80,
    road_value_min=30,
    road_value_max=210,
    road_morph_kernel=15,
) -> Dict[str, Any]:
    height, width = image.shape[:2]
    total_area = float(height * width)
    metadata = {
        "enabled": True,
        "success": False,
        "method": "auto_hsv_connected_component" if method == "auto" else method,
        "road_area_ratio": 0.0,
        "used_connected_component": False,
        "contains_forklift_ground_point": False,
        "warning": None,
    }

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        low_sat = saturation < int(road_sat_max)
        value_range = (value > int(road_value_min)) & (value < int(road_value_max))
        green = (hue >= 35) & (hue <= 85) & (saturation > 60)
        blue = (hue >= 90) & (hue <= 135) & (saturation > 60)
        raw_mask = (low_sat & value_range & ~green & ~blue).astype(np.uint8) * 255

        kernel_size = max(3, int(road_morph_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            metadata["warning"] = "no_road_candidate_component"
            return {"mask": None, "metadata": metadata}

        forklift_points = [bottom_center(box) for box in forklift_boxes]
        selected_labels = set()
        contains_forklift = False
        for point in forklift_points:
            x = int(round(point[0]))
            y = int(round(point[1]))
            if 0 <= x < width and 0 <= y < height:
                label = int(labels[y, x])
                if label > 0:
                    selected_labels.add(label)
                    contains_forklift = True

        if not selected_labels:
            candidate_labels = range(1, num_labels)
            largest_label = max(candidate_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA])
            selected_labels.add(int(largest_label))

        road_mask = np.zeros((height, width), dtype=np.uint8)
        for label in selected_labels:
            road_mask[labels == label] = 255

        area_ratio = float(cv2.countNonZero(road_mask) / total_area) if total_area > 0 else 0.0
        metadata["road_area_ratio"] = round(area_ratio, 4)
        metadata["used_connected_component"] = True
        metadata["contains_forklift_ground_point"] = bool(contains_forklift)

        if area_ratio < float(min_area_ratio):
            metadata["warning"] = "road_mask_area_too_small"
            return {"mask": None, "metadata": metadata}
        if area_ratio > 0.9:
            metadata["warning"] = "road_mask_area_too_large"
            return {"mask": None, "metadata": metadata}

        metadata["success"] = True
        if not contains_forklift:
            metadata["warning"] = "selected_largest_component_without_forklift_ground_point"
        return {"mask": road_mask, "metadata": metadata}
    except Exception as exc:
        metadata["warning"] = f"road_mask_error: {exc}"
        return {"mask": None, "metadata": metadata}


def generate_road_mask(
    image,
    forklift_ground_point=None,
    road_sat_max=80,
    road_value_min=25,
    road_value_max=220,
    road_morph_kernel=15,
) -> Optional[np.ndarray]:
    try:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        low_sat = saturation < int(road_sat_max)
        value_range = (value > int(road_value_min)) & (value < int(road_value_max))
        green = (hue >= 35) & (hue <= 85) & (saturation > 60)
        blue = (hue >= 90) & (hue <= 135) & (saturation > 60)
        mask = (low_sat & value_range & ~green & ~blue).astype(np.uint8) * 255

        kernel_size = max(3, int(road_morph_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return None

        selected_label = None
        if forklift_ground_point is not None:
            x = int(round(forklift_ground_point[0]))
            y = int(round(forklift_ground_point[1]))
            if 0 <= x < width and 0 <= y < height:
                label = int(labels[y, x])
                if label > 0:
                    selected_label = label

        if selected_label is None:
            selected_label = max(range(1, num_labels), key=lambda label: stats[label, cv2.CC_STAT_AREA])

        road_mask = np.zeros((height, width), dtype=np.uint8)
        road_mask[labels == selected_label] = 255
        return road_mask if cv2.countNonZero(road_mask) > 0 else None
    except Exception:
        return None


def estimate_local_road_direction(
    road_mask,
    forklift_ground_point,
    roi_radius,
    min_pixels=500,
) -> Tuple[Optional[float], float]:
    if road_mask is None:
        return None, 0.0
    height, width = road_mask.shape[:2]
    gx, gy = forklift_ground_point
    x1 = max(0, int(round(gx - roi_radius)))
    y1 = max(0, int(round(gy - roi_radius)))
    x2 = min(width, int(round(gx + roi_radius)))
    y2 = min(height, int(round(gy + roi_radius)))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    roi = road_mask[y1:y2, x1:x2]
    ys, xs = np.nonzero(roi)
    if len(xs) < int(min_pixels):
        return None, 0.0

    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    if np.shape(cov) != (2, 2) or not np.all(np.isfinite(cov)):
        return None, 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if len(eigenvalues) == 0 or eigenvectors.shape[0] < 2:
        return None, 0.0
    order = np.argsort(eigenvalues)[::-1]
    primary_value = float(eigenvalues[order[0]])
    secondary_value = float(eigenvalues[order[1]]) if len(eigenvalues) > 1 else 0.0
    if primary_value <= 1e-6:
        return None, 0.0

    vector = eigenvectors[:, order[0]]
    angle = float(np.arctan2(vector[1], vector[0]))
    confidence = max(0.0, min(1.0, 1.0 - secondary_value / primary_value))
    return angle, confidence


def box_to_mask(image_shape, box: Sequence[float]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = int_box(clip_box(box, width, height))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def polygon_mask(image_shape, polygon: np.ndarray) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def mask_largest_polygon(mask: Optional[np.ndarray], min_area: float = 20.0) -> Optional[List[List[int]]]:
    polygons = mask_to_polygons(mask, min_area=min_area)
    if not polygons:
        return None
    return max(polygons, key=lambda polygon: cv2.contourArea(np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)))


def make_fence_clip_mask(
    image_shape,
    forklift_box,
    fence_group_boxes,
    padding_px=80,
) -> Optional[Dict[str, Any]]:
    if not fence_group_boxes:
        return None

    height, width = image_shape[:2]
    fx1, _, fx2, fy2 = forklift_box
    points = [[fx1, fy2], [fx2, fy2], list(bottom_center(forklift_box))]
    for fence_box in fence_group_boxes:
        x1, _, x2, y2 = fence_box
        points.extend([[x1, y2], [x2, y2], list(bottom_center(fence_box))])

    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return None

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [hull.astype(np.int32)], 255)
    kernel_radius = max(1, int(round(padding_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_radius * 2 + 1, kernel_radius * 2 + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    if cv2.countNonZero(mask) <= 0:
        return None
    return {
        "mask": mask,
        "polygon": mask_largest_polygon(mask),
    }


def make_fence_barrier_mask(
    image_shape,
    fence_group_boxes,
    thickness=20,
) -> Optional[Dict[str, Any]]:
    if not fence_group_boxes:
        return None

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    barrier_boxes = []
    line_thickness = max(1, int(round(thickness)))
    for fence_box in fence_group_boxes:
        x1, y1, x2, y2 = int_box(clip_box(fence_box, width, height))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.line(mask, (x1, y2), (x2, y2), 255, line_thickness)
        barrier_boxes.append([x1, y1, x2, y2])

    if cv2.countNonZero(mask) <= 0:
        return None
    return {
        "mask": mask,
        "boxes": barrier_boxes,
        "polygons": mask_to_polygons(mask, min_area=1.0),
    }


def keep_component_containing_point(mask: np.ndarray, point: Tuple[float, float]) -> Tuple[Optional[np.ndarray], bool]:
    if mask is None or cv2.countNonZero(mask) <= 0:
        return None, False
    height, width = mask.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    if x < 0 or x >= width or y < 0 or y >= height:
        return None, False

    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None, False
    label = int(labels[y, x])
    if label <= 0:
        return None, False
    kept = np.zeros_like(mask)
    kept[labels == label] = 255
    return kept, cv2.countNonZero(kept) > 0


def rotated_zone_polygon_from_base_rect(
    image_shape,
    forklift_box,
    base_rect_work_zone,
    road_direction_angle,
    max_length_scale=2.2,
    max_width_scale=1.3,
    forward_ratio=0.65,
    backward_ratio=0.35,
    length_factor=1.0,
    width_factor=1.0,
) -> Tuple[np.ndarray, float, float]:
    height, width = image_shape[:2]
    base_x1, base_y1, base_x2, base_y2 = base_rect_work_zone
    base_w = max(1.0, float(base_x2 - base_x1))
    base_h = max(1.0, float(base_y2 - base_y1))
    x1, y1, x2, y2 = forklift_box
    forklift_width = max(1.0, float(x2 - x1))
    forklift_height = max(1.0, float(y2 - y1))

    base_long = max(base_w, base_h)
    base_short = min(base_w, base_h)
    max_len = max(forklift_width, forklift_height) * float(max_length_scale)
    max_wid = forklift_width * float(max_width_scale)
    zone_length = max(1.0, min(base_long, max_len) * float(length_factor))
    zone_width = max(1.0, min(base_short, max_wid) * float(width_factor))

    forward_len = zone_length * float(forward_ratio)
    backward_len = zone_length * float(backward_ratio)
    half_width = zone_width / 2.0
    direction = np.array([np.cos(road_direction_angle), np.sin(road_direction_angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    anchor = np.array(bottom_center(forklift_box), dtype=np.float32)
    front = anchor + direction * forward_len
    back = anchor - direction * backward_len
    polygon = np.array(
        [
            back - normal * half_width,
            back + normal * half_width,
            front + normal * half_width,
            front - normal * half_width,
        ],
        dtype=np.float32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon, float(zone_length), float(zone_width)


def make_road_aligned_forklift_work_zone_from_base_rect(
    image_shape,
    forklift_box,
    base_rect_work_zone,
    road_direction_angle,
    road_mask=None,
    fence_group_boxes=None,
    use_fence_clip=True,
    use_road_mask_clip=True,
    max_length_scale=2.2,
    max_width_scale=1.3,
    forward_ratio=0.65,
    backward_ratio=0.35,
    fence_clip_padding=80,
    road_zone_min_retain_ratio=0.2,
    max_aligned_area_ratio=1.5,
    enable_fence_clipped_zone=True,
    fence_barrier_thickness=20,
) -> Dict[str, Any]:
    base_rect_work_zone = clip_box(base_rect_work_zone, image_shape[1], image_shape[0])
    base_mask = box_to_mask(image_shape, base_rect_work_zone)
    base_area = int(cv2.countNonZero(base_mask))
    result = {
        "mask": None,
        "polygon": None,
        "rotated_base_polygon": None,
        "fence_clip_polygon": None,
        "fence_barrier_polygons": [],
        "fence_barrier_boxes": [],
        "success": False,
        "source": "base_rect_rotated_by_road_direction",
        "used_fence_clip": False,
        "used_road_mask_clip": False,
        "used_road_mask_intersection": False,
        "fence_clipped_work_zone": {
            "enabled": bool(enable_fence_clipped_zone),
            "success": False,
            "used_road_mask": False,
            "used_fence_barrier": False,
            "kept_component_contains_forklift": False,
            "area_ratio": 0.0,
        },
        "base_rect_work_zone": int_box(base_rect_work_zone),
        "base_area": int(base_area),
        "aligned_area": 0,
        "aligned_area_ratio": 0.0,
    }
    if road_direction_angle is None or base_area <= 0:
        return result

    def build_candidate(length_factor: float = 1.0, width_factor: float = 1.0) -> Dict[str, Any]:
        polygon, _, _ = rotated_zone_polygon_from_base_rect(
            image_shape,
            forklift_box,
            base_rect_work_zone,
            road_direction_angle,
            max_length_scale=max_length_scale,
            max_width_scale=max_width_scale,
            forward_ratio=forward_ratio,
            backward_ratio=backward_ratio,
            length_factor=length_factor,
            width_factor=width_factor,
        )
        rotated_mask = polygon_mask(image_shape, polygon)
        raw_area = int(cv2.countNonZero(rotated_mask))
        mask = rotated_mask.copy()
        fence_clip = None
        used_fence_clip = False
        if use_fence_clip and fence_group_boxes:
            fence_clip = make_fence_clip_mask(
                image_shape,
                forklift_box,
                fence_group_boxes,
                padding_px=fence_clip_padding,
            )
            if fence_clip and fence_clip.get("mask") is not None:
                clipped = cv2.bitwise_and(mask, fence_clip["mask"])
                if cv2.countNonZero(clipped) > 0:
                    mask = clipped
                    used_fence_clip = True

        used_road_mask_clip = False
        road_mask_candidate = mask.copy()
        if use_road_mask_clip and road_mask is not None:
            before_area = int(cv2.countNonZero(mask))
            clipped = cv2.bitwise_and(mask, road_mask)
            clipped_area = int(cv2.countNonZero(clipped))
            retain_ratio = float(clipped_area / before_area) if before_area > 0 else 0.0
            if clipped_area > 0 and retain_ratio >= float(road_zone_min_retain_ratio):
                mask = clipped
                road_mask_candidate = clipped.copy()
                used_road_mask_clip = True

        fence_barrier = make_fence_barrier_mask(
            image_shape,
            fence_group_boxes,
            thickness=fence_barrier_thickness,
        ) if enable_fence_clipped_zone and fence_group_boxes else None
        final_mask = mask.copy()
        used_fence_barrier = False
        kept_component_contains_forklift = False
        if fence_barrier and fence_barrier.get("mask") is not None:
            candidate_zone = road_mask_candidate.copy()
            candidate_zone[fence_barrier["mask"] > 0] = 0
            kept_mask, kept_component_contains_forklift = keep_component_containing_point(
                candidate_zone,
                bottom_center(forklift_box),
            )
            if kept_mask is not None and kept_component_contains_forklift:
                final_mask = kept_mask
                used_fence_barrier = True

        aligned_area = int(cv2.countNonZero(final_mask))
        fence_clipped_area_ratio = round(float(aligned_area / raw_area), 4) if raw_area > 0 else 0.0
        return {
            "mask": final_mask if aligned_area > 0 else None,
            "polygon": [[int(x), int(y)] for x, y in np.round(polygon).astype(np.int32).tolist()],
            "rotated_base_polygon": [[int(x), int(y)] for x, y in np.round(polygon).astype(np.int32).tolist()],
            "polygons": mask_to_polygons(final_mask),
            "fence_clip_polygon": fence_clip.get("polygon") if fence_clip else None,
            "fence_barrier_polygons": fence_barrier.get("polygons", []) if fence_barrier else [],
            "fence_barrier_boxes": fence_barrier.get("boxes", []) if fence_barrier else [],
            "success": bool(aligned_area > 0),
            "used_fence_clip": bool(used_fence_clip),
            "used_road_mask_clip": bool(used_road_mask_clip),
            "used_road_mask_intersection": bool(used_road_mask_clip),
            "fence_clipped_work_zone": {
                "enabled": bool(enable_fence_clipped_zone),
                "success": bool(used_fence_barrier and aligned_area > 0),
                "used_road_mask": bool(used_road_mask_clip),
                "used_fence_barrier": bool(used_fence_barrier),
                "kept_component_contains_forklift": bool(kept_component_contains_forklift),
                "area_ratio": fence_clipped_area_ratio,
            },
            "raw_area": int(raw_area),
            "aligned_area": int(aligned_area),
            "aligned_area_ratio": round(float(aligned_area / base_area), 4) if base_area > 0 else 0.0,
        }

    candidate = build_candidate()
    if candidate["success"] and candidate["aligned_area"] > base_area * float(max_aligned_area_ratio):
        shrink = max(0.1, float(max_aligned_area_ratio) * float(base_area) / max(1.0, float(candidate["aligned_area"])))
        candidate = build_candidate(length_factor=np.sqrt(shrink), width_factor=np.sqrt(shrink))

    if not candidate["success"] or candidate["aligned_area"] > base_area * float(max_aligned_area_ratio):
        return result

    result.update(candidate)
    return result


def make_road_aligned_forklift_work_zone(
    image_shape,
    forklift_box,
    road_direction_angle,
    road_mask=None,
    length_scale=2.2,
    width_scale=1.2,
    back_scale=0.6,
) -> Dict[str, Any]:
    height, width = image_shape[:2]
    if road_direction_angle is None:
        return {"mask": None, "polygon": None, "used_road_mask_intersection": False, "success": False}

    ground_x, ground_y = bottom_center(forklift_box)
    x1, y1, x2, y2 = forklift_box
    forklift_width = max(1.0, float(x2 - x1))
    forklift_height = max(1.0, float(y2 - y1))
    base_size = max(forklift_width, forklift_height)
    forward_len = base_size * float(length_scale)
    backward_len = base_size * float(back_scale)
    half_width = forklift_width * float(width_scale)

    direction = np.array([np.cos(road_direction_angle), np.sin(road_direction_angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    anchor = np.array([ground_x, ground_y], dtype=np.float32)
    front = anchor + direction * forward_len
    back = anchor - direction * backward_len
    polygon = np.array(
        [
            back - normal * half_width,
            back + normal * half_width,
            front + normal * half_width,
            front - normal * half_width,
        ],
        dtype=np.float32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    polygon_int = np.round(polygon).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_int.reshape(-1, 1, 2)], 255)
    used_road_mask_intersection = road_mask is not None
    if used_road_mask_intersection:
        mask = cv2.bitwise_and(mask, road_mask)
    success = cv2.countNonZero(mask) > 0
    return {
        "mask": mask if success else None,
        "polygon": [[int(x), int(y)] for x, y in polygon_int.tolist()],
        "used_road_mask_intersection": bool(used_road_mask_intersection),
        "success": bool(success),
    }


def build_road_aligned_work_zones(
    image,
    forklift_boxes,
    fence_boxes=None,
    enable=True,
    left_scale=1.2,
    right_scale=1.2,
    front_scale=1.0,
    back_scale=0.35,
    height_depth_ratio=0.6,
    width_depth_ratio=0.8,
    fence_iou_thres=0.01,
    road_sat_max=80,
    road_value_min=25,
    road_value_max=220,
    road_morph_kernel=15,
    road_direction_roi_scale=3.0,
    road_direction_min_pixels=500,
    road_zone_length_scale=2.2,
    road_zone_width_scale=1.2,
    road_zone_back_scale=0.6,
    road_zone_intersect_road_mask=True,
    road_zone_from_base_rect=True,
    road_zone_max_length_scale=2.2,
    road_zone_max_width_scale=1.3,
    road_zone_forward_ratio=0.65,
    road_zone_backward_ratio=0.35,
    enable_fence_clip=True,
    fence_clip_padding=80,
    road_zone_min_retain_ratio=0.2,
    max_aligned_area_ratio=1.5,
    enable_fence_clipped_zone=True,
    fence_barrier_thickness=20,
) -> Dict[int, Dict[str, Any]]:
    zones = {}
    fence_boxes = fence_boxes or []
    for forklift_id, forklift_box in enumerate(forklift_boxes, start=1):
        ground_point = bottom_center(forklift_box)
        base_size = max(float(forklift_box[2] - forklift_box[0]), float(forklift_box[3] - forklift_box[1]), 1.0)
        base_rect_work_zone, _ = make_forklift_ground_zone(
            forklift_box,
            image.shape,
            left_scale,
            right_scale,
            front_scale,
            back_scale,
            height_depth_ratio,
            width_depth_ratio,
        )
        fence_group_boxes = [
            fence_box
            for fence_box in fence_boxes
            if box_iou(fence_box, base_rect_work_zone) > fence_iou_thres or point_in_box(box_center(fence_box), base_rect_work_zone)
        ]
        zone = {
            "mask": None,
            "enabled": bool(enable),
            "success": False,
            "road_direction_angle": None,
            "direction_confidence": 0.0,
            "polygon": None,
            "rotated_base_polygon": None,
            "source": "base_rect_rotated_by_road_direction" if road_zone_from_base_rect else "forklift_size_rotated_by_road_direction",
            "base_rect_work_zone": int_box(base_rect_work_zone),
            "fence_clip_polygon": None,
            "fence_barrier_polygons": [],
            "fence_barrier_boxes": [],
            "used_fence_clip": False,
            "used_road_mask_clip": False,
            "used_road_mask_intersection": False,
            "fence_clipped_work_zone": {
                "enabled": bool(enable_fence_clipped_zone),
                "success": False,
                "used_road_mask": False,
                "used_fence_barrier": False,
                "kept_component_contains_forklift": False,
                "area_ratio": 0.0,
            },
            "base_area": int(cv2.countNonZero(box_to_mask(image.shape, base_rect_work_zone))),
            "aligned_area": 0,
            "aligned_area_ratio": 0.0,
        }
        if enable:
            road_mask = generate_road_mask(
                image,
                ground_point,
                road_sat_max=road_sat_max,
                road_value_min=road_value_min,
                road_value_max=road_value_max,
                road_morph_kernel=road_morph_kernel,
            )
            angle, confidence = estimate_local_road_direction(
                road_mask,
                ground_point,
                base_size * float(road_direction_roi_scale),
                min_pixels=road_direction_min_pixels,
            )
            if angle is not None:
                if road_zone_from_base_rect:
                    built = make_road_aligned_forklift_work_zone_from_base_rect(
                        image.shape,
                        forklift_box,
                        base_rect_work_zone,
                        angle,
                        road_mask if road_zone_intersect_road_mask else None,
                        fence_group_boxes=fence_group_boxes,
                        use_fence_clip=enable_fence_clip,
                        use_road_mask_clip=road_zone_intersect_road_mask,
                        max_length_scale=road_zone_max_length_scale,
                        max_width_scale=road_zone_max_width_scale,
                        forward_ratio=road_zone_forward_ratio,
                        backward_ratio=road_zone_backward_ratio,
                        fence_clip_padding=fence_clip_padding,
                        road_zone_min_retain_ratio=road_zone_min_retain_ratio,
                        max_aligned_area_ratio=max_aligned_area_ratio,
                        enable_fence_clipped_zone=enable_fence_clipped_zone,
                        fence_barrier_thickness=fence_barrier_thickness,
                    )
                else:
                    built = make_road_aligned_forklift_work_zone(
                        image.shape,
                        forklift_box,
                        angle,
                        road_mask if road_zone_intersect_road_mask else None,
                        length_scale=road_zone_length_scale,
                        width_scale=road_zone_width_scale,
                        back_scale=road_zone_back_scale,
                    )
                zone.update(built)
                zone["road_direction_angle"] = round(float(np.degrees(angle)), 2)
                zone["direction_confidence"] = round(float(confidence), 4)
        zones[forklift_id] = zone
    return zones


def build_road_constrained_work_area(
    ground_point_work_area: Optional[Dict[str, Any]],
    road_surface_mask: Optional[np.ndarray],
    road_mask_success: bool,
) -> Dict[str, Any]:
    gp_mask = ground_point_work_area.get("mask") if ground_point_work_area else None
    if gp_mask is None:
        return {
            "mask": None,
            "enabled": False,
            "type": "ground_point_work_area_intersect_road_mask",
            "success": False,
            "person_count": 0,
            "polygons": [],
        }

    if road_mask_success and road_surface_mask is not None:
        mask = cv2.bitwise_and(gp_mask, road_surface_mask)
        success = cv2.countNonZero(mask) > 0
    else:
        mask = gp_mask
        success = False

    return {
        "mask": mask,
        "enabled": True,
        "type": "ground_point_work_area_intersect_road_mask",
        "success": bool(success),
        "person_count": 0,
        "polygons": mask_to_polygons(mask),
    }


def is_person_in_mask_area(person_box, mask: Optional[np.ndarray]) -> bool:
    return first_point_in_mask(person_foot_points(person_box), mask) is not None


def point_in_polygon(point: Tuple[float, float], polygon: Optional[List[List[int]]]) -> bool:
    if not polygon:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def evaluate_helmet_status(
    detections: List[Dict[str, Any]],
    image_shape,
    no_helmet_overlap: float,
) -> List[Dict[str, Any]]:
    return evaluate_persons(detections, image_shape, no_helmet_overlap)


def draw_transparent_zone(
    image,
    zone: Sequence[float],
    color: Tuple[int, int, int] = (0, 165, 255),
    alpha: float = 0.25,
) -> None:
    x1, y1, x2, y2 = int_box(zone) or [0, 0, 0, 0]
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def draw_transparent_polygon(
    image,
    polygon: Optional[List[List[int]]],
    color: Tuple[int, int, int] = (255, 120, 0),
    alpha: float = 0.22,
) -> None:
    if not polygon or len(polygon) < 3:
        return
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)


def ground_point_work_area_to_json(
    ground_point_work_area: Optional[Dict[str, Any]],
    enabled: bool,
    padding_px: float,
    fallback_zone_radius: float,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "type": None,
            "padding_px": int(round(padding_px)),
            "fallback_zone_radius": int(round(fallback_zone_radius)),
            "polygon": None,
            "ground_points": {"forklifts": [], "fences": []},
        }
    if ground_point_work_area is None:
        return {
            "enabled": True,
            "type": None,
            "padding_px": int(round(padding_px)),
            "fallback_zone_radius": int(round(fallback_zone_radius)),
            "polygon": None,
            "ground_points": {"forklifts": [], "fences": []},
        }
    return {
        "enabled": True,
        "type": ground_point_work_area["type"],
        "padding_px": ground_point_work_area["padding_px"],
        "fallback_zone_radius": ground_point_work_area["fallback_zone_radius"],
        "polygon": ground_point_work_area["polygon"],
        "ground_points": ground_point_work_area["ground_points"],
    }


def build_compliance_summary(forklift_results: List[Dict[str, Any]], helmet_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    helmet_status_by_box = {tuple(item["box"]): item["status"] for item in helmet_results}
    excluded_boxes = {tuple(person["box"]) for forklift in forklift_results for person in forklift["excluded_persons"]}

    helmet_unknown = sum(1 for item in helmet_results if item["status"] == "unknown")
    helmet_no_helmet = sum(1 for item in helmet_results if item["status"] == "no_helmet")
    helmet_result = (
        "unsafe"
        if helmet_no_helmet > 0
        else ("unknown" if not helmet_results or helmet_unknown > 0 else "safe")
    )

    driver_statuses = [helmet_status_by_box.get(box, "unknown") for box in excluded_boxes]
    driver_helmet_no_helmet = sum(1 for status in driver_statuses if status == "no_helmet")
    driver_helmet_unknown = sum(1 for status in driver_statuses if status == "unknown")
    if not driver_statuses:
        driver_helmet_result = "unknown"
    elif driver_helmet_no_helmet > 0:
        driver_helmet_result = "unsafe"
    elif driver_helmet_unknown > 0:
        driver_helmet_result = "unknown"
    else:
        driver_helmet_result = "safe"

    has_forklift = bool(forklift_results)
    has_fence_nearby = has_forklift and all(item["has_fence_nearby"] for item in forklift_results)
    has_person_nearby = any(item["has_person_nearby"] for item in forklift_results)
    fence_result = "unknown" if not has_forklift else ("safe" if has_fence_nearby else "unsafe")
    work_radius_result = "unknown" if not has_forklift else ("unsafe" if has_person_nearby else "safe")

    forklift_is_safe = has_forklift and has_fence_nearby and not has_person_nearby
    if helmet_result == "unsafe" or driver_helmet_result == "unsafe" or (has_forklift and not forklift_is_safe):
        overall_result = "unsafe"
    elif has_forklift:
        overall_result = "safe"
    else:
        overall_result = "unknown"

    return {
        "helmet_result": helmet_result,
        "driver_helmet_result": driver_helmet_result,
        "fence_result": fence_result,
        "work_radius_result": work_radius_result,
        "overall_result": overall_result,
        "helmet_no_helmet_count": helmet_no_helmet,
        "helmet_unknown_count": helmet_unknown,
        "driver_no_helmet_count": driver_helmet_no_helmet,
        "driver_unknown_count": driver_helmet_unknown,
    }


def evaluate_forklifts(
    detections: List[Dict[str, Any]],
    image_shape,
    left_scale: float,
    right_scale: float,
    front_scale: float,
    back_scale: float,
    fence_iou_thres: float,
    height_depth_ratio: float,
    width_depth_ratio: float,
    driver_iou_thres: float,
    driver_inside_ratio_thres: float,
    driver_dx_norm_thres: float,
    driver_min_dy_up_norm: float,
    driver_max_dy_up_norm: float,
    person_distance_width_scale: float,
    person_distance_height_scale: float,
    min_person_distance_px: float,
    enable_fence_zone: bool,
    fence_zone_padding: float,
    nearby_mode: str = "circle",
    forklift_radius_scale: float = 1.2,
    min_forklift_radius_px: float = 180.0,
    max_forklift_radius_px: float = 450.0,
    camera_mode: str = "oblique",
    ground_point_work_area: Optional[Dict[str, Any]] = None,
    road_constrained_work_area: Optional[Dict[str, Any]] = None,
    road_aligned_work_zones: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    forklifts = filter_detections(detections, "forklift")
    persons = filter_detections(detections, "person")
    fences = filter_detections(detections, "fence")

    results = []
    for forklift_index, forklift in enumerate(forklifts, start=1):
        forklift_id = forklift.get("track_id") if forklift.get("track_id") is not None else forklift_index
        forklift_ground_point = bottom_center(forklift["box"])
        work_zone, ground_depth_px = make_forklift_ground_zone(
            forklift["box"],
            image_shape,
            left_scale,
            right_scale,
            front_scale,
            back_scale,
            height_depth_ratio,
            width_depth_ratio,
        )
        road_aligned_work_zone = (road_aligned_work_zones or {}).get(
            forklift_index,
            {
                "mask": None,
                "enabled": False,
                "success": False,
                "road_direction_angle": None,
                "direction_confidence": 0.0,
                "polygon": None,
                "used_road_mask_intersection": False,
            },
        )
        road_direction_angle = road_aligned_work_zone.get("road_direction_angle")
        road_direction_confidence = float(road_aligned_work_zone.get("direction_confidence", 0.0) or 0.0)
        circle_work_zone = build_circle_work_zone(
            forklift["box"],
            work_zone,
            forklift_radius_scale,
            min_forklift_radius_px,
            max_forklift_radius_px,
            road_direction_angle=road_direction_angle,
            road_direction_confidence=road_direction_confidence,
            camera_mode=camera_mode,
        )
        x1, y1, x2, y2 = forklift["box"]
        forklift_width = x2 - x1
        forklift_height = y2 - y1
        near_threshold_px = max(
            forklift_width * person_distance_width_scale,
            forklift_height * person_distance_height_scale,
            min_person_distance_px,
        )
        excluded_persons = []
        nearby_persons = []
        # Fence compliance is a frame-level presence rule: when at least one
        # fence and one forklift are detected in the same frame, every detected
        # forklift is considered fenced.  Keep the spatially matched subset
        # separate because it is still useful for constructing the optional
        # fence-shaped person proximity zone.
        nearby_fences = []
        spatially_nearby_fences = []

        for fence_id, fence in enumerate(fences, start=1):
            fence_iou = box_iou(fence["box"], work_zone)
            fence_item = {
                "fence_id": fence_id,
                "box": int_box(fence["box"]),
                "iou_with_work_zone": round(float(fence_iou), 4),
            }
            nearby_fences.append(fence_item)
            if fence_iou > fence_iou_thres or point_in_box(box_center(fence["box"]), work_zone):
                spatially_nearby_fences.append(fence_item)

        fence_based_work_zone = (
            make_fence_based_work_zone(
                forklift["box"],
                spatially_nearby_fences,
                image_shape,
                fence_zone_padding,
            )
            if enable_fence_zone
            else None
        )
        ground_point_work_area_mask = ground_point_work_area.get("mask") if ground_point_work_area else None
        detection_rules_used = ["rect_work_zone", "ground_point_distance"]
        if fence_based_work_zone is not None:
            detection_rules_used.append("fence_based_work_zone")
        if ground_point_work_area_mask is not None:
            detection_rules_used.append("ground_point_work_area")
        if nearby_mode in {"original", "both"}:
            detection_rules_used.append("original_nearby_rules")
        if nearby_mode in {"circle", "both"}:
            detection_rules_used.append("circle_work_zone")
        road_constrained_work_area_mask = road_constrained_work_area.get("mask") if road_constrained_work_area else None
        use_road_constrained_area = bool(
            road_constrained_work_area
            and road_constrained_work_area.get("enabled")
            and road_constrained_work_area.get("success")
            and road_constrained_work_area_mask is not None
        )
        if road_constrained_work_area_mask is not None:
            detection_rules_used.append("road_constrained_work_area")
        road_aligned_work_zone_mask = road_aligned_work_zone.get("mask")
        if road_aligned_work_zone.get("success") and road_aligned_work_zone_mask is not None:
            detection_rules_used.append("road_aligned_forklift_work_zone")

        for person_index, person in enumerate(persons, start=1):
            person_id = person.get("track_id") if person.get("track_id") is not None else person_index
            person_ground_point = bottom_center(person["box"])
            relative_metrics = ground_relative_position(person_ground_point, forklift["box"], forklift_ground_point)
            person_forklift_iou = box_iou(person["box"], forklift["box"])
            person_inside_forklift_ratio = overlap_ratio(person["box"], forklift["box"])
            person_box_fully_inside_forklift = is_box_fully_inside(person["box"], forklift["box"])
            if is_driver_or_excluded_person(
                person["box"],
                person_ground_point,
                forklift["box"],
                forklift_ground_point,
                driver_iou_thres,
                driver_inside_ratio_thres,
                driver_dx_norm_thres,
                driver_min_dy_up_norm,
                driver_max_dy_up_norm,
            ):
                excluded_persons.append(
                    {
                        "person_id": person_id,
                        "track_id": person.get("track_id"),
                        "box": int_box(person["box"]),
                        "ground_point": [round(person_ground_point[0], 2), round(person_ground_point[1], 2)],
                        "dx_norm": round(relative_metrics["dx_norm"], 4),
                        "dy_up_norm": round(relative_metrics["dy_up_norm"], 4),
                        "person_forklift_iou": round(float(person_forklift_iou), 4),
                        "person_inside_forklift_ratio": round(float(person_inside_forklift_ratio), 4),
                        "person_box_fully_inside_forklift": person_box_fully_inside_forklift,
                        "reason": (
                            "person_box_fully_inside_forklift"
                            if person_box_fully_inside_forklift
                            else "driver_or_on_forklift"
                        ),
                    }
                )
                continue

            original_result = check_person_nearby_original_rules(
                person,
                person_id,
                forklift,
                forklift_id,
                forklift_ground_point,
                work_zone,
                fence_based_work_zone,
                ground_point_work_area_mask,
                road_constrained_work_area_mask if use_road_constrained_area else None,
                road_aligned_work_zone,
                near_threshold_px,
            )
            circle_result = check_person_nearby_circle_rule(
                person,
                forklift["box"],
                forklift_ground_point,
                circle_work_zone,
            )
            original_matched = bool(original_result["matched"])
            circle_matched = bool(circle_result["matched"])
            if nearby_mode == "original":
                person_nearby = original_matched
            elif nearby_mode == "circle":
                person_nearby = circle_matched
            else:
                person_nearby = original_matched or circle_matched

            if person_nearby:
                distance_px = min(
                    float(original_result["debug_info"].get("distance_to_forklift_px", 0.0) or 0.0),
                    float(circle_result["debug_info"].get("distance_to_forklift_px", 0.0) or 0.0),
                )
                reasons = []
                if original_matched:
                    reasons.extend(original_result["reasons"])
                if circle_matched:
                    reasons.extend(circle_result["reasons"])
                if reasons:
                    reasons = list(dict.fromkeys(reasons))
                in_rect_work_zone = original_result["debug_info"]["in_rect_work_zone"]
                near_by_distance = original_result["debug_info"]["near_by_distance"]
                in_fence_based_work_zone = original_result["debug_info"]["in_fence_based_work_zone"]
                in_ground_point_work_area = original_result["debug_info"]["in_ground_point_work_area"]
                in_road_constrained_work_area = original_result["debug_info"]["in_road_constrained_work_area"]
                in_road_aligned_forklift_work_zone = original_result["debug_info"]["in_road_aligned_forklift_work_zone"]
                in_fence_clipped_forklift_work_zone = original_result["debug_info"]["in_fence_clipped_forklift_work_zone"]
                matched_by_original_rules = original_matched
                matched_by_circle_rule = circle_matched
                in_circle_work_zone = circle_matched
                nearby_persons.append(
                    {
                        "person_id": person_id,
                        "track_id": person.get("track_id"),
                        "box": int_box(person["box"]),
                        "ground_point": [round(person_ground_point[0], 2), round(person_ground_point[1], 2)],
                        "foot_points": original_result["debug_info"].get("person_foot_points", []),
                        "closest_foot_point": original_result["debug_info"].get("closest_foot_point"),
                        "distance_to_forklift_px": round(float(distance_px), 2),
                        "near_threshold_px": round(float(near_threshold_px), 2),
                        "dx_norm": round(relative_metrics["dx_norm"], 4),
                        "dy_up_norm": round(relative_metrics["dy_up_norm"], 4),
                        "nearby_mode": nearby_mode,
                        "matched_by_original_rules": matched_by_original_rules,
                        "matched_by_circle_rule": matched_by_circle_rule,
                        "in_circle_work_zone": in_circle_work_zone,
                        "in_ground_point_work_area": bool(in_ground_point_work_area),
                        "in_road_constrained_work_area": bool(in_road_constrained_work_area),
                        "in_road_aligned_forklift_work_zone": bool(in_road_aligned_forklift_work_zone),
                        "in_fence_clipped_forklift_work_zone": bool(in_fence_clipped_forklift_work_zone),
                        "reason": reasons[0] if len(reasons) == 1 else reasons,
                        "reasons": reasons,
                    }
                )

        violations = []
        if nearby_persons:
            violations.append("PERSON_NEAR_FORKLIFT")
        if not nearby_fences:
            violations.append("NO_FENCE_NEAR_FORKLIFT")
        is_safe = not violations
        if is_safe:
            message = "safe: fence is detected in frame and no person is nearby"
        elif violations == ["PERSON_NEAR_FORKLIFT"]:
            message = "unsafe: person is inside forklift work zone"
        elif violations == ["NO_FENCE_NEAR_FORKLIFT"]:
            message = "unsafe: no fence is detected in frame"
        else:
            message = "unsafe: person is inside forklift work zone and no fence is detected in frame"

        results.append(
            {
                "forklift_id": forklift_id,
                "forklift_box": int_box(forklift["box"]),
                "forklift_ground_point": [round(forklift_ground_point[0], 2), round(forklift_ground_point[1], 2)],
                "circle_work_zone": circle_work_zone if nearby_mode in {"circle", "both"} else {**circle_work_zone, "enabled": False},
                "ground_depth_px": round(float(ground_depth_px), 2),
                "near_threshold_px": round(float(near_threshold_px), 2),
                "work_zone": int_box(work_zone),
                "fence_based_work_zone": fence_based_work_zone,
                "road_aligned_work_zone": {
                    "enabled": bool(road_aligned_work_zone.get("enabled")),
                    "success": bool(road_aligned_work_zone.get("success")),
                    "source": road_aligned_work_zone.get("source"),
                    "road_direction_angle": road_aligned_work_zone.get("road_direction_angle"),
                    "direction_confidence": road_aligned_work_zone.get("direction_confidence", 0.0),
                    "base_rect_work_zone": road_aligned_work_zone.get("base_rect_work_zone"),
                    "polygon": road_aligned_work_zone.get("polygon"),
                    "rotated_base_polygon": road_aligned_work_zone.get("rotated_base_polygon"),
                    "polygons": road_aligned_work_zone.get("polygons", []),
                    "fence_clip_polygon": road_aligned_work_zone.get("fence_clip_polygon"),
                    "fence_barrier_polygons": road_aligned_work_zone.get("fence_barrier_polygons", []),
                    "fence_barrier_boxes": road_aligned_work_zone.get("fence_barrier_boxes", []),
                    "used_fence_clip": bool(road_aligned_work_zone.get("used_fence_clip")),
                    "used_road_mask_clip": bool(road_aligned_work_zone.get("used_road_mask_clip")),
                    "used_road_mask_intersection": bool(road_aligned_work_zone.get("used_road_mask_intersection")),
                    "fence_clipped_work_zone": road_aligned_work_zone.get("fence_clipped_work_zone"),
                    "base_area": int(road_aligned_work_zone.get("base_area", 0) or 0),
                    "aligned_area": int(road_aligned_work_zone.get("aligned_area", 0) or 0),
                    "aligned_area_ratio": float(road_aligned_work_zone.get("aligned_area_ratio", 0.0) or 0.0),
                },
                "detection_rules_used": detection_rules_used,
                "excluded_persons": excluded_persons,
                "has_person_nearby": bool(nearby_persons),
                "nearby_persons": nearby_persons,
                "has_fence_nearby": bool(nearby_fences),
                "nearby_fences": nearby_fences,
                "violations": violations,
                "is_safe": is_safe,
                "message": message,
            }
        )
    return results


def draw_results(
    image,
    forklift_results: List[Dict[str, Any]],
    helmet_results: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    ground_point_work_area: Optional[Dict[str, Any]] = None,
    draw_ground_point_work_area: bool = True,
    road_surface_mask: Optional[np.ndarray] = None,
    draw_road_mask: bool = True,
    road_mask_success: bool = False,
    road_constrained_work_area: Optional[Dict[str, Any]] = None,
    draw_road_aligned_zone: bool = True,
    draw_fence_barrier: bool = True,
    nearby_mode: str = "circle",
    draw_circle_work_zone: bool = True,
    compliance_summary: Optional[Dict[str, Any]] = None,
) -> None:
    nearby_person_boxes = {tuple(person["box"]) for f in forklift_results for person in f["nearby_persons"]}
    nearby_person_by_box = {
        tuple(person["box"]): person
        for forklift in forklift_results
        for person in forklift["nearby_persons"]
    }
    excluded_person_by_box = {
        tuple(person["box"]): person
        for forklift in forklift_results
        for person in forklift["excluded_persons"]
    }
    excluded_person_boxes = {tuple(person["box"]) for f in forklift_results for person in f["excluded_persons"]}
    nearby_fence_boxes = {tuple(fence["box"]) for f in forklift_results for fence in f["nearby_fences"]}
    helmet_by_box = {tuple(person["box"]): person for person in helmet_results}

    # Fence, road-mask and ground-point geometry remain available to the rules,
    # but they are intentionally not rendered. The only work-area overlay is
    # the forklift radius (circle/ellipse) below.
    for forklift in forklift_results:
        has_violation = bool(forklift["violations"])
        color = (0, 0, 255) if has_violation else (0, 180, 0)
        circle_work_zone = forklift.get("circle_work_zone", {})
        circle_polygon = circle_work_zone.get("polygon") or []
        if (
            nearby_mode in {"circle", "both"}
            and draw_circle_work_zone
            and circle_work_zone.get("enabled")
            and circle_work_zone.get("center")
            and len(circle_polygon) >= 3
        ):
            pts = np.array(circle_polygon, dtype=np.int32).reshape(-1, 1, 2)
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (0, 210, 255))
            cv2.addWeighted(overlay, 0.24, image, 0.76, 0, image)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 140, 255), thickness=4, lineType=cv2.LINE_AA)
            label_point = tuple(circle_polygon[0])
            cv2.putText(
                image,
                f"FORKLIFT RADIUS {circle_work_zone.get('radius_x_px', 0):.0f}x{circle_work_zone.get('radius_y_px', 0):.0f}",
                (int(label_point[0]), max(15, int(label_point[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
        draw_box(image, forklift["forklift_box"], f'forklift {forklift["forklift_id"]}', color, 2)

    for det in filter_detections(detections, "person"):
        box = tuple(int_box(det["box"]))
        helmet_status = helmet_by_box.get(box, {}).get("status", "unknown")
        if box in excluded_person_boxes:
            color = (255, 120, 0)
            if helmet_status == "helmet":
                label = "driver helmet OK"
                color = (0, 180, 0)
            elif helmet_status == "no_helmet":
                label = "driver helmet NG"
                color = (0, 0, 255)
            else:
                label = f"driver/excluded {helmet_status}"
            thickness = 2
            px, py = excluded_person_by_box[box]["ground_point"]
            cv2.circle(image, (int(round(px)), int(round(py))), 4, color, -1)
        elif box in nearby_person_boxes:
            color = (0, 0, 255)
            nearby_item = nearby_person_by_box[box]
            if nearby_item.get("matched_by_circle_rule") and nearby_mode in {"circle", "both"} and not nearby_item.get("matched_by_original_rules"):
                label = "PERSON IN CIRCLE ZONE"
            elif nearby_item.get("matched_by_circle_rule") and nearby_item.get("matched_by_original_rules"):
                label = "PERSON NEAR: circle/original"
            elif nearby_item.get("in_fence_clipped_forklift_work_zone"):
                label = "PERSON IN FENCE-CLIPPED ZONE"
            elif nearby_item.get("in_road_aligned_forklift_work_zone"):
                label = "PERSON IN ROAD-ALIGNED ZONE"
            elif nearby_item.get("in_road_constrained_work_area"):
                label = "PERSON IN ROAD WORK AREA"
            elif nearby_item.get("in_ground_point_work_area"):
                label = "PERSON IN GP AREA"
            else:
                label = f'PERSON NEAR, dist={nearby_item["distance_to_forklift_px"]:.0f} px'
            thickness = 2
            px, py = nearby_item["ground_point"]
            cv2.circle(image, (int(round(px)), int(round(py))), 4, color, -1)
        else:
            color = HELMET_STATUS_COLORS.get(helmet_status, (180, 180, 180))
            label = f"person {helmet_status}"
            thickness = 1
            px, py = bottom_center(det["box"])
            cv2.circle(image, (int(round(px)), int(round(py))), 3, color, -1)
        draw_box(image, det["box"], label, color, thickness)
        if box in helmet_by_box:
            draw_box(image, helmet_by_box[box]["_head_region"], "", color, 1)

    for det in filter_detections(detections, "fence"):
        box = tuple(int_box(det["box"]))
        color = (0, 180, 0) if box in nearby_fence_boxes else (180, 180, 180)
        label = "fence detected" if box in nearby_fence_boxes else "fence"
        draw_box(image, det["box"], label, color, 2 if box in nearby_fence_boxes else 1)

    for det in filter_detections(detections, "helmet"):
        draw_box(image, det["box"], f'helmet {det["confidence"]:.2f}', (0, 160, 0), 1)
    for det in filter_detections(detections, "head_no_helmet"):
        draw_box(image, det["box"], f'no_helmet {det["confidence"]:.2f}', (0, 0, 220), 1)

    has_forklift = bool(forklift_results)
    has_fence_nearby = any(item["has_fence_nearby"] for item in forklift_results)
    has_person_nearby = any(item["has_person_nearby"] for item in forklift_results)
    excluded_person_count = sum(len(item["excluded_persons"]) for item in forklift_results)
    no_helmet_count = sum(1 for item in helmet_results if item["status"] == "no_helmet")
    compliance_summary = compliance_summary or build_compliance_summary(forklift_results, helmet_results)
    gp_area_enabled = bool(ground_point_work_area and ground_point_work_area.get("enabled"))
    gp_area_person_count = sum(
        1
        for forklift in forklift_results
        for person in forklift["nearby_persons"]
        if person.get("in_ground_point_work_area")
    )
    road_constrained_person_count = sum(
        1
        for forklift in forklift_results
        for person in forklift["nearby_persons"]
        if person.get("in_road_constrained_work_area")
    )
    road_aligned_success = any(item.get("road_aligned_work_zone", {}).get("success") for item in forklift_results)
    road_aligned_clipped_by_fence = any(item.get("road_aligned_work_zone", {}).get("used_fence_clip") for item in forklift_results)
    road_aligned_clipped_by_road_mask = any(item.get("road_aligned_work_zone", {}).get("used_road_mask_clip") for item in forklift_results)
    fence_clipped_success = any(
        item.get("road_aligned_work_zone", {}).get("fence_clipped_work_zone", {}).get("success")
        for item in forklift_results
    )
    road_aligned_ratios = [
        float(item.get("road_aligned_work_zone", {}).get("aligned_area_ratio", 0.0) or 0.0)
        for item in forklift_results
        if item.get("road_aligned_work_zone", {}).get("success")
    ]
    road_aligned_area_ratio = max(road_aligned_ratios) if road_aligned_ratios else 0.0
    forklift_is_safe = has_forklift and all(item["is_safe"] for item in forklift_results)
    helmet_is_safe = no_helmet_count == 0
    is_safe = forklift_is_safe and helmet_is_safe
    result = "safe" if is_safe else "unsafe"
    draw_text_lines(
        image,
        [
            f"has_person_nearby: {str(has_person_nearby).lower()}",
            f"excluded_person_count: {excluded_person_count}",
            f"has_fence_nearby: {str(has_fence_nearby).lower()}",
            f"gp_area_enabled: {str(gp_area_enabled).lower()}",
            f"gp_area_person_count: {gp_area_person_count}",
            f"road_mask_success: {str(road_mask_success).lower()}",
            f"road_constrained_person_count: {road_constrained_person_count}",
            f"road_aligned_success: {str(road_aligned_success).lower()}",
            f"road_aligned_clipped_by_fence: {str(road_aligned_clipped_by_fence).lower()}",
            f"road_aligned_clipped_by_road_mask: {str(road_aligned_clipped_by_road_mask).lower()}",
            f"road_aligned_area_ratio: {road_aligned_area_ratio:.2f}",
            f"fence_clipped_success: {str(fence_clipped_success).lower()}",
            f"no_helmet_count: {no_helmet_count}",
            f"nearby_mode: {nearby_mode}",
            f"result: {result}",
            f"helmet_result: {compliance_summary['helmet_result']}",
            f"driver_helmet_result: {compliance_summary['driver_helmet_result']}",
            f"fence_result: {compliance_summary['fence_result']}",
            f"work_radius_result: {compliance_summary['work_radius_result']}",
        ],
    )


def build_output(
    image_path: str,
    forklift_results: List[Dict[str, Any]],
    helmet_results: List[Dict[str, Any]],
    nearby_mode: str,
    fence_count: int = 0,
    ground_point_work_area_json: Optional[Dict[str, Any]] = None,
    road_surface_mask_json: Optional[Dict[str, Any]] = None,
    road_constrained_work_area_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compliance_summary = build_compliance_summary(forklift_results, helmet_results)
    has_forklift = bool(forklift_results)
    has_fence_nearby = any(item["has_fence_nearby"] for item in forklift_results)
    has_person_nearby = any(item["has_person_nearby"] for item in forklift_results)
    excluded_person_count = sum(len(item["excluded_persons"]) for item in forklift_results)
    public_persons = []
    for person in helmet_results:
        item = dict(person)
        item.pop("_head_region", None)
        item["helmet_status"] = item.get("status", "unknown")
        public_persons.append(item)

    forklift_is_safe = has_forklift and all(item["is_safe"] for item in forklift_results)
    no_helmet_count = sum(1 for p in public_persons if p["status"] == "no_helmet")
    helmet_is_safe = no_helmet_count == 0
    has_hazard = bool(no_helmet_count or any(item.get("violations") for item in forklift_results))
    is_safe = bool(has_forklift and forklift_is_safe and helmet_is_safe)
    violations = sorted({violation for forklift in forklift_results for violation in forklift.get("violations", [])})
    nearby_persons = [person for forklift in forklift_results for person in forklift.get("nearby_persons", [])]
    excluded_persons = [person for forklift in forklift_results for person in forklift.get("excluded_persons", [])]
    return {
        "image": str(image_path),
        "total_persons": len(public_persons),
        "person_count": len(public_persons),
        "helmet_worn_count": sum(1 for p in public_persons if p["status"] == "helmet"),
        "no_helmet_count": no_helmet_count,
        "unknown_count": sum(1 for p in public_persons if p["status"] == "unknown"),
        "persons": public_persons,
        "has_forklift": has_forklift,
        "fence_count": int(fence_count),
        "has_fence": bool(fence_count > 0),
        "has_fence_nearby": has_fence_nearby,
        "has_person_nearby": has_person_nearby,
        "excluded_person_count": excluded_person_count,
        "forklift_is_safe": forklift_is_safe,
        "helmet_is_safe": helmet_is_safe,
        "is_safe": is_safe,
        "result": "unsafe" if has_hazard else ("safe" if is_safe else "unknown"),
        "violations": violations,
        "nearby_persons": nearby_persons,
        "excluded_persons": excluded_persons,
        "nearby_mode": nearby_mode,
        "compliance_summary": compliance_summary,
        "ground_point_work_area": ground_point_work_area_json,
        "road_surface_mask": road_surface_mask_json,
        "road_constrained_work_area": road_constrained_work_area_json,
        "forklift_count": len(forklift_results),
        "forklifts": forklift_results,
    }


def enrich_forklift_person_helmet_status(
    forklift_results: List[Dict[str, Any]],
    helmet_results: List[Dict[str, Any]],
) -> None:
    helmet_status_by_box = {tuple(item.get("box") or []): item.get("status", "unknown") for item in helmet_results}
    for forklift in forklift_results:
        for key in ("nearby_persons", "excluded_persons"):
            for person in forklift.get(key, []):
                status = helmet_status_by_box.get(tuple(person.get("box") or []), "unknown")
                person["helmet_status"] = status


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forklift work-zone safety check for one image.")
    parser.add_argument("--weights", required=True, help="YOLO weights path, e.g. safety_best.pt")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output annotated image path")
    parser.add_argument("--json", help="Output JSON path. If omitted, JSON is saved beside the image output")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 for GPU or cpu for CPU")
    parser.add_argument("--no-helmet-overlap", type=float, default=0.2, help="Minimum head_no_helmet overlap with head region")
    parser.add_argument("--expand-scale", type=float, default=2.0, help="Deprecated; kept for command compatibility")
    parser.add_argument("--left-scale", type=float, default=1.2, help="Work-zone width scale to the left of forklift ground point")
    parser.add_argument("--right-scale", type=float, default=1.2, help="Work-zone width scale to the right of forklift ground point")
    parser.add_argument("--front-scale", type=float, default=1.0, help="Work-zone ground-depth scale above forklift ground point")
    parser.add_argument("--back-scale", type=float, default=0.35, help="Work-zone ground-depth scale below forklift ground point")
    parser.add_argument("--height-depth-ratio", type=float, default=0.6, help="Ground depth ratio based on forklift box height")
    parser.add_argument("--width-depth-ratio", type=float, default=0.8, help="Ground depth ratio based on forklift box width")
    parser.add_argument(
        "--fence-iou-thres",
        type=float,
        default=0.01,
        help="Minimum fence/work-zone IoU used only to build the optional fence-shaped person proximity zone",
    )
    parser.add_argument("--driver-iou-thres", type=float, default=0.05, help="Deprecated; kept for command compatibility")
    parser.add_argument(
        "--driver-inside-ratio-thres",
        type=float,
        default=0.4,
        help="Deprecated; kept for command compatibility",
    )
    parser.add_argument(
        "--driver-dx-norm-thres",
        type=float,
        default=0.45,
        help="Driver exclusion threshold for normalized horizontal ground-point distance",
    )
    parser.add_argument(
        "--driver-min-dy-up-norm",
        type=float,
        default=0.35,
        help="Driver exclusion minimum normalized upward ground-point offset; higher avoids excluding ground-standing persons behind forklift",
    )
    parser.add_argument(
        "--driver-max-dy-up-norm",
        type=float,
        default=0.85,
        help="Driver exclusion maximum normalized upward ground-point offset",
    )
    parser.add_argument(
        "--person-distance-width-scale",
        type=float,
        default=1.2,
        help="Nearby-person distance threshold scale based on forklift box width",
    )
    parser.add_argument(
        "--person-distance-height-scale",
        type=float,
        default=0.8,
        help="Nearby-person distance threshold scale based on forklift box height",
    )
    parser.add_argument(
        "--min-person-distance-px",
        type=float,
        default=180.0,
        help="Minimum nearby-person ground-point distance threshold in pixels",
    )
    parser.add_argument(
        "--nearby-mode",
        choices=["original", "circle", "both"],
        default="circle",
        help="Nearby-person mode: original, circle, or both",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["oblique", "overhead"],
        default="oblique",
        help="Camera view: oblique keeps perspective ellipse; overhead uses a box-center circle",
    )
    parser.add_argument("--forklift-radius-scale", type=float, default=1.2, help="Scale for circle nearby radius")
    parser.add_argument(
        "--min-forklift-radius-px",
        type=float,
        default=180.0,
        help="Deprecated compatibility option; no longer limits the work-zone radius",
    )
    parser.add_argument(
        "--max-forklift-radius-px",
        type=float,
        default=450.0,
        help="Deprecated compatibility option; no longer limits the work-zone radius",
    )
    parser.add_argument(
        "--draw-circle-work-zone",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Draw circle nearby work zone; accepts true/false",
    )
    parser.add_argument(
        "--enable-fence-zone",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Enable fence-based convex-hull work zone; accepts true/false",
    )
    parser.add_argument("--fence-zone-padding", type=float, default=40.0, help="Padding in pixels for fence-based work zone")
    parser.add_argument(
        "--enable-ground-point-work-area",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Enable global forklift/fence ground-point work area; accepts true/false",
    )
    parser.add_argument(
        "--ground-point-work-area-padding",
        type=float,
        default=80.0,
        help="Dilation padding in pixels for ground-point work area",
    )
    parser.add_argument(
        "--fallback-zone-radius",
        type=float,
        default=220.0,
        help="Fallback circle radius in pixels when fewer than 3 ground points exist",
    )
    parser.add_argument(
        "--draw-ground-point-work-area",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Draw global ground-point work area overlay; accepts true/false",
    )
    parser.add_argument(
        "--enable-road-mask",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Enable heuristic road surface mask; accepts true/false",
    )
    parser.add_argument(
        "--draw-road-mask",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Draw heuristic road surface mask overlay; accepts true/false",
    )
    parser.add_argument("--road-sat-max", type=int, default=80, help="Maximum HSV saturation for road candidate pixels")
    parser.add_argument("--road-value-min", type=int, default=25, help="Minimum HSV value for road candidate pixels")
    parser.add_argument("--road-value-max", type=int, default=220, help="Maximum HSV value for road candidate pixels")
    parser.add_argument("--road-morph-kernel", type=int, default=15, help="Morphology kernel size for road mask cleanup")
    parser.add_argument("--road-min-area-ratio", type=float, default=0.03, help="Minimum road mask area ratio")
    parser.add_argument(
        "--enable-road-aligned-zone",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Enable road-direction-aligned forklift work zone; accepts true/false",
    )
    parser.add_argument(
        "--draw-road-aligned-zone",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Draw road-direction-aligned forklift work zone; accepts true/false",
    )
    parser.add_argument("--road-direction-roi-scale", type=float, default=3.0, help="ROI scale for local road direction estimation")
    parser.add_argument("--road-direction-min-pixels", type=int, default=500, help="Minimum road pixels for PCA direction estimation")
    parser.add_argument("--road-zone-length-scale", type=float, default=2.2, help="Forward length scale for road-aligned zone")
    parser.add_argument("--road-zone-width-scale", type=float, default=1.2, help="Half-width scale for road-aligned zone")
    parser.add_argument("--road-zone-back-scale", type=float, default=0.6, help="Backward length scale for road-aligned zone")
    parser.add_argument(
        "--road-zone-intersect-road-mask",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Intersect road-aligned zone with local road mask; accepts true/false",
    )
    parser.add_argument(
        "--road-zone-from-base-rect",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Build road-aligned zone by rotating the original rectangular work zone; accepts true/false",
    )
    parser.add_argument("--road-zone-max-length-scale", type=float, default=2.2, help="Maximum road-aligned zone length scale")
    parser.add_argument("--road-zone-max-width-scale", type=float, default=1.3, help="Maximum road-aligned zone width scale")
    parser.add_argument("--road-zone-forward-ratio", type=float, default=0.65, help="Forward length ratio for base-rect road zone")
    parser.add_argument("--road-zone-backward-ratio", type=float, default=0.35, help="Backward length ratio for base-rect road zone")
    parser.add_argument(
        "--enable-fence-clip",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Clip road-aligned zone by nearby fence hull; accepts true/false",
    )
    parser.add_argument("--fence-clip-padding", type=float, default=80.0, help="Padding in pixels for road-aligned fence clip mask")
    parser.add_argument(
        "--enable-fence-clipped-zone",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Cut road-aligned zone by fence barriers and keep component containing forklift ground point; accepts true/false",
    )
    parser.add_argument("--fence-barrier-thickness", type=float, default=20.0, help="Fence barrier line thickness in pixels")
    parser.add_argument(
        "--draw-fence-barrier",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Draw fence barrier lines used to cut road-aligned zone; accepts true/false",
    )
    parser.add_argument(
        "--road-zone-min-retain-ratio",
        type=float,
        default=0.2,
        help="Minimum retained area ratio when clipping road-aligned zone by road mask",
    )
    parser.add_argument(
        "--max-aligned-area-ratio",
        type=float,
        default=1.5,
        help="Maximum road-aligned zone area ratio relative to original rectangular work zone",
    )
    return parser.parse_args(argv)


def analyze_forklift_work_frame(
    model,
    image: np.ndarray,
    args: argparse.Namespace,
    frame_name: str = "frame",
    tracker: Optional[str] = None,
    detection_result: Any = None,
    draw_output: bool = True,
    detections: Optional[List[Dict[str, Any]]] = None,
    helmet_smoother: Any = None,
    frame_id: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run forklift safety rules, optionally reusing a detection result produced by a shared tracker."""
    if image is None:
        raise ValueError("image frame is required")
    image = image.copy()
    if args.nearby_mode not in {"original", "circle", "both"}:
        raise ValueError("--nearby-mode must be one of original, circle, both")
    if args.camera_mode not in {"oblique", "overhead"}:
        raise ValueError("--camera-mode must be one of oblique, overhead")

    if detections is not None or detection_result is not None:
        pass
    elif tracker:
        track_kwargs = {
            "source": image,
            "imgsz": INFERENCE_IMAGE_SIZE,
            "tracker": tracker,
            "persist": True,
            "conf": args.conf,
            "iou": NMS_IOU,
            "end2end": False,
            "agnostic_nms": False,
            "verbose": False,
        }
        if args.device:
            track_kwargs["device"] = args.device
        tracked = model.track(**track_kwargs)
        detection_result = tracked[0] if tracked else None
    else:
        detection_result = run_detection(model, image, conf=args.conf, device=args.device)
    if detections is None:
        detections = parse_detections(detection_result)
    helmet_results = evaluate_helmet_status(detections, image.shape, args.no_helmet_overlap)
    if helmet_smoother is not None:
        helmet_smoother.update(helmet_results, frame_id)
    forklift_boxes = [det["box"] for det in filter_detections(detections, "forklift")]
    fence_boxes = [det["box"] for det in filter_detections(detections, "fence")]
    ground_point_work_area = (
        generate_ground_point_work_area(
            image.shape,
            forklift_boxes,
            fence_boxes,
            padding_px=args.ground_point_work_area_padding,
            fallback_zone_radius=args.fallback_zone_radius,
        )
        if args.enable_ground_point_work_area
        else None
    )
    ground_point_work_area_json = ground_point_work_area_to_json(
        ground_point_work_area,
        args.enable_ground_point_work_area,
        args.ground_point_work_area_padding,
        args.fallback_zone_radius,
    )
    road_surface = (
        generate_road_surface_mask(
            image,
            forklift_boxes,
            fence_boxes,
            method="auto",
            min_area_ratio=args.road_min_area_ratio,
            road_sat_max=args.road_sat_max,
            road_value_min=args.road_value_min,
            road_value_max=args.road_value_max,
            road_morph_kernel=args.road_morph_kernel,
        )
        if args.enable_road_mask
        else {
            "mask": None,
            "metadata": {
                "enabled": False,
                "success": False,
                "method": "auto_hsv_connected_component",
                "road_area_ratio": 0.0,
                "used_connected_component": False,
                "contains_forklift_ground_point": False,
                "warning": None,
            },
        }
    )
    road_surface_mask = road_surface["mask"]
    road_surface_mask_json = road_surface["metadata"]
    road_constrained_work_area = (
        build_road_constrained_work_area(
            ground_point_work_area,
            road_surface_mask,
            bool(args.enable_road_mask and road_surface_mask_json.get("success")),
        )
        if args.enable_road_mask
        else {
            "mask": None,
            "enabled": False,
            "type": "ground_point_work_area_intersect_road_mask",
            "success": False,
            "person_count": 0,
            "polygons": [],
        }
    )
    road_aligned_work_zones = build_road_aligned_work_zones(
        image,
        forklift_boxes,
        fence_boxes,
        enable=args.enable_road_aligned_zone,
        left_scale=args.left_scale,
        right_scale=args.right_scale,
        front_scale=args.front_scale,
        back_scale=args.back_scale,
        height_depth_ratio=args.height_depth_ratio,
        width_depth_ratio=args.width_depth_ratio,
        fence_iou_thres=args.fence_iou_thres,
        road_sat_max=args.road_sat_max,
        road_value_min=args.road_value_min,
        road_value_max=args.road_value_max,
        road_morph_kernel=args.road_morph_kernel,
        road_direction_roi_scale=args.road_direction_roi_scale,
        road_direction_min_pixels=args.road_direction_min_pixels,
        road_zone_length_scale=args.road_zone_length_scale,
        road_zone_width_scale=args.road_zone_width_scale,
        road_zone_back_scale=args.road_zone_back_scale,
        road_zone_intersect_road_mask=args.road_zone_intersect_road_mask,
        road_zone_from_base_rect=args.road_zone_from_base_rect,
        road_zone_max_length_scale=args.road_zone_max_length_scale,
        road_zone_max_width_scale=args.road_zone_max_width_scale,
        road_zone_forward_ratio=args.road_zone_forward_ratio,
        road_zone_backward_ratio=args.road_zone_backward_ratio,
        enable_fence_clip=args.enable_fence_clip,
        fence_clip_padding=args.fence_clip_padding,
        road_zone_min_retain_ratio=args.road_zone_min_retain_ratio,
        max_aligned_area_ratio=args.max_aligned_area_ratio,
        enable_fence_clipped_zone=args.enable_fence_clipped_zone,
        fence_barrier_thickness=args.fence_barrier_thickness,
    )
    forklift_results = evaluate_forklifts(
        detections,
        image.shape,
        args.left_scale,
        args.right_scale,
        args.front_scale,
        args.back_scale,
        args.fence_iou_thres,
        args.height_depth_ratio,
        args.width_depth_ratio,
        args.driver_iou_thres,
        args.driver_inside_ratio_thres,
        args.driver_dx_norm_thres,
        args.driver_min_dy_up_norm,
        args.driver_max_dy_up_norm,
        args.person_distance_width_scale,
        args.person_distance_height_scale,
        args.min_person_distance_px,
        args.enable_fence_zone,
        args.fence_zone_padding,
        args.nearby_mode,
        args.forklift_radius_scale,
        args.min_forklift_radius_px,
        args.max_forklift_radius_px,
        args.camera_mode,
        ground_point_work_area,
        road_constrained_work_area,
        road_aligned_work_zones,
    )
    enrich_forklift_person_helmet_status(forklift_results, helmet_results)
    road_constrained_person_count = sum(
        1
        for forklift in forklift_results
        for person in forklift["nearby_persons"]
        if person.get("in_road_constrained_work_area")
    )
    road_constrained_work_area["person_count"] = road_constrained_person_count
    road_constrained_work_area_json = {
        "enabled": bool(road_constrained_work_area["enabled"]),
        "type": road_constrained_work_area["type"],
        "success": bool(road_constrained_work_area["success"]),
        "person_count": int(road_constrained_work_area["person_count"]),
        "polygons": road_constrained_work_area.get("polygons", []),
    }
    compliance_summary = build_compliance_summary(forklift_results, helmet_results)
    result_json = build_output(
        frame_name,
        forklift_results,
        helmet_results,
        args.nearby_mode,
        len(filter_detections(detections, "fence")),
        ground_point_work_area_json,
        road_surface_mask_json,
        road_constrained_work_area_json,
    )
    if draw_output:
        image = draw_raw_detection_boxes(image, detections)
        for forklift in forklift_results:
            if forklift.get("violations"):
                draw_box(image, forklift.get("forklift_box"), "UNSAFE: FORKLIFT WORK", (0, 0, 255), 5)
            for person in forklift.get("nearby_persons", []) or []:
                draw_box(image, person.get("box"), "UNSAFE: PERSON NEAR FORKLIFT", (0, 0, 255), 5)
        for person in helmet_results:
            if person.get("status") == "no_helmet":
                draw_box(image, person.get("box"), "UNSAFE: NO HELMET", (0, 0, 255), 5)
    return image, result_json


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")
    model = load_model(args.weights)
    annotated, result_json = analyze_forklift_work_frame(model, image, args, Path(args.image).name)
    result_json.update(frame_id=1, source_type="single_image")
    result_json["has_hazard"] = result_json.get("result") == "unsafe"
    if result_json["has_hazard"]:
        violation_messages = {
            "PERSON_NEAR_FORKLIFT": "叉车作业半径内有人",
            "NO_FENCE_NEAR_FORKLIFT": "叉车作业区域未设置有效围蔽",
        }
        draw_unsafe_banner(annotated, [
            violation_messages.get(code, str(code))
            for code in result_json.get("violations", [])
        ] + (["人员未佩戴安全帽"] if any(person.get("status") == "no_helmet" for person in result_json.get("persons", [])) else []))
        evidence = save_hazard_image(annotated, args.output, "forklift_work", 1)
        violations = list(result_json.get("violations", []) or [])
        no_helmet = [person for person in result_json.get("persons", []) or [] if person.get("status") == "no_helmet"]
        hazards = [
            {
                "code": code,
                "check": "forklift_work",
                "message": {
                    "PERSON_NEAR_FORKLIFT": "叉车作业半径内有人",
                    "NO_FENCE_NEAR_FORKLIFT": "叉车作业区域未设置有效围蔽",
                }.get(code, code),
                "count": 1,
            }
            for code in violations
        ]
        if no_helmet:
            hazards.append({
                "code": "NO_HELMET",
                "check": "person_helmet",
                "message": "人员未佩戴安全帽",
                "count": len(no_helmet),
                "details": no_helmet,
            })
        if not hazards:
            hazards.append({
                "code": "FORKLIFT_WORK_UNSAFE",
                "check": "forklift_work",
                "message": "叉车作业存在安全隐患",
                "count": 1,
            })
        event = {
            "algorithm": "forklift_work",
            "source_type": "single_image",
            "frame_id": 1,
            "is_first_hazard": True,
            "hazard_timestamp": evidence["hazard_timestamp"],
            "hazard_image_path": evidence["hazard_image_path"],
            "hazard_count": sum(int(item["count"]) for item in hazards),
            "hazards": hazards,
        }
        result_json.update(
            is_first_hazard=True,
            hazard_timestamp=event["hazard_timestamp"],
            hazard_image_path=event["hazard_image_path"],
            hazard_count=event["hazard_count"],
            hazards=hazards,
            hazard_event=event,
            hazard_events=[event],
            hazard_event_count=1,
        )
    else:
        result_json.update(hazard_events=[], hazard_event_count=0)
    result_json = add_finished_at(result_json)
    json_path = args.json or str(Path(args.output).with_suffix(".json"))
    save_json(result_json, json_path)


if __name__ == "__main__":
    main()
