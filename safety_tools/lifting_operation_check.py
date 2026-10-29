"""Simple lifting compliance: a crane needs an officer and fence below it."""

from __future__ import annotations

from typing import Any

from common import is_valid_box
from model_config import normalize_class_name


LIFTING_CHECK = "lifting_operation"
LIFTING_CLASS_NAMES = {"crane", "safety_officer", "fence"}
LIFTING_VIOLATIONS = {
    "NO_SAFETY_OFFICER_BELOW_CRANE": "吊装作业下方缺少安全监护人员",
    "NO_FENCE_BELOW_CRANE": "吊装作业下方缺少围栏",
}


def analyze_lifting_operation(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate each crane using box-center Y only; no enclosure or distance rule.

    Inputs have already passed the shared confidence threshold. An officer or
    fence counts when its box center is below the crane's box center, anywhere
    in the same frame. No crane means unknown, rather than compliant.
    """
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in LIFTING_CLASS_NAMES}
    for detection in detections:
        name = normalize_class_name(detection.get("class_name", ""))
        if name in groups and is_valid_box(detection.get("box")):
            groups[name].append(detection)

    cranes = []
    for index, crane in enumerate(groups["crane"]):
        center_y = (crane["box"][1] + crane["box"][3]) * 0.5
        officers = [item for item in groups["safety_officer"]
                    if (item["box"][1] + item["box"][3]) * 0.5 > center_y]
        fences = [item for item in groups["fence"]
                  if (item["box"][1] + item["box"][3]) * 0.5 > center_y]
        violations = []
        if not officers:
            violations.append("NO_SAFETY_OFFICER_BELOW_CRANE")
        if not fences:
            violations.append("NO_FENCE_BELOW_CRANE")
        track_id = crane.get("track_id")
        cranes.append({
            "crane_id": track_id if track_id is not None else f"crane_{index}",
            "track_id": track_id,
            "box": crane["box"],
            "confidence": crane["confidence"],
            "center_y": center_y,
            "has_safety_officer": bool(officers),
            "has_fence": bool(fences),
            "safety_officers": officers,
            "fences": fences,
            "result": "unsafe" if violations else "safe",
            "violations": violations,
        })

    unsafe_count = sum(item["result"] == "unsafe" for item in cranes)
    return {
        "result": "unsafe" if unsafe_count else ("safe" if cranes else "unknown"),
        "crane_count": len(cranes),
        "safety_officer_count": len(groups["safety_officer"]),
        "fence_count": len(groups["fence"]),
        "compliant_crane_count": len(cranes) - unsafe_count,
        "unsafe_crane_count": unsafe_count,
        "missing_safety_officer_count": sum(not item["has_safety_officer"] for item in cranes),
        "missing_fence_count": sum(not item["has_fence"] for item in cranes),
        "cranes": cranes,
    }
