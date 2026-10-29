from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_TOOLS = PROJECT_ROOT / "safety_tools"
if str(SAFETY_TOOLS) not in sys.path:
    sys.path.insert(0, str(SAFETY_TOOLS))

from crane_safety_check import (  # noqa: E402
    CraneDangerTemporalFilter,
    analyze_crane_obb_result,
    build_boom_geometry,
    evaluate_crane_pair,
    parse_crane_obb_detections,
    validate_crane_model,
)
from safety_video_monitor import build_hazard_instances, build_selected_frame_result  # noqa: E402


BOOM_POINTS = [[90, 180], [290, 80], [300, 100], [100, 200]]


class FakeOBB:
    def __init__(self, points, boxes, confidences, classes, track_ids=None):
        self.xyxyxyxy = np.asarray(points, dtype=float)
        self.xyxy = np.asarray(boxes, dtype=float)
        self.conf = np.asarray(confidences, dtype=float)
        self.cls = np.asarray(classes, dtype=float)
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=float)

    def cpu(self):
        return self

    def __len__(self):
        return len(self.cls)


def fake_result(person_box=(160, 180, 190, 260), boom_conf=0.9, person_conf=0.9):
    x1, y1, x2, y2 = person_box
    person_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return SimpleNamespace(
        names={0: "boom", 1: "person"},
        obb=FakeOBB(
            [BOOM_POINTS, person_points],
            [[90, 80, 300, 200], list(person_box)],
            [boom_conf, person_conf],
            [0, 1],
            [3, 8],
        ),
    )


def test_model_validation_requires_obb_boom_and_person() -> None:
    validate_crane_model(SimpleNamespace(task="obb", names={0: "boom", 1: "person"}))
    with pytest.raises(ValueError, match="OBB"):
        validate_crane_model(SimpleNamespace(task="detect", names={0: "boom", 1: "person"}))
    with pytest.raises(ValueError, match="person"):
        validate_crane_model(SimpleNamespace(task="obb", names={0: "boom"}))


def test_parser_uses_model_names_and_separate_thresholds() -> None:
    result = fake_result(boom_conf=0.85, person_conf=0.35)
    booms, persons = parse_crane_obb_detections(result, boom_conf=0.8, person_conf=0.4)
    assert len(booms) == 1
    assert persons == []
    assert booms[0]["track_id"] == 3
    assert np.allclose(booms[0]["points"], BOOM_POINTS)


def test_boom_geometry_is_independent_of_input_point_order() -> None:
    expected = build_boom_geometry(BOOM_POINTS)
    for points in itertools.permutations(BOOM_POINTS):
        geometry = build_boom_geometry(points)
        assert geometry["lowest_y"] == pytest.approx(expected["lowest_y"])
        assert geometry["long_edge_length"] == pytest.approx(expected["long_edge_length"])
        assert geometry["short_edge_length"] == pytest.approx(expected["short_edge_length"])


def test_vertical_boom_is_reported_as_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="vertical"):
        build_boom_geometry([[100, 50], [120, 50], [120, 300], [100, 300]])


@pytest.mark.parametrize(
    "person_box, expected",
    [
        ([160, 180, 190, 260], (True, True, True, True)),
        ([390, 180, 420, 260], (True, False, True, False)),
        ([160, 180, 190, 195], (True, True, False, False)),
        ([160, 140, 190, 260], (False, True, True, False)),
    ],
)
def test_pair_rule_conditions(person_box, expected) -> None:
    boom = {
        "target_id": 3,
        "track_id": 3,
        "box": [90, 80, 300, 200],
        "confidence": 0.9,
        "geometry": build_boom_geometry(BOOM_POINTS),
    }
    person = {
        "target_id": 8,
        "track_id": 8,
        "box": person_box,
        "confidence": 0.9,
    }
    pair = evaluate_crane_pair(boom, person, 400, 5.0, 5.0, 0.05)
    assert (
        pair["fully_below"],
        pair["in_boom_range"],
        pair["foot_below_lowest"],
        pair["raw_danger"],
    ) == expected


def test_temporal_filter_enters_on_three_of_five_and_exits_after_five_safe_frames() -> None:
    temporal = CraneDangerTemporalFilter(window=5, enter_hits=3, exit_safe_frames=5, ttl_frames=30)

    def update(raw_danger: bool, frame_id: int) -> bool:
        pair = {"boom_id": 3, "person_id": 8, "raw_danger": raw_danger}
        temporal.update([pair], frame_id)
        return bool(pair["confirmed_danger"])

    assert [update(value, index) for index, value in enumerate([True, False, True, False, True], start=1)] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [update(False, index) for index in range(6, 11)] == [True, True, True, True, False]


def test_crane_result_integrates_with_unified_result_and_hazard_reporting() -> None:
    crane_result = analyze_crane_obb_result(
        fake_result(),
        (400, 500, 3),
        frame_id=1,
        temporal_applied=False,
    )
    frame_result = build_selected_frame_result(
        {"crane_under_boom"},
        1,
        [],
        None,
        None,
        crane_result,
    )
    assert frame_result["result"] == "unsafe"
    assert frame_result["results"]["crane_under_boom"]["danger_person_count"] == 1
    hazards = build_hazard_instances("safety_combined", frame_result)
    assert len(hazards) == 1
    assert hazards[0]["code"] == "PERSON_UNDER_BOOM"
    assert hazards[0]["_incident_key"] == ("PERSON_UNDER_BOOM", "boom", "3", "person", "8")
