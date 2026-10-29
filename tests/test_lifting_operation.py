from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "safety_tools"))

import safety_video_monitor as monitor
from common import HazardIncidentTracker, validate_model_classes
from lifting_operation_check import LIFTING_CHECK, analyze_lifting_operation
from safety_web_server import SAFETY_CHECK_LABELS, build_unified_response


def detection(name, box, track_id=1, confidence=0.9):
    return {"class_name": name, "class_id": 0, "box": box, "track_id": track_id, "confidence": confidence}


CRANE = detection("crane", [530, 14, 735, 255], 16)
OFFICER = detection("safety officer", [198, 263, 254, 345], 15)
FENCE = detection("fence", [9, 399, 142, 464], 4)


@pytest.mark.parametrize(
    "extras, expected, missing_officer, missing_fence",
    [([OFFICER, FENCE], "safe", 0, 0), ([OFFICER], "unsafe", 0, 1),
     ([FENCE], "unsafe", 1, 0), ([], "unsafe", 1, 1)],
)
def test_compliance_truth_table(extras, expected, missing_officer, missing_fence):
    result = analyze_lifting_operation([CRANE, *extras])
    assert result["result"] == expected
    assert result["missing_safety_officer_count"] == missing_officer
    assert result["missing_fence_count"] == missing_fence
    assert result["unsafe_crane_count"] == int(expected == "unsafe")
    assert len(result["cranes"][0]["violations"]) == missing_officer + missing_fence


def test_without_crane_is_unknown_even_with_both_protections():
    for detections in ([], [OFFICER, FENCE]):
        result = analyze_lifting_operation(detections)
        assert result["result"] == "unknown"
        assert result["cranes"] == []
        assert result["missing_safety_officer_count"] == result["missing_fence_count"] == 0


@pytest.mark.parametrize("name", ["safety officer", "safety_officer", " Safety_Officer "])
def test_checkpoint_alias_and_reordered_class_ids(name):
    model = SimpleNamespace(names={8: "crane", 2: name, 11: "fence"})
    assert set(validate_model_classes(model, monitor.CHECK_CLASS_REQUIREMENTS[LIFTING_CHECK])) == {8, 2, 11}
    assert analyze_lifting_operation([CRANE, {**OFFICER, "class_name": name}, FENCE])["result"] == "safe"


def test_ordinary_person_and_protections_above_crane_do_not_satisfy_rule():
    result = analyze_lifting_operation([
        CRANE, {**OFFICER, "class_name": "person"},
        detection("safety_officer", [100, 5, 130, 100]),
        detection("fence", [50, 20, 100, 80]),
    ])
    assert result["missing_safety_officer_count"] == result["missing_fence_count"] == 1


def test_each_crane_is_evaluated_separately_by_center_y():
    result = analyze_lifting_operation([
        CRANE, detection("crane", [300, 300, 400, 460], 17), OFFICER, FENCE,
    ])
    assert result["result"] == "unsafe"
    assert result["compliant_crane_count"] == result["unsafe_crane_count"] == 1
    assert result["cranes"][1]["violations"] == ["NO_SAFETY_OFFICER_BELOW_CRANE"]


def test_invalid_boxes_are_ignored():
    result = analyze_lifting_operation([
        CRANE, {**OFFICER, "box": [1, 1, 1, 10]}, {**FENCE, "box": [1, 2, 3, float("nan")]},
    ])
    assert result["result"] == "unsafe"
    assert result["safety_officer_count"] == result["fence_count"] == 0


def test_reporting_counts_each_missing_requirement_and_deduplicates_tracks(tmp_path):
    lifting = analyze_lifting_operation([CRANE])
    frame = monitor.build_selected_frame_result({LIFTING_CHECK}, 1, [], None, None, lifting_result=lifting)
    assert frame["result"] == "unsafe"
    assert set(frame["results"]) == {LIFTING_CHECK}
    hazards = monitor.build_hazard_instances("safety_combined", frame)
    assert {item["code"] for item in hazards} == {"NO_SAFETY_OFFICER_BELOW_CRANE", "NO_FENCE_BELOW_CRANE"}
    tracker = HazardIncidentTracker()
    incidents = tracker.update(hazards)
    assert len(incidents) == 2
    assert tracker.update(hazards) == []
    assert monitor.unsafe_target_boxes(frame)[0][0] == CRANE["box"]
    annotated = monitor.draw_selected_frame(np.zeros((480, 827, 3), dtype=np.uint8), {LIFTING_CHECK}, frame, [])
    assert annotated.any()
    assert monitor.attach_hazard_event(frame, annotated, tmp_path / "result.jpg", "safety_combined", incidents, True, 1)
    assert Path(frame["hazard_image_path"]).is_file()
    response = build_unified_response(frame, 2, tmp_path / "result.jpg", tmp_path / "result.json", [])
    assert response["hazards_by_check"][LIFTING_CHECK] == response["detected_hazards"] == 2


def test_missing_checkpoint_class_fails_before_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "load_model", lambda _: SimpleNamespace(names={0: "crane", 1: "fence"}))
    with pytest.raises(ValueError, match="safety_officer"):
        monitor.run_unified_safety_monitor({LIFTING_CHECK}, "single_image", tmp_path / "x.jpg", "all_best.pt", None)


def test_lifting_uses_shared_threshold_and_one_inference_per_frame(monkeypatch, tmp_path):
    # A low-confidence officer retained for the boom check must not satisfy lifting.
    model = SimpleNamespace(names={0: "boom", 1: "person", 2: "crane", 3: "safety officer", 4: "fence"}, task="obb")
    monkeypatch.setattr(monitor, "load_model", lambda _: model)
    monkeypatch.setattr(monitor.cv2, "imread", lambda _: np.zeros((480, 827, 3), dtype=np.uint8))
    calls = []
    monkeypatch.setattr(monitor, "_track_frame", lambda *args: calls.append(args) or None)
    monkeypatch.setattr(monitor, "parse_detections", lambda _: [CRANE, {**OFFICER, "confidence": 0.45}, FENCE])
    result = monitor.run_unified_safety_monitor(
        {LIFTING_CHECK, "crane_under_boom"}, "single_image", tmp_path / "x.jpg", "all_best.pt",
        tmp_path / "result.jpg", conf=0.55,
    )
    assert len(calls) == 1 and calls[0][3] == 0.4
    assert result["results"][LIFTING_CHECK]["cranes"][0]["violations"] == ["NO_SAFETY_OFFICER_BELOW_CRANE"]


def test_web_page_and_server_accept_lifting_check():
    html = (Path(__file__).resolve().parents[1] / "safety_tools/web/index.html").read_text(encoding="utf-8")
    assert f'value="{LIFTING_CHECK}"' in html
    assert LIFTING_CHECK in SAFETY_CHECK_LABELS
