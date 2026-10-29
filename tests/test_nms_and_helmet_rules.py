from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SAFETY_TOOLS = Path(__file__).resolve().parents[1] / "safety_tools"
if str(SAFETY_TOOLS) not in sys.path:
    sys.path.insert(0, str(SAFETY_TOOLS))

from common import CLASS_IDS, NMS_IOU, load_model, yolo_detection_logging  # noqa: E402
from helmet_check import build_output, evaluate_persons  # noqa: E402
from forklift_work_check import evaluate_helmet_status  # noqa: E402
from safety_video_monitor import (  # noqa: E402
    HelmetTrackSmoother, build_hazard_instances, build_selected_frame_result,
)
from ultralytics.engine.results import Results  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402


def detection(name, box, track_id=None, confidence=0.9):
    return {"class_id": CLASS_IDS[name], "class_name": name, "box": box,
            "track_id": track_id, "confidence": confidence}


def model_result(rows):
    return Results(orig_img=np.zeros((240, 160, 3), dtype=np.uint8), path="frame.jpg",
                   names={0: "person", 1: "helmet"}, boxes=torch.tensor(rows).reshape(-1, 6))


def test_framework_obb_nms_keeps_strongest_and_preserves_other_classes() -> None:
    prediction = torch.zeros(1, 20, 3)  # xywh + 15 classes + angle
    prediction[0, :4, :] = torch.tensor([[60, 60, 60], [80, 80, 80], [80, 80, 80], [40, 40, 40]])
    prediction[0, 4, :2] = torch.tensor([0.65, 0.95])
    prediction[0, 5, 2] = 0.85
    prediction[0, -1, :] = 0.4
    result = non_max_suppression(prediction, 0.4, NMS_IOU, nc=15, rotated=True, end2end=False)[0]
    assert result[:, 5].tolist() == [0, 1]
    assert result[:, 4].tolist() == pytest.approx([0.95, 0.85])


def test_model_loader_selects_native_nms_before_predictor_setup(monkeypatch, tmp_path) -> None:
    import common
    weights = tmp_path / "model.pt"
    weights.touch()
    model = SimpleNamespace(overrides={}, callbacks={"on_predict_postprocess_end": []})
    monkeypatch.setattr(common, "YOLO", lambda _path: model)
    assert load_model(weights) is model
    assert model.overrides == {"end2end": False, "agnostic_nms": False, "iou": 0.7, "imgsz": 640}
    assert model.callbacks["on_predict_postprocess_end"] == []


def test_logging_runs_after_native_nms_and_before_tracker() -> None:
    tracked_counts = []
    tracker = lambda predictor: tracked_counts.append(len(predictor.results[0].boxes))
    model = SimpleNamespace(callbacks={"on_predict_postprocess_end": [tracker]})
    logs = []
    with yolo_detection_logging(model, 42, lambda frame, message: logs.append(message)):
        predictor = SimpleNamespace(args=SimpleNamespace(iou=0.4), results=[model_result([
            [10, 10, 110, 210, 0.95, 0], [12, 12, 112, 212, 0.85, 0],
        ])])
        for callback in model.callbacks["on_predict_postprocess_end"]:
            callback(predictor)
    assert tracked_counts == [2]
    assert len(logs) == 1 and "count=2" in logs[0]
    assert model.callbacks["on_predict_postprocess_end"] == [tracker]


@pytest.mark.parametrize("evaluate", [evaluate_persons, evaluate_helmet_status])
def test_head_without_person_is_one_unsafe_person(evaluate) -> None:
    persons = evaluate([detection("head_no_helmet", [20, 10, 50, 40], 7)], (240, 160, 3), 0.2)
    result = build_output("frame.jpg", persons)
    assert result["result"] == "unsafe" and result["has_hazard"]
    assert result["total_persons"] == result["no_helmet_count"] == 1
    assert persons[0]["box"] == [20, 10, 50, 40]
    assert persons[0]["track_id"] == 7
    assert persons[0]["detection_source"] == "head_no_helmet"
    assert len(build_hazard_instances("helmet", result)) == 1


@pytest.mark.parametrize("evaluate", [evaluate_persons, evaluate_helmet_status])
def test_associated_head_and_person_are_counted_once(evaluate) -> None:
    persons = evaluate([
        detection("person", [10, 10, 110, 210], 1),
        detection("head_no_helmet", [25, 15, 55, 45], 2),
        detection("head_no_helmet", [60, 15, 80, 40], 3),
    ], (240, 160, 3), 0.2)
    assert len(persons) == 1
    assert persons[0]["status"] == "no_helmet"
    assert persons[0]["track_id"] == 1


def test_unassociated_head_is_added_alongside_helmeted_person() -> None:
    persons = evaluate_persons([
        detection("person", [10, 10, 100, 210], 1),
        detection("helmet", [25, 15, 55, 45], 2),
        detection("head_no_helmet", [120, 15, 145, 45], 3),
    ], (240, 160, 3))
    assert [person["status"] for person in persons] == ["helmet", "no_helmet"]
    unified = build_selected_frame_result({"person_helmet"}, 1, persons, None, None)
    assert unified["results"]["person_helmet"]["no_helmet_count"] == 1
    assert len(build_hazard_instances("safety_combined", unified)) == 1


def test_person_with_helmet_remains_safe() -> None:
    persons = evaluate_persons([
        detection("person", [10, 10, 110, 210]), detection("helmet", [25, 15, 55, 45]),
    ], (240, 160, 3))
    assert build_output("frame.jpg", persons)["result"] == "safe"


def test_small_person_without_helmet_is_unsafe() -> None:
    persons = evaluate_persons([detection("person", [1, 1, 2, 3])], (240, 160, 3))
    assert persons[0]["status"] == "no_helmet"


@pytest.mark.parametrize("source", ["person", "head_no_helmet"])
def test_no_helmet_requires_five_consecutive_frames_after_helmet(source) -> None:
    smoother = HelmetTrackSmoother()
    for frame_id in range(1, 5):
        smoother.update([{"track_id": 1, "status": "helmet"}], frame_id)
    for frame_id in range(5, 11):
        persons = evaluate_persons([detection(source, [10, 10, 110, 210], 1)], (240, 160, 3))
        smoother.update(persons, frame_id)
        assert persons[0]["raw_status"] == "no_helmet"
        assert persons[0]["status"] == ("no_helmet" if frame_id >= 9 else "unknown")
        assert build_output("frame.jpg", persons)["has_hazard"] is (frame_id >= 9)


@pytest.mark.parametrize("interruption", ["helmet", "unknown", "missing"])
@pytest.mark.parametrize("prior_frames", [4, 5])
def test_helmet_unknown_or_missed_person_restarts_confirmation(interruption, prior_frames) -> None:
    smoother = HelmetTrackSmoother()
    for frame_id in range(1, prior_frames + 1):
        smoother.update([{"track_id": 1, "status": "no_helmet"}], frame_id)
    persons = [] if interruption == "missing" else [{"track_id": 1, "status": interruption}]
    smoother.update(persons, prior_frames + 1)
    if persons:
        assert persons[0]["status"] == interruption
    for offset in range(1, 6):
        persons = [{"track_id": 1, "status": "no_helmet"}]
        smoother.update(persons, prior_frames + 1 + offset)
        assert persons[0]["status"] == ("no_helmet" if offset == 5 else "unknown")


def test_each_track_needs_its_own_five_frames() -> None:
    smoother = HelmetTrackSmoother()
    for frame_id in range(1, 7):
        persons = [{"track_id": 1, "status": "no_helmet"}]
        if frame_id >= 3:
            persons.append({"track_id": 2, "status": "no_helmet"})
        smoother.update(persons, frame_id)
        assert persons[0]["status"] == ("no_helmet" if frame_id >= 5 else "unknown")
        if len(persons) == 2:
            assert persons[1]["status"] == "unknown"
    persons = [{"track_id": 2, "status": "no_helmet"}]
    smoother.update(persons, 7)
    assert persons[0]["status"] == "no_helmet"


def test_confirmation_counts_processed_frames_and_does_not_repeat_same_frame() -> None:
    smoother = HelmetTrackSmoother()
    for index, frame_id in enumerate([1, 16, 31, 46, 61], start=1):
        for _ in range(5):
            persons = [{"track_id": 1, "status": "no_helmet"}]
            smoother.update(persons, frame_id)
            assert persons[0]["status"] == ("no_helmet" if index == 5 else "unknown")


def test_untracked_person_cannot_confirm_a_five_frame_violation() -> None:
    smoother = HelmetTrackSmoother()
    for frame_id in range(1, 7):
        persons = [{"track_id": None, "person_id": 1, "status": "no_helmet"}]
        smoother.update(persons, frame_id)
        assert persons[0]["raw_status"] == "no_helmet"
        assert persons[0]["status"] == "unknown"


@pytest.mark.parametrize("track_id", [None, 1])
def test_single_image_window_confirms_immediately(track_id) -> None:
    persons = [{"track_id": track_id, "status": "no_helmet"}]
    HelmetTrackSmoother(window=1).update(persons, 1)
    assert persons[0]["raw_status"] == persons[0]["status"] == "no_helmet"


@pytest.mark.parametrize("entrypoint, source_type", [
    ("person_helmet", "image_sequence"), ("driver_helmet", "image_sequence"),
    ("helmet", "image_sequence"), ("forklift_work", "image_sequence"),
    ("person_helmet", "video"), ("driver_helmet", "video"),
])
def test_monitor_delays_helmet_alarm_and_evidence_until_fifth_frame(
    monkeypatch, tmp_path, entrypoint, source_type,
) -> None:
    import safety_video_monitor as monitor

    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    output = Results(orig_img=frame, path="frame.jpg", names={value: key for key, value in CLASS_IDS.items()},
                     boxes=torch.tensor([[10, 10, 110, 210, 7, 0.9, CLASS_IDS["person"]]]))
    model = SimpleNamespace(names=output.names, track=lambda **kwargs: [output])
    monkeypatch.setattr(monitor, "load_model", lambda _path: model)
    monkeypatch.setattr(monitor, "_track_frame", lambda *args: output)
    monkeypatch.setattr(monitor, "get_image_files", lambda path: [path / f"{i}.jpg" for i in range(6)])
    monkeypatch.setattr(monitor.cv2, "imread", lambda _path: frame.copy())
    source_path = tmp_path
    if source_type == "video":
        class Capture:
            def __init__(self):
                self.frame_id = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 25 if prop == monitor.cv2.CAP_PROP_FPS else 76

            def read(self):
                self.frame_id += 1
                return (True, frame.copy()) if self.frame_id <= 76 else (False, None)

            def release(self):
                pass

        source_path = tmp_path / "input.mp4"
        source_path.touch()
        monkeypatch.setattr(monitor.cv2, "VideoCapture", lambda _path: Capture())
    if entrypoint == "driver_helmet":
        monkeypatch.setattr(monitor, "analyze_forklift_work_frame", lambda *args, **kwargs: (frame.copy(), {
            "forklifts": [{"forklift_id": 20, "excluded_persons": [{
                "track_id": 7, "box": [10, 10, 110, 210], "helmet_status": "no_helmet",
            }]}],
        }))

    observed = []

    def preview(_frame_id, _frame, status):
        observed.append(status)
        assert len(list(tmp_path.glob("result_*hazard_*.jpg"))) == (1 if len(observed) >= 5 else 0)

    options = dict(source_type=source_type, source_path=source_path, weights_path="unused.pt",
                   output_media=tmp_path / "result.jpg", preview_callback=preview,
                   sample_interval=15 if source_type == "video" else 1)
    if entrypoint in {"person_helmet", "driver_helmet"}:
        result = monitor.run_unified_safety_monitor(enabled_checks={entrypoint}, **options)
        helmet_results = [status["results"][entrypoint] for status in observed]
    else:
        result = monitor.run_safety_monitor(mode=entrypoint, **options)
        helmet_results = observed
    assert [status["result"] for status in helmet_results] == ["unknown"] * 4 + ["unsafe"] * 2
    assert [status["no_helmet_count"] for status in helmet_results] == [0] * 4 + [1] * 2
    assert [status["has_hazard"] for status in observed] == [False] * 4 + [True] * 2
    assert ["hazard_event" in status for status in observed] == [False] * 4 + [True, False]
    assert result["hazard_event_count"] == 1
    assert result["hazard_events"][0]["frame_id"] == (61 if source_type == "video" else 5)
