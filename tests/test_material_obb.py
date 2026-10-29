from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_TOOLS = PROJECT_ROOT / "safety_tools"
if str(SAFETY_TOOLS) not in sys.path:
    sys.path.insert(0, str(SAFETY_TOOLS))

import material_count_cli as mcc  # noqa: E402
import safety_web_server as sws  # noqa: E402
from common import format_yolo_detection_log  # noqa: E402


CLASS_NAMES = {
    1: "4D distribution box",
    3: "200kV Transformer",
}


class FakeOBB:
    def __init__(self, track_ids=None) -> None:
        self.xyxyxyxy = np.asarray(
            [
                [[10, 20], [80, 10], [90, 50], [20, 60]],
                [[110, 120], [180, 110], [190, 150], [120, 160]],
            ],
            dtype=float,
        )
        self.xyxy = np.asarray([[10, 10, 90, 60], [110, 110, 190, 160]], dtype=float)
        self.conf = np.asarray([0.91, 0.87], dtype=float)
        self.cls = np.asarray([1, 3], dtype=float)
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=float)

    def cpu(self):
        return self

    def __len__(self) -> int:
        return len(self.cls)


def fake_result(track_ids=None):
    return SimpleNamespace(obb=FakeOBB(track_ids), boxes=None, names=CLASS_NAMES)


class FakeDetector:
    def __init__(self, result) -> None:
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]

    def track(self, **_kwargs):
        return [self.result]


def make_counter() -> mcc.MaterialCounter:
    counter = mcc.MaterialCounter("item_best.pt", 0.4, mcc.TrackerArgs(), "cpu")
    counter.target_class_ids = [1, 3]
    counter.target_classes = [CLASS_NAMES[1], CLASS_NAMES[3]]
    counter.class_counts = defaultdict(int, {name: 0 for name in counter.target_classes})
    return counter


def test_material_defaults_and_supported_ui_classes_match_new_model() -> None:
    assert sws.DEFAULT_MATERIAL_MODEL == sws.DEFAULT_WEIGHTS
    assert sws.DEFAULT_MATERIAL_MODEL.name == "all_best.pt"
    assert list(sws.MATERIAL_CLASS_CANDIDATES) == [
        "large cable",
        "4D distribution box",
        "6D distribution box",
        "200kV Transformer",
        "400kV Transformer",
        "400kV distribution box",
        "500kV distribution box",
        "630kV Transformer",
        "630kV Transformer NX2",
    ]


def test_parse_material_detections_reads_obb_polygon_and_track_id() -> None:
    detections = mcc.parse_material_detections(fake_result([21, 22]))
    assert len(detections) == 2
    assert detections[0]["track_id"] == 21
    assert detections[0]["class_id"] == 1
    assert detections[0]["points"] == [[10.0, 20.0], [80.0, 10.0], [90.0, 50.0], [20.0, 60.0]]
    assert detections[0]["box"] == [10.0, 10.0, 90.0, 60.0]


def test_single_image_counts_and_draws_selected_obb_detections(monkeypatch) -> None:
    monkeypatch.setattr(mcc, "draw_detection_label", lambda *_args, **_kwargs: None)
    counter = make_counter()
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    painted, detected = counter._process_single_image(frame, FakeDetector(fake_result()), CLASS_NAMES)

    assert detected == 2
    assert counter.total_count == 2
    assert counter.class_counts == {"4D distribution box": 1, "200kV Transformer": 1}
    assert painted.any()


def test_tracked_obb_is_counted_once_per_class_and_track(monkeypatch) -> None:
    monkeypatch.setattr(mcc, "draw_detection_label", lambda *_args, **_kwargs: None)
    counter = make_counter()
    detector = FakeDetector(fake_result([21, 22]))
    frame = np.zeros((200, 240, 3), dtype=np.uint8)

    _, active_first = counter._process_frame(frame.copy(), 1, detector, CLASS_NAMES, Path("tracker.yaml"))
    _, active_second = counter._process_frame(frame.copy(), 2, detector, CLASS_NAMES, Path("tracker.yaml"))

    assert active_first == 2
    assert active_second == 2
    assert counter.total_count == 2
    assert counter.class_counts == {"4D distribution box": 1, "200kV Transformer": 1}


def test_obb_detection_log_contains_polygon_coordinates() -> None:
    message = format_yolo_detection_log(fake_result(), 7)
    assert "[YOLO OBB原始检测] frame=7 count=2" in message
    assert "class=1 4D distribution box conf=0.9100" in message
    assert "obb=[10.00, 20.00; 80.00, 10.00; 90.00, 50.00; 20.00, 60.00]" in message


@pytest.mark.parametrize("iou, embedding_distance, keep_match", [
    (0.65, 0.36, False),  # Appearance previously passed 0.8, but fails 0.85.
    (0.55, 0.16, False),  # Spatial overlap previously passed 0.5, but fails 0.6.
    (0.75, 0.16, True),   # Strong spatial and appearance match remains eligible.
])
def test_material_reid_rejects_weaker_associations(monkeypatch, iou, embedding_distance, keep_match):
    from ultralytics.trackers.bot_sort import BOTSORT, matching

    monkeypatch.setattr(matching, "iou_distance", lambda *_args: np.array([[1.0 - iou]]))
    monkeypatch.setattr(matching, "embedding_distance", lambda *_args: np.array([[embedding_distance]]))
    tracker = BOTSORT.__new__(BOTSORT)
    tracker.args = SimpleNamespace(with_reid=True, fuse_score=False)
    tracker.encoder = object()
    tracker.proximity_thresh, tracker.appearance_thresh = 0.5, 0.8
    old_matches = matching.linear_assignment(tracker.get_dists([object()], [object()]), thresh=0.25)[0]
    assert len(old_matches) == 1

    defaults = mcc.TrackerArgs()
    tracker.proximity_thresh = defaults.proximity_thresh
    tracker.appearance_thresh = defaults.appearance_thresh
    matches = matching.linear_assignment(tracker.get_dists([object()], [object()]), thresh=0.25)[0]
    assert bool(len(matches)) is keep_match


@pytest.mark.parametrize("overrides, expected", [
    ({}, (0.6, 0.85)),
    ({"proximity_thresh": "0.7", "appearance_thresh": "0.9"}, (0.7, 0.9)),
])
def test_material_api_passes_reid_options_to_tracker_yaml(monkeypatch, tmp_path, overrides, expected):
    weights = tmp_path / "model.pt"
    weights.touch()
    image = tmp_path / "input.jpg"
    image.touch()
    values = {"model": str(weights), "source_path": str(image), "output_dir": str(tmp_path),
              "target_classes": "4D distribution box", "device": "cpu", **overrides}
    form = {key: SimpleNamespace(value=value, filename=None) for key, value in values.items()}
    captured = {}
    responses = []
    monkeypatch.setattr(sws, "create_job", lambda *_args: None)
    monkeypatch.setattr(sws, "resolve_device", lambda _value: "cpu")
    monkeypatch.setattr(sws, "start_managed_job",
                        lambda _job_id, _target, args: captured.update(tracker=args[6], command=args[11]))
    handler = SimpleNamespace(parse_form=lambda: form, send_json=responses.append)
    sws.SafetyWebHandler.start_material_count(handler)
    assert responses[0]["ok"] is True
    monkeypatch.setattr(mcc, "WEB_TRACKER_YAML", tmp_path / "tracker.yaml")
    config = mcc.YAML.load(mcc.write_tracker_yaml(mcc.TrackerArgs(**captured["tracker"])))
    assert (config["proximity_thresh"], config["appearance_thresh"]) == expected
    assert config["match_thresh"] == 0.25
    for key, value in zip(("proximity", "appearance"), expected):
        flag = f"--{key}-thresh"
        assert float(captured["command"][captured["command"].index(flag) + 1]) == value


@pytest.mark.parametrize("name, value", [("proximity_thresh", "-0.1"), ("appearance_thresh", "1.1")])
def test_material_api_rejects_out_of_range_reid_before_start(name, value):
    handler = SimpleNamespace(parse_form=lambda: {name: SimpleNamespace(value=value, filename=None)})
    with pytest.raises(ValueError, match="0 到 1"):
        sws.SafetyWebHandler.start_material_count(handler)


class FakeBoxes:
    def __init__(self, track_ids=None) -> None:
        self.xyxy = np.asarray([[10, 10, 90, 60], [110, 110, 190, 160]], dtype=float)
        self.conf = np.asarray([0.91, 0.87], dtype=float)
        self.cls = np.asarray([1, 3], dtype=float)
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=float)

    def cpu(self):
        return self

    def __len__(self) -> int:
        return len(self.cls)


def fake_detect_result(track_ids=None):
    return SimpleNamespace(obb=None, boxes=FakeBoxes(track_ids), names=CLASS_NAMES)


def test_parse_material_detections_reads_plain_detect_boxes() -> None:
    detections = mcc.parse_material_detections(fake_detect_result([5, 6]))
    assert len(detections) == 2
    assert detections[0]["track_id"] == 5
    assert detections[0]["class_id"] == 1
    assert detections[0]["box"] == [10.0, 10.0, 90.0, 60.0]
    assert detections[0]["points"] == [[10.0, 10.0], [90.0, 10.0], [90.0, 60.0], [10.0, 60.0]]


def test_single_image_counts_plain_detect_boxes(monkeypatch) -> None:
    monkeypatch.setattr(mcc, "draw_detection_label", lambda *_args, **_kwargs: None)
    counter = make_counter()
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    painted, detected = counter._process_single_image(frame, FakeDetector(fake_detect_result()), CLASS_NAMES)

    assert detected == 2
    assert counter.total_count == 2
    assert counter.class_counts == {"4D distribution box": 1, "200kV Transformer": 1}
    assert painted.any()


def test_material_run_rejects_unsupported_task(monkeypatch, tmp_path) -> None:
    counter = mcc.MaterialCounter("model.pt", 0.4, mcc.TrackerArgs(), "cpu")
    monkeypatch.setattr(mcc, "load_model", lambda _path: SimpleNamespace(task="segment"))
    with pytest.raises(ValueError, match="OBB 或目标检测"):
        counter.run("single_image", tmp_path / "frame.jpg", None)


def test_material_api_rejects_unsupported_model_task(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "model.pt"
    weights.touch()
    image = tmp_path / "input.jpg"
    image.touch()
    values = {"model": str(weights), "source_path": str(image), "output_dir": str(tmp_path),
              "target_classes": "4D distribution box", "device": "cpu"}
    form = {key: SimpleNamespace(value=value, filename=None) for key, value in values.items()}
    monkeypatch.setattr(sws, "describe_model", lambda _path: {
        "ok": True, "path": str(weights), "task": "segment", "names": ["4D distribution box"], "error": "",
    })
    handler = SimpleNamespace(parse_form=lambda: form, send_json=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="OBB 或目标检测"):
        sws.SafetyWebHandler.start_material_count(handler)
