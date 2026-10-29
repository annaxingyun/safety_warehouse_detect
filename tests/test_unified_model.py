from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "safety_tools"))
import common
import safety_video_monitor as monitor
from model_config import CLASS_NAMES, MATERIAL_CLASS_NAMES
from ultralytics.engine.results import Results


def obb_result(person_conf=0.9, names=None):
    return Results(
        orig_img=np.zeros((320, 480, 3), dtype=np.uint8), path="frame.jpg", names=names or CLASS_NAMES,
        obb=torch.tensor([
            [180, 220, 40, 120, 0, 11, person_conf, 0],
            [180, 166, 30, 15, 0, 12, 0.9, 1],
            [350, 240, 90, 60, 0.1, 13, 0.9, 3],
            [400, 100, 80, 15, 0, 14, 0.9, 4],
            [180, 100, 200, 20, -0.4, 15, 0.9, 5],
        ]),
    )


def test_obb_parser_preserves_polygon_track_and_model_class_names():
    result = obb_result(names={0: "forklift", 1: "helmet", 3: "person", 4: "fence", 5: "boom"})
    detections = common.parse_detections(result)
    assert len(detections[0]["points"]) == 4
    assert detections[0]["track_id"] == 11
    assert common.filter_detections(detections, "person")[0]["class_id"] == 3
    assert common.filter_detections(detections, "forklift")[0]["class_id"] == 0


CHECK_COMBINATIONS = [set(group) for size in range(1, len(monitor.SAFETY_CHECK_ORDER) + 1)
                      for group in itertools.combinations(monitor.SAFETY_CHECK_ORDER, size)]


@pytest.mark.parametrize("checks", CHECK_COMBINATIONS, ids=lambda checks: ",".join(sorted(checks)))
def test_all_safety_combinations_load_once_and_track_once_per_frame(monkeypatch, tmp_path, checks):
    model = SimpleNamespace(names=CLASS_NAMES, task="obb", callbacks={})
    loads, tracks = [], []
    monkeypatch.setattr(monitor, "load_model", lambda path: loads.append(path) or model)
    monkeypatch.setattr(monitor, "get_image_files", lambda path: [path / "1.jpg", path / "2.jpg"])
    monkeypatch.setattr(monitor.cv2, "imread", lambda _path: np.zeros((320, 480, 3), dtype=np.uint8))

    def track(*args):
        tracks.append(args)
        return obb_result()

    monkeypatch.setattr(monitor, "_track_frame", track)
    result = monitor.run_unified_safety_monitor(
        enabled_checks=checks, source_type="image_sequence", source_path=tmp_path,
        weights_path="all_best.pt", output_media=tmp_path / "result.jpg", conf=0.55,
    )
    assert loads == ["all_best.pt"]
    assert len(tracks) == result["processed_frames"] == 2
    assert set(result["results"]) == checks
    assert all(args[3] == (0.4 if "crane_under_boom" in checks else 0.55) for args in tracks)


def test_shared_person_can_pass_crane_threshold_but_fail_helmet_threshold(monkeypatch, tmp_path):
    model = SimpleNamespace(names=CLASS_NAMES, task="obb", callbacks={})
    monkeypatch.setattr(monitor, "load_model", lambda _path: model)
    monkeypatch.setattr(monitor.cv2, "imread", lambda _path: np.zeros((320, 480, 3), dtype=np.uint8))
    monkeypatch.setattr(monitor, "_track_frame", lambda *args: obb_result(person_conf=0.45))
    result = monitor.run_unified_safety_monitor(
        enabled_checks={"person_helmet", "crane_under_boom"}, source_type="single_image",
        source_path=tmp_path / "input.jpg", weights_path="all_best.pt",
        output_media=tmp_path / "result.jpg", conf=0.55,
    )
    assert result["results"]["person_helmet"]["total_persons"] == 0
    assert result["results"]["crane_under_boom"]["person_count"] == 1


def test_missing_class_is_reported_before_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "load_model", lambda _path: SimpleNamespace(names={0: "person"}, task="obb"))
    with pytest.raises(ValueError, match="boom"):
        monitor.run_unified_safety_monitor(
            {"crane_under_boom"}, "single_image", tmp_path / "frame.jpg", "all_best.pt", None,
        )


def test_page_has_one_safety_model_and_dynamic_material_classes():
    import re
    html = (Path(__file__).resolve().parents[1] / "safety_tools/web/index.html").read_text(encoding="utf-8")
    assert 'name="crane_weights"' not in html
    assert 'name="iou"' not in html
    assert 'name="fence_iou_thres"' in html
    # 物资类别改为按所选模型的 names 动态渲染，页面只保留表格容器和查询接口。
    assert "data-material-target-table" in html
    assert "/api/model-classes" in html
    assert re.findall(r'data-class-name="([^"]+)"', html) == []
