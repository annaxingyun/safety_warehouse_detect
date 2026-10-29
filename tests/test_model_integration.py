"""Smoke tests against the supplied checkpoint and a local validation preview."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "safety_tools"))
from common import load_model, parse_detections
from material_count_cli import MaterialCounter, TrackerArgs
from model_config import MATERIAL_CLASS_NAMES
from safety_video_monitor import SAFETY_CHECK_ORDER, run_unified_safety_monitor


@pytest.fixture(scope="module")
def checkpoint():
    try:
        torch.zeros(1).numpy()
    except RuntimeError:
        pytest.skip("PyTorch/NumPy binary incompatibility in this Python environment")
    path = PROJECT / "all_best.pt"
    if not path.is_file():
        pytest.skip("The supplied all_best.pt checkpoint is not available")
    return path


@pytest.mark.parametrize("device", ["cpu", 0], ids=["cpu", "cuda"])
def test_actual_checkpoint_uses_one_to_many_and_native_nms(checkpoint, device):
    if device == 0 and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    image_path = PROJECT / "all_cls2/val_batch0_pred.jpg"
    if not image_path.exists():
        pytest.skip("Local validation preview is not available")
    model = load_model(checkpoint)
    results = model.predict(source=cv2.imread(str(image_path)), device=device, conf=0.4, verbose=False)
    detections = parse_detections(results[0])
    assert detections and all(len(det["points"]) == 4 for det in detections)
    assert model.predictor.model.end2end is False
    assert model.predictor.args.imgsz == 640 and model.predictor.args.iou == 0.7
    assert model.model.model[-1].cv2 is not None  # one-to-many head survived inference fusion
    assert len(model.names) == 17


def test_actual_checkpoint_runs_all_safety_checks_and_material_page(checkpoint, tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(2):
        assert cv2.imwrite(str(source / f"{index}.jpg"), np.zeros((320, 480, 3), dtype=np.uint8))
    safety = run_unified_safety_monitor(
        set(SAFETY_CHECK_ORDER), "image_sequence", source, str(checkpoint), tmp_path / "safety.jpg",
        conf=0.55, device="cpu",
    )
    assert safety["processed_frames"] == 2
    assert set(safety["results"]) == set(SAFETY_CHECK_ORDER)
    counter = MaterialCounter(str(checkpoint), 0.4, TrackerArgs(), "cpu", target_classes=list(MATERIAL_CLASS_NAMES))
    materials = counter.run("single_image", source / "0.jpg", tmp_path / "materials.jpg")
    assert materials["target_classes"] == list(MATERIAL_CLASS_NAMES)
    assert materials["total"] == 0
    assert (tmp_path / "materials.jpg").is_file()
