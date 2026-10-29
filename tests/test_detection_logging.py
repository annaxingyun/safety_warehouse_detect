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

from common import format_yolo_detection_log, yolo_detection_logging  # noqa: E402
import safety_web_server as server  # noqa: E402
import safety_video_monitor as monitor  # noqa: E402
from ultralytics.engine.results import Results  # noqa: E402


def detection_result() -> Results:
    return Results(
        orig_img=np.zeros((240, 160, 3), dtype=np.uint8), path="frame.jpg",
        names={7: "自定义物资"},
        boxes=torch.tensor([[10.25, 20, 110, 210, 0.95, 7], [12, 22, 112, 212, 0.65, 7]]),
    )


def test_log_contains_model_class_names_confidence_and_pixel_coordinates() -> None:
    message = format_yolo_detection_log(detection_result(), 42)
    assert "frame=42 count=2" in message
    assert "#1 class=7 自定义物资 conf=0.9500 xyxy=[10.25, 20.00, 110.00, 210.00]" in message
    assert "#2 class=7 自定义物资 conf=0.6500" in message


def test_empty_detections_are_logged() -> None:
    message = format_yolo_detection_log(detection_result()[:0], 8)
    assert "frame=8 count=0" in message
    assert "未检测到目标" in message


def test_capture_precedes_tracker_and_does_not_accumulate_callbacks() -> None:
    def tracker(predictor):
        predictor.results = [predictor.results[0][:1]]

    callbacks = [tracker]
    model = SimpleNamespace(callbacks={"on_predict_postprocess_end": callbacks})
    logs = []
    for frame_id in (1, 3):
        predictor = SimpleNamespace(results=[detection_result()])
        with yolo_detection_logging(model, frame_id, lambda frame, message: logs.append((frame, message))):
            for callback in callbacks:
                callback(predictor)
        assert len(predictor.results[0].boxes) == 1
        assert callbacks == [tracker]
    assert [frame for frame, _ in logs] == [1, 3]
    assert all("count=2" in message and "#2" in message for _, message in logs)


def test_capture_is_removed_if_inference_fails() -> None:
    callbacks = []
    model = SimpleNamespace(callbacks={"on_predict_postprocess_end": callbacks})
    with pytest.raises(RuntimeError, match="inference failed"):
        with yolo_detection_logging(model, 1, lambda *_: None):
            raise RuntimeError("inference failed")
    assert callbacks == []


@pytest.fixture
def jobs(monkeypatch):
    isolated = {}
    monkeypatch.setattr(server, "JOBS", isolated)
    server.create_job("job-a", {"ok": True, "job_id": "job-a", "status": "running"})
    server.create_job("job-b", {"ok": True, "job_id": "job-b", "status": "running"})
    return isolated


def test_job_logs_are_incremental_isolated_and_survive_completion(jobs) -> None:
    server.append_job_detection_log("job-a", 9, "first")
    first = server.build_job_status_payload(jobs["job-a"], {"after_log_id": ["0"]})
    assert [entry["message"] for entry in first["detection_logs"]] == ["first"]
    assert first["detection_logs_dropped"] == 0
    assert server.build_job_status_payload(jobs["job-b"])["detection_logs"] == []
    # Frame numbers may restart after a live-stream reconnection; the log cursor must not.
    server.append_job_detection_log("job-a", 1, "last")
    server.update_job("job-a", status="done")
    server.append_job_detection_log("job-a", 2, "late callback")
    final = server.build_job_status_payload(jobs["job-a"], {"after_log_id": ["1"]})
    assert final["status"] == "done"
    assert [(entry["log_id"], entry["message"]) for entry in final["detection_logs"]] == [(2, "last")]
    assert server.build_job_status_payload(jobs["job-a"], {"after_log_id": ["2"]})["detection_logs"] == []
    assert all(not key.startswith("_") for key in final)


def test_job_log_retention_reports_unread_overflow(jobs, monkeypatch) -> None:
    monkeypatch.setattr(server, "MAX_RETAINED_DETECTION_LOGS", 3)
    for frame_id in range(1, 6):
        server.append_job_detection_log("job-a", frame_id, f"frame {frame_id}")
    payload = server.build_job_status_payload(jobs["job-a"], {"after_log_id": ["0"]})
    assert [entry["log_id"] for entry in payload["detection_logs"]] == [3, 4, 5]
    assert payload["detection_logs_dropped"] == 2
    caught_up = server.build_job_status_payload(jobs["job-a"], {"after_log_id": ["5"]})
    assert caught_up["detection_logs"] == []
    assert caught_up["detection_logs_dropped"] == 0


def test_unified_web_job_exposes_detection_logs_on_completion(jobs, monkeypatch, tmp_path) -> None:
    def run_monitor(**kwargs):
        kwargs["detection_log_callback"](42, format_yolo_detection_log(detection_result(), 42))
        return {"result": "unknown"}

    monkeypatch.setattr(monitor, "run_unified_safety_monitor", run_monitor)
    monkeypatch.setattr(server, "build_unified_response", lambda *args: {"output_media_url": ""})
    server.run_unified_safety_job(
        job_id="job-a", enabled_checks={"person_helmet"}, source_type="single_image",
        source=tmp_path / "frame.jpg", weights=tmp_path / "model.pt",
        output_media=tmp_path / "result.jpg", output_json=tmp_path / "result.json",
        expected=0, conf=0.5, iou=0.4, device="cpu", work_options={}, speed_options={}, command=[],
    )
    payload = server.build_job_status_payload(jobs["job-a"])
    assert payload["status"] == "done", payload.get("error")
    assert payload["detection_logs"][0]["frame_id"] == 42
    assert "#2 class=7" in payload["detection_logs"][0]["message"]
