from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_TOOLS = PROJECT_ROOT / "safety_tools"
if str(SAFETY_TOOLS) not in sys.path:
    sys.path.insert(0, str(SAFETY_TOOLS))

import live_stream as live_stream_module  # noqa: E402
import safety_web_server as safety_web_server_module  # noqa: E402

import safety_video_monitor as svm  # noqa: E402
import material_count_cli as mcc  # noqa: E402
import helmet_check as helmet_check_module  # noqa: E402
from common import CLASS_IDS, HazardIncidentTracker, is_valid_box, parse_detections  # noqa: E402
from safety_video_monitor import attach_hazard_event, build_hazard_instances, build_selected_frame_result  # noqa: E402
from safety_web_server import (  # noqa: E402
    build_job_status_payload,
    build_safety_preview_updates,
    create_job,
    create_speed_command,
    field_value,
    field_values,
    is_annotated_image_updated,
    prune_finished_jobs,
    select_new_hazard_event,
    start_managed_job,
    update_job,
    update_safety_job_preview,
    validate_live_stream_url,
    write_json_atomic,
)
from live_stream import (  # noqa: E402
    LiveStreamReader,
    acquire_shared_live_reader,
    install_aiortc_decoder_resilience,
    is_whep_source,
    normalize_whep_url,
    parse_webrtc_answer,
    preferred_whep_video_codecs,
    receiver_packet_counts,
    reconnect_backoff_seconds,
    request_whep_keyframe,
    release_shared_live_reader,
    shared_live_reader_is_ready,
    sdp_video_codec_names,
    strip_auxiliary_video_codecs,
)
from forklift_work_check import (  # noqa: E402
    build_circle_work_zone,
    build_output as build_forklift_work_output,
    check_person_nearby_circle_rule,
    draw_results as draw_forklift_work_results,
    evaluate_helmet_status,
    evaluate_forklifts,
    draw_transparent_polygon,
    generate_ground_point_work_area,
)
from forklift_speed_check import estimate_meters_per_pixel, update_speeds  # noqa: E402
from helmet_check import build_output as build_helmet_output, evaluate_persons  # noqa: E402


class _FormField:
    def __init__(self, value: str, filename: str | None = None) -> None:
        self.value = value
        self.filename = filename


def _detection(class_name: str, box: list[float], confidence: float = 0.9) -> dict:
    return {
        "class_id": CLASS_IDS[class_name],
        "class_name": class_name,
        "box": box,
        "confidence": confidence,
        "track_id": None,
    }


@pytest.mark.parametrize("entrypoint", ["image", "helmet_frame", "unified"])
@pytest.mark.parametrize("class_name, first_box, second_box", [
    ("person", [10, 10, 110, 210], [115, 10, 150, 210]),
    ("head_no_helmet", [10, 10, 40, 40], [100, 10, 140, 40]),
])
def test_safety_entrypoints_report_person_or_head_only_detections_as_unsafe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: str,
    class_name: str, first_box: list[int], second_box: list[int],
) -> None:
    class OutputBoxes:
        """Inference output double with the tensor views used by detection parsing."""

        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=float)
            self.xyxy = self.rows[:, :4]
            self.id = self.rows[:, 4]
            self.conf = self.rows[:, 5]
            self.cls = self.rows[:, 6]

        def cpu(self):
            return self

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, indices):
            return OutputBoxes(self.rows[indices])

    # Stub the already postprocessed model output to exercise each safety entrypoint.
    output = SimpleNamespace(boxes=OutputBoxes([
        [*first_box, 11, 0.95, CLASS_IDS[class_name]],
        [*second_box, 12, 0.85, CLASS_IDS[class_name]],
    ]))
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(helmet_check_module.cv2, "imread", lambda _path: frame.copy())

    if entrypoint == "image":
        monkeypatch.setattr(sys, "argv", [
            "helmet_check.py", "--weights", "unused.pt", "--image", "frame.jpg",
            "--output", str(tmp_path / "result.jpg"), "--device", "cpu",
        ])
        monkeypatch.setattr(helmet_check_module, "load_model", lambda _path: object())
        monkeypatch.setattr(helmet_check_module, "run_detection", lambda *args, **kwargs: output)
        captured = []
        monkeypatch.setattr(helmet_check_module, "print_json", captured.append)
        helmet_check_module.main()
        result = captured[0]
    else:
        monkeypatch.setattr(svm, "_track_frame", lambda *args, **kwargs: output)
        if entrypoint == "helmet_frame":
            smoother = svm.HelmetTrackSmoother()
            for frame_id in range(1, 6):
                _, result = svm.analyze_helmet_frame(
                    object(), frame, frame_id, 0.5, 0.4, "cpu", "unused.yaml", smoother
                )
                if frame_id < 5:
                    assert result["result"] == "unknown"
                    assert result["no_helmet_count"] == 0
        else:
            monkeypatch.setattr(svm, "load_model", lambda _path: SimpleNamespace(names={v: k for k, v in CLASS_IDS.items()}))
            result = svm.run_unified_safety_monitor(
                enabled_checks={"person_helmet"}, source_type="single_image",
                source_path=tmp_path / "frame.jpg", weights_path="unused.pt", output_media=None,
            )["results"]["person_helmet"]

    assert result["total_persons"] == 2
    assert [person["track_id"] for person in result["persons"]] == [11, 12]
    assert [person["box"] for person in result["persons"]] == [first_box, second_box]
    assert result["unknown_count"] == 0
    assert result["no_helmet_count"] == 2
    assert result["result"] == "unsafe"
    assert len(output.boxes) == 2


def test_helmet_result_is_unsafe_when_only_person_is_detected() -> None:
    detections = [_detection("person", [10, 10, 110, 210])]

    persons = evaluate_persons(detections, (240, 160, 3))
    result = build_helmet_output("frame.jpg", persons)

    assert persons[0]["status"] == "no_helmet"
    assert result["result"] == "unsafe"
    assert result["has_hazard"] is True


def test_helmet_result_is_unknown_when_only_helmet_is_detected() -> None:
    detections = [_detection("helmet", [35, 15, 75, 45])]

    persons = evaluate_persons(detections, (240, 160, 3))
    result = build_helmet_output("frame.jpg", persons)

    assert persons == []
    assert result["result"] == "unknown"
    assert result["has_hazard"] is False


def test_forklift_helmet_evaluation_flags_person_alone() -> None:
    detections = [_detection("person", [10, 10, 110, 210])]

    persons = evaluate_helmet_status(detections, (240, 160, 3), 0.2)

    assert persons[0]["status"] == "no_helmet"


def test_any_fence_in_frame_counts_as_forklift_enclosure() -> None:
    detections = [
        _detection("forklift", [100, 100, 220, 260]),
        _detection("fence", [850, 700, 980, 850]),
    ]

    forklifts = evaluate_forklifts(
        detections,
        (1000, 1000, 3),
        left_scale=1.2,
        right_scale=1.2,
        front_scale=1.0,
        back_scale=0.35,
        fence_iou_thres=0.01,
        height_depth_ratio=0.6,
        width_depth_ratio=0.8,
        driver_iou_thres=0.05,
        driver_inside_ratio_thres=0.4,
        driver_dx_norm_thres=0.45,
        driver_min_dy_up_norm=0.35,
        driver_max_dy_up_norm=0.85,
        person_distance_width_scale=1.2,
        person_distance_height_scale=0.8,
        min_person_distance_px=180.0,
        enable_fence_zone=True,
        fence_zone_padding=20.0,
    )

    assert forklifts[0]["has_fence_nearby"] is True
    assert "NO_FENCE_NEAR_FORKLIFT" not in forklifts[0]["violations"]
    assert forklifts[0]["is_safe"] is True


def test_moving_forklift_is_excluded_from_fence_check() -> None:
    work_result = {
        "fence_count": 0,
        "forklifts": [
            {
                "forklift_id": 7,
                "forklift_box": [100, 100, 220, 260],
                "has_fence_nearby": False,
                "nearby_fences": [],
            }
        ],
    }
    speed_result = {
        "movement_threshold_kmh": 1.0,
        "forklifts": [{"track_id": 7, "smoothed_speed_kmh": 3.5}],
    }

    frame_result = build_selected_frame_result(
        {"forklift_fence"},
        2,
        [],
        work_result,
        speed_result,
    )
    fence_result = frame_result["results"]["forklift_fence"]

    assert fence_result["result"] == "unknown"
    assert fence_result["checked_forklift_count"] == 0
    assert fence_result["moving_forklift_count"] == 1
    assert fence_result["forklifts"][0]["fence_check_applicable"] is False
    assert build_hazard_instances("safety_combined", frame_result) == []


def test_moving_forklift_skips_only_fence_check() -> None:
    nearby_person = {"track_id": 10, "box": [10, 10, 50, 100], "status": "no_helmet"}
    driver = {
        "track_id": 11,
        "box": [110, 110, 150, 210],
        "helmet_status": "no_helmet",
    }
    work_result = {
        "fence_count": 0,
        "forklifts": [{
            "forklift_id": 7,
            "forklift_box": [100, 100, 220, 260],
            "has_person_nearby": True,
            "nearby_persons": [nearby_person],
            "excluded_persons": [driver],
            "has_fence_nearby": False,
            "nearby_fences": [],
        }],
    }
    speed_result = {
        "result": "unsafe",
        "movement_threshold_kmh": 1.0,
        "forklift_count": 1,
        "forklifts": [{
            "track_id": 7,
            "box": [100, 100, 220, 260],
            "current_speed_px_per_frame": 3.5,
            "current_speed_kmh": 7.0,
            "smoothed_speed_kmh": 7.0,
            "overspeed": True,
        }],
    }

    frame_result = build_selected_frame_result(
        {"person_helmet", "driver_helmet", "forklift_proximity", "forklift_fence", "forklift_speed"},
        2,
        [nearby_person, {"track_id": 11, "box": driver["box"], "status": "no_helmet"}],
        work_result,
        speed_result,
    )
    assert frame_result["results"]["forklift_fence"]["result"] == "unknown"
    assert frame_result["results"]["forklift_fence"]["forklifts"][0]["fence_check_applicable"] is False
    assert frame_result["results"]["person_helmet"]["result"] == "unsafe"
    assert frame_result["results"]["driver_helmet"]["result"] == "unsafe"
    assert frame_result["results"]["forklift_proximity"]["result"] == "unsafe"
    assert frame_result["results"]["forklift_speed"]["result"] == "unsafe"
    assert set(frame_result["results"]) == {
        "person_helmet",
        "driver_helmet",
        "forklift_proximity",
        "forklift_fence",
        "forklift_speed",
    }
    hazard_codes = {item["code"] for item in build_hazard_instances("safety_combined", frame_result)}
    assert "NO_FENCE_NEAR_FORKLIFT" not in hazard_codes
    assert {"NO_HELMET", "DRIVER_NO_HELMET", "PERSON_NEAR_FORKLIFT", "FORKLIFT_OVERSPEED"} <= hazard_codes


def test_stationary_forklift_without_fence_remains_unsafe() -> None:
    work_result = {
        "fence_count": 0,
        "forklifts": [
            {
                "forklift_id": 8,
                "forklift_box": [100, 100, 220, 260],
                "has_fence_nearby": False,
                "nearby_fences": [],
            }
        ],
    }
    speed_result = {
        "movement_threshold_kmh": 1.0,
        "forklifts": [{"track_id": 8, "smoothed_speed_kmh": 0.4}],
    }

    frame_result = build_selected_frame_result(
        {"forklift_fence"},
        2,
        [],
        work_result,
        speed_result,
    )

    assert frame_result["results"]["forklift_fence"]["result"] == "unsafe"
    assert len(build_hazard_instances("safety_combined", frame_result)) == 1


def test_each_forklift_uses_its_own_box_matched_motion_state() -> None:
    work_result = {
        "fence_count": 0,
        "forklifts": [
            {
                "forklift_id": 1,
                "forklift_box": [10, 10, 110, 110],
                "has_fence_nearby": False,
                "nearby_fences": [],
            },
            {
                "forklift_id": 2,
                "forklift_box": [200, 10, 300, 110],
                "has_fence_nearby": False,
                "nearby_fences": [],
            },
        ],
    }
    # IDs are deliberately crossed to reproduce temporary tracker ID
    # mismatch. Box A is moving and box B is stationary.
    speed_result = {
        "movement_threshold_kmh": 1.0,
        "forklifts": [
            {"track_id": 2, "box": [10, 10, 110, 110], "smoothed_speed_kmh": 3.5},
            {"track_id": 1, "box": [200, 10, 300, 110], "smoothed_speed_kmh": 0.2},
        ],
    }

    frame_result = build_selected_frame_result(
        {"forklift_fence"},
        2,
        [],
        work_result,
        speed_result,
    )
    fence_result = frame_result["results"]["forklift_fence"]

    assert fence_result["moving_forklift_count"] == 1
    assert fence_result["checked_forklift_count"] == 1
    assert fence_result["forklifts"][0]["is_moving"] is True
    assert fence_result["forklifts"][0]["fence_check_applicable"] is False
    assert fence_result["forklifts"][1]["is_moving"] is False
    assert fence_result["forklifts"][1]["fence_check_applicable"] is True
    assert fence_result["result"] == "unsafe"


def test_fence_motion_state_uses_smoothed_not_instantaneous_speed() -> None:
    work_result = {
        "fence_count": 0,
        "forklifts": [
            {
                "forklift_id": 1,
                "forklift_box": [10, 10, 110, 110],
                "has_fence_nearby": False,
                "nearby_fences": [],
            },
            {
                "forklift_id": 2,
                "forklift_box": [200, 10, 300, 110],
                "has_fence_nearby": False,
                "nearby_fences": [],
            },
        ],
    }
    speed_result = {
        "movement_threshold_kmh": 1.0,
        "forklifts": [
            {
                "track_id": 1,
                "box": [10, 10, 110, 110],
                "current_speed_kmh": 4.0,
                "smoothed_speed_kmh": 0.4,
            },
            {
                "track_id": 2,
                "box": [200, 10, 300, 110],
                "current_speed_kmh": 0.2,
                "smoothed_speed_kmh": 2.0,
            },
        ],
    }

    frame_result = build_selected_frame_result(
        {"forklift_fence"}, 2, [], work_result, speed_result
    )
    forklifts = frame_result["results"]["forklift_fence"]["forklifts"]

    assert forklifts[0]["is_moving"] is False
    assert forklifts[0]["fence_check_applicable"] is True
    assert forklifts[1]["is_moving"] is True
    assert forklifts[1]["fence_check_applicable"] is False


def test_unified_monitor_draws_forklift_radius_overlay() -> None:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    polygon = [[50, 100], [250, 100], [250, 260], [50, 260]]
    frame_result = {
        "result": "safe",
        "results": {
            "forklift_proximity": {
                "result": "safe",
                "forklifts": [
                    {
                        "forklift_id": 1,
                        "forklift_box": [120, 140, 180, 220],
                        "has_person_nearby": False,
                        "nearby_persons": [],
                        "circle_work_zone": {
                            "enabled": True,
                            "center": [150, 220],
                            "polygon": polygon,
                        },
                    }
                ],
            }
        },
    }

    annotated = svm.draw_selected_frame(
        frame, {"forklift_proximity"}, frame_result, []
    )

    assert annotated[180, 150].any()
    assert annotated[100, 50].any()
    assert not annotated[20, 20].any()


def test_person_foot_point_inside_forklift_radius_is_detected() -> None:
    forklift_box = [120, 120, 220, 260]
    work_zone = [80, 80, 260, 280]
    circle = build_circle_work_zone(
        forklift_box,
        work_zone,
        forklift_radius_scale=1.2,
        min_forklift_radius_px=100.0,
        max_forklift_radius_px=450.0,
    )
    person = {"box": [230, 130, 270, 250]}

    result = check_person_nearby_circle_rule(
        person,
        forklift_box,
        (170.0, 260.0),
        circle,
    )

    assert result["matched"] is True
    assert result["reasons"] == ["in_circle_work_zone"]


def test_overhead_camera_uses_detection_box_center_and_true_circle() -> None:
    forklift_box = [100, 80, 220, 200]
    work_zone = [40, 20, 280, 240]
    zone = build_circle_work_zone(
        forklift_box,
        work_zone,
        forklift_radius_scale=1.2,
        min_forklift_radius_px=100.0,
        max_forklift_radius_px=450.0,
        road_direction_angle=37.0,
        road_direction_confidence=0.9,
        camera_mode="overhead",
    )

    assert zone["center"] == [160.0, 140.0]
    assert zone["shape"] == "circle"
    assert zone["radius_x_px"] == zone["radius_y_px"] == zone["radius_px"]
    assert zone["radius_px"] == pytest.approx(72.0)
    assert zone["rotation_angle_deg"] == 0.0


def test_work_zone_radius_ignores_expanded_zone_and_legacy_pixel_limits() -> None:
    zone = build_circle_work_zone(
        [100, 80, 220, 200],
        [-1000, -1000, 1000, 1000],
        forklift_radius_scale=1.2,
        min_forklift_radius_px=1000.0,
        max_forklift_radius_px=1.0,
        camera_mode="overhead",
    )

    assert zone["radius_px"] == pytest.approx(72.0)


def test_oblique_camera_keeps_ground_point_perspective_ellipse() -> None:
    zone = build_circle_work_zone(
        [100, 80, 220, 240],
        [40, 20, 280, 240],
        forklift_radius_scale=1.2,
        min_forklift_radius_px=100.0,
        max_forklift_radius_px=450.0,
        road_direction_angle=37.0,
        camera_mode="oblique",
    )

    assert zone["center"] == [160.0, 240.0]
    assert zone["shape"] == "ellipse"
    assert zone["rotation_angle_deg"] == 37.0
    assert zone["radius_x_px"] != zone["radius_y_px"]
    assert zone["radius_x_px"] / zone["radius_y_px"] == pytest.approx(2.5 / 1.5)


def test_oblique_camera_keeps_ellipse_for_square_box_without_road_direction() -> None:
    zone = build_circle_work_zone(
        [100, 80, 220, 200],
        [40, 20, 280, 240],
        forklift_radius_scale=1.2,
        min_forklift_radius_px=100.0,
        max_forklift_radius_px=450.0,
        road_direction_angle=None,
        camera_mode="oblique",
    )

    assert zone["shape"] == "ellipse"
    assert zone["radius_x_px"] == pytest.approx(72.0)
    assert zone["radius_y_px"] == pytest.approx(43.2)


def test_circle_rule_uses_configured_zone_center() -> None:
    result = check_person_nearby_circle_rule(
        {"box": [145, 105, 175, 140]},
        [100, 80, 220, 200],
        (160.0, 200.0),
        {
            "center": [160.0, 140.0],
            "radius_px": 25.0,
            "radius_x_px": 25.0,
            "radius_y_px": 25.0,
            "shape": "circle",
        },
    )

    assert result["matched"] is True


def test_safety_output_draws_raw_detection_boxes_without_text(monkeypatch) -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [
        _detection("person", [10, 10, 40, 80]),
        _detection("helmet", [18, 12, 32, 25]),
        _detection("forklift", [60, 30, 110, 90]),
    ]

    def reject_text(*args, **kwargs):
        raise AssertionError("safety output must not draw text")

    monkeypatch.setattr(svm.cv2, "putText", reject_text)
    annotated = svm.draw_selected_frame(
        frame,
        {"person_helmet", "forklift_fence"},
        {"selected_checks": ["person_helmet", "forklift_fence"], "results": {}},
        detections,
    )

    assert annotated[10, 10].any()
    assert annotated[12, 18].any()
    assert annotated[30, 60].any()
    assert not annotated[50, 50].any()


def test_unsafe_person_detection_box_is_red() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [_detection("person", [10, 10, 40, 80])]
    annotated = svm.draw_selected_frame(
        frame,
        {"person_helmet"},
        {
            "result": "unknown",
            "results": {
                "person_helmet": {
                    "result": "unsafe",
                    "persons": [{"status": "no_helmet", "box": [10, 10, 40, 80]}],
                }
            },
        },
        detections,
    )

    assert annotated[10, 10].tolist() == [0, 0, 255]


def test_repeated_text_form_fields_are_supported() -> None:
    form = {
        "target_classes": [_FormField("pallet"), _FormField("carton")],
        "source_path": _FormField(r"C:\\ruoyi\\uploadPath\\video.mp4"),
    }

    assert field_values(form, "target_classes") == ["pallet", "carton"]
    assert field_value(form, "target_classes") == "pallet"
    assert "\n".join(field_values(form, "target_classes")) == "pallet\ncarton"
    assert field_value(form, "source_path") == r"C:\\ruoyi\\uploadPath\\video.mp4"


def test_whep_url_detection_and_normalization() -> None:
    assert is_whep_source("https://media.example.com/rtc/v1/whep/?app=live")
    assert is_whep_source("whep+https://media.example.com/live/camera-1")
    assert is_whep_source("http://127.0.0.1:8090/index/api/webrtc?app=live&stream=test&type=play")
    assert not is_whep_source("http://127.0.0.1:8090/index/api/webrtc?app=live&stream=test&type=push")
    assert not is_whep_source("rtsp://camera.example.com/live")
    assert normalize_whep_url("whep+https://media.example.com/live/camera-1") == "https://media.example.com/live/camera-1"


def test_live_stream_input_rejects_non_webrtc_protocols() -> None:
    assert validate_live_stream_url("http://media.example.com/index/api/whep?app=live&stream=cam")
    with pytest.raises(ValueError, match="WebRTC/WHEP"):
        validate_live_stream_url("rtsp://media.example.com/live/cam")


def test_live_stream_reader_always_uses_webrtc_backend(monkeypatch) -> None:
    reader = LiveStreamReader("http://media.example.com/custom/playback")
    calls: list[bool] = []
    monkeypatch.setattr(reader, "_run_whep", lambda: calls.append(True))

    reader._run()

    assert calls == [True]


def test_shared_live_reader_handoff_keeps_one_connection(monkeypatch) -> None:
    timers = []

    class FakeReader:
        def __init__(self, source):
            self.source = source
            self.open_timeout_ms = 5000
            self.callbacks = []
            self.start_count = 0
            self.stop_count = 0
            self.join_count = 0

        def start(self):
            self.start_count += 1
            return self

        def add_frame_callback(self, callback):
            if callback is not None and callback not in self.callbacks:
                self.callbacks.append(callback)

        def remove_frame_callback(self, callback):
            self.callbacks = [item for item in self.callbacks if item is not callback]

        def stop(self):
            self.stop_count += 1

        def join(self, timeout=0):
            self.join_count += 1

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.cancelled = False
            self.daemon = False
            timers.append(self)

        def start(self):
            return None

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(live_stream_module, "LiveStreamReader", FakeReader)
    monkeypatch.setattr(live_stream_module.threading, "Timer", FakeTimer)
    live_stream_module._SHARED_LIVE_READERS.clear()

    preview_callback = lambda packet: None
    algorithm_callback = lambda packet: None
    preview_reader = acquire_shared_live_reader("http://media/webrtc", preview_callback)
    algorithm_reader = acquire_shared_live_reader("http://media/webrtc", algorithm_callback)

    assert preview_reader is algorithm_reader
    assert preview_reader.start_count == 1
    release_shared_live_reader("http://media/webrtc", preview_reader, preview_callback, idle_seconds=0)
    assert preview_reader.stop_count == 0
    # Release is idempotent: a repeated cleanup from one consumer must not
    # steal the algorithm consumer's still-active reference.
    release_shared_live_reader("http://media/webrtc", preview_reader, preview_callback, idle_seconds=0)
    assert preview_reader.stop_count == 0
    release_shared_live_reader("http://media/webrtc", algorithm_reader, algorithm_callback, idle_seconds=0)
    assert preview_reader.stop_count == 0

    resumed_reader = acquire_shared_live_reader("http://media/webrtc")
    assert resumed_reader is preview_reader
    assert timers[-1].cancelled is True
    release_shared_live_reader("http://media/webrtc", resumed_reader, idle_seconds=0)
    timers[-1].callback()
    assert preview_reader.stop_count == 1
    assert preview_reader.join_count == 1
    live_stream_module._SHARED_LIVE_READERS.clear()


def test_shared_live_reader_readiness_requires_recent_connected_frame() -> None:
    source = "http://media.example.com/index/api/whep?app=live&stream=ready-test"

    class HealthReader:
        def __init__(self) -> None:
            self.snapshot = {
                "thread_alive": True,
                "opened_once": True,
                "connected": True,
                "recovering": False,
                "last_frame_age_seconds": 0.2,
            }

        def health_snapshot(self):
            return dict(self.snapshot)

    reader = HealthReader()
    key = live_stream_module._shared_stream_key(source)
    entry = live_stream_module._SharedLiveReaderEntry(reader=reader, references=1)
    live_stream_module._SHARED_LIVE_READERS[key] = entry
    try:
        assert shared_live_reader_is_ready(source)
        reader.snapshot["recovering"] = True
        assert not shared_live_reader_is_ready(source)
        reader.snapshot["recovering"] = False
        reader.snapshot["last_frame_age_seconds"] = 4.0
        assert not shared_live_reader_is_ready(source, max_frame_age=3.0)
    finally:
        live_stream_module._SHARED_LIVE_READERS.pop(key, None)


def test_aiortc_decoder_worker_discards_bad_unit_and_keeps_running(monkeypatch) -> None:
    aiortc_module = ModuleType("aiortc")
    aiortc_module.__path__ = []
    receiver_module = ModuleType("aiortc.rtcrtpreceiver")
    receiver_module.decoder_worker = lambda *args: None
    receiver_module.get_decoder = lambda codec: None
    aiortc_module.rtcrtpreceiver = receiver_module
    monkeypatch.setitem(sys.modules, "aiortc", aiortc_module)
    monkeypatch.setitem(sys.modules, "aiortc.rtcrtpreceiver", receiver_module)

    original_worker = receiver_module.decoder_worker
    original_flag = getattr(receiver_module, "_safety_decoder_resilience_installed", None)

    class Decoder:
        def __init__(self) -> None:
            self.calls = 0

        def decode(self, encoded_frame):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("damaged H264 access unit")
            return [f"decoded:{encoded_frame}"]

    decoder = Decoder()
    original_get_decoder = receiver_module.get_decoder
    try:
        receiver_module._safety_decoder_resilience_installed = False
        receiver_module.get_decoder = lambda codec: decoder
        assert install_aiortc_decoder_resilience()

        input_q = queue.Queue()
        output_q = asyncio.Queue()
        codec = SimpleNamespace(name="H264")
        input_q.put((codec, "bad"))
        input_q.put((codec, "good"))
        input_q.put(None)
        loop = asyncio.new_event_loop()
        try:
            receiver_module.decoder_worker(loop, input_q, output_q)
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()

        assert decoder.calls == 2
        assert output_q.get_nowait() == "decoded:good"
        assert output_q.get_nowait() is None
    finally:
        receiver_module.decoder_worker = original_worker
        receiver_module.get_decoder = original_get_decoder
        if original_flag is None:
            with contextlib.suppress(AttributeError):
                del receiver_module._safety_decoder_resilience_installed
        else:
            receiver_module._safety_decoder_resilience_installed = original_flag

def test_target_whep_url_and_h264_sdp_filtering() -> None:
    target = "http://192.168.119.20:8090/index/api/whep?app=live&stream=CZI202605200048_81_0_0"
    assert is_whep_source(target)
    assert normalize_whep_url("whep+" + target) == target

    answer = "\r\n".join([
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 96 97 98 99",
        "a=rtpmap:96 H264/90000",
        "a=fmtp:96 packetization-mode=1;profile-level-id=42e01f",
        "a=rtpmap:97 rtx/90000",
        "a=fmtp:97 apt=96",
        "a=rtpmap:98 red/90000",
        "a=rtpmap:99 ulpfec/90000",
        "",
    ])
    filtered = strip_auxiliary_video_codecs(answer)
    assert "m=video 9 UDP/TLS/RTP/SAVPF 96" in filtered
    assert "H264/90000" in filtered
    assert "rtx/90000" not in filtered
    assert "red/90000" not in filtered
    assert "ulpfec/90000" not in filtered
    assert sdp_video_codec_names(filtered) == ["H264"]

    recovery_filtered = strip_auxiliary_video_codecs(answer, keep_rtx=True)
    assert "rtx/90000" in recovery_filtered
    assert "a=fmtp:97 apt=96" in recovery_filtered
    assert "red/90000" not in recovery_filtered
    assert "ulpfec/90000" not in recovery_filtered


def test_whep_codec_preferences_keep_vp8_as_h264_fallback() -> None:
    codecs = [
        SimpleNamespace(mimeType="video/VP8", name="vp8"),
        SimpleNamespace(mimeType="video/rtx", name="rtx"),
        SimpleNamespace(mimeType="video/H264", name="h264-baseline"),
        SimpleNamespace(mimeType="video/VP9", name="vp9"),
        SimpleNamespace(mimeType="video/H264", name="h264-constrained"),
    ]

    preferred = preferred_whep_video_codecs(codecs)
    assert [codec.name for codec in preferred] == ["h264-baseline", "h264-constrained", "rtx", "vp8"]

    vp8_answer = "\r\n".join([
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 100",
        "a=rtpmap:100 VP8/90000",
        "",
    ])
    assert sdp_video_codec_names(vp8_answer) == ["VP8"]


def test_whep_reconnect_backoff_is_fast_then_bounded() -> None:
    assert reconnect_backoff_seconds(0) == 1.0
    assert reconnect_backoff_seconds(1) == 1.0
    assert reconnect_backoff_seconds(2) == 2.0
    assert reconnect_backoff_seconds(3) == 4.0
    assert reconnect_backoff_seconds(20) == 8.0
    assert reconnect_backoff_seconds(3, base_delay=0.25, max_delay=1.0) == 1.0


def test_whep_receiver_packet_counts_aggregate_video_and_ignore_failures() -> None:
    class Receiver:
        async def getStats(self):
            return {
                "video": SimpleNamespace(type="inbound-rtp", kind="video", packetsReceived=120, packetsLost=3),
                "audio": SimpleNamespace(type="inbound-rtp", kind="audio", packetsReceived=50, packetsLost=2),
                "transport": SimpleNamespace(type="transport"),
            }

    class BrokenReceiver:
        async def getStats(self):
            raise RuntimeError("stats unavailable")

    assert asyncio.run(receiver_packet_counts(Receiver())) == (120, 3)
    assert asyncio.run(receiver_packet_counts(BrokenReceiver())) == (0, 0)


def test_webrtc_answer_parser_supports_raw_sdp_and_zlm_json() -> None:
    sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 126\r\na=rtpmap:126 H264/90000\r\n"
    assert parse_webrtc_answer(sdp) == (sdp.strip(), "answer")

    wrapped = json.dumps({"code": 0, "id": "zlm_1", "sdp": sdp, "type": "answer"})
    parsed_sdp, parsed_type = parse_webrtc_answer(wrapped)
    assert parsed_sdp == sdp.strip()
    assert parsed_type == "answer"
    assert sdp_video_codec_names(parsed_sdp) == ["H264"]


def test_whep_keyframe_request_sends_pli_for_each_active_source() -> None:
    class Source:
        def __init__(self, source: int) -> None:
            self.source = source

    class Receiver:
        def __init__(self) -> None:
            self.requested: list[int] = []

        def getSynchronizationSources(self) -> list[Source]:
            return [Source(101), Source(202), Source(101)]

        async def _send_rtcp_pli(self, source: int) -> None:
            self.requested.append(source)

    receiver = Receiver()
    assert asyncio.run(request_whep_keyframe(receiver)) is True
    assert sorted(receiver.requested) == [101, 202]


def test_whep_keyframe_request_is_optional_for_aiortc_compatibility() -> None:
    class ReceiverWithoutPrivatePli:
        def getSynchronizationSources(self) -> list[object]:
            return []

    assert asyncio.run(request_whep_keyframe(ReceiverWithoutPrivatePli())) is False


def test_whep_initial_keyframe_request_uses_sdp_before_any_rtp_arrives() -> None:
    from live_stream import sdp_video_source_ids
    sdp = ("v=0\r\nm=audio 9 RTP/AVP 0\r\na=ssrc:99 cname:audio\r\n"
           "m=video 9 UDP/TLS/RTP/SAVPF 126 127\r\na=ssrc-group:FID 101 202\r\n"
           "a=ssrc:101 cname:video\r\na=ssrc:202 cname:rtx\r\n")
    assert sdp_video_source_ids(sdp) == {101}
    requested = []

    async def send_pli(source):
        requested.append(source)

    receiver = SimpleNamespace(getSynchronizationSources=lambda: [], _send_rtcp_pli=send_pli)
    assert asyncio.run(request_whep_keyframe(receiver, sdp_video_source_ids(sdp))) is True
    assert requested == [101]


def test_material_video_sampling_only_detects_selected_frames_and_adjusts_output_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = [np.full((8, 10, 3), index, dtype=np.uint8) for index in range(1, 7)]

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == mcc.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == mcc.cv2.CAP_PROP_FRAME_COUNT:
                return float(len(frames))
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame.copy()

        def release(self) -> None:
            pass

    class FakeWriter:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())

        def release(self) -> None:
            pass

    created: dict[str, object] = {}

    def create_writer(_path: str, fps: float, _width: int, _height: int, *_args: object) -> FakeWriter:
        writer = FakeWriter()
        created.update(fps=fps, writer=writer)
        return writer

    monkeypatch.setattr(mcc.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(mcc, "write_tracker_yaml", lambda _args: tmp_path / "tracker.yaml")
    monkeypatch.setattr(mcc, "draw_info_panel", lambda frame, *_args: frame)
    monkeypatch.setattr(mcc, "make_preview_frame", lambda frame: frame)
    monkeypatch.setattr(mcc, "create_video_writer", create_writer)

    source = tmp_path / "input.mp4"
    source.touch()
    counter = mcc.MaterialCounter("model.pt", 0.3, mcc.TrackerArgs(), "cpu", sample_interval=2)
    detected_frame_ids: list[int] = []

    def process_frame(
        frame: np.ndarray, frame_idx: int, _detector: object, _class_names: dict[int, str], _tracker: Path
    ) -> tuple[np.ndarray, int]:
        detected_frame_ids.append(frame_idx)
        return frame.copy(), 0

    monkeypatch.setattr(counter, "_process_frame", process_frame)
    counter._run_video(object(), {}, source, tmp_path / "result.mp4")

    assert detected_frame_ids == [1, 3, 5]
    assert created["fps"] == pytest.approx(15.0)
    assert len(created["writer"].frames) == 3


def test_forklift_scale_uses_dimensions_for_selected_camera_mode() -> None:
    # 30x40 px gives a 50 px image diagonal. Overhead views expose the
    # length/width plane, while oblique views expose length/height.
    overhead_scale = (2.5**2 + 1.5**2) ** 0.5 / 50.0
    oblique_scale = (2.5**2 + 2.0**2) ** 0.5 / 50.0
    assert estimate_meters_per_pixel([0, 0, 30, 40], "overhead") == pytest.approx(overhead_scale)
    assert estimate_meters_per_pixel([0, 0, 30, 40], "oblique") == pytest.approx(oblique_scale)


def test_speed_result_reports_selected_camera_calibration() -> None:
    item = update_speeds(
        [{"track_id": 1, "box": [0, 0, 30, 40]}],
        {},
        defaultdict(lambda: deque(maxlen=30)),
        None,
        False,
        "affine",
        5.0,
        camera_mode="overhead",
    )[0]

    assert item["speed_calibration"]["camera_mode"] == "overhead"
    assert item["speed_calibration"]["calibration_axes"] == ["length", "width"]


def test_speed_command_forwards_camera_mode() -> None:
    command = create_speed_command(
        Path("weights.pt"),
        Path("input.mp4"),
        Path("output.mp4"),
        Path("output.json"),
        "5.0",
        "0.5",
        "30",
        "20",
        "affine",
        "cpu",
        camera_mode="overhead",
    )

    assert command[command.index("--camera-mode") + 1] == "overhead"


def test_overspeed_uses_30_frame_average_instead_of_instantaneous_speed() -> None:
    def calculate(previous_speeds: list[float]) -> dict:
        previous_points = {1: (50.0, 200.0)}
        history = defaultdict(lambda: deque(maxlen=30))
        history[1].extend(previous_speeds)
        return update_speeds(
            [{"track_id": 1, "box": [6, 0, 106, 200]}],
            previous_points,
            history,
            None,
            False,
            "affine",
            5.0,
            fps=25.0,
            unsafe_speed_kmh=6.0,
        )[0]

    one_frame_spike = calculate([5.0] * 29)
    sustained_overspeed = calculate([7.0] * 29)
    assert one_frame_spike["current_speed_kmh"] > 6.0
    assert one_frame_spike["smoothed_speed_kmh"] < 6.0
    assert one_frame_spike["decision_speed_kmh"] == one_frame_spike["smoothed_speed_kmh"]
    assert one_frame_spike["overspeed"] is False
    assert sustained_overspeed["smoothed_speed_kmh"] > 6.0
    assert sustained_overspeed["overspeed"] is True
    assert sustained_overspeed["speed_average_window_frames"] == 30


def test_speed_average_uses_all_available_samples_before_window_is_full() -> None:
    previous_points = {1: (50.0, 200.0)}
    history = defaultdict(lambda: deque(maxlen=30))
    history[1].append(4.0)
    item = update_speeds(
        [{"track_id": 1, "box": [6, 0, 106, 200]}],
        previous_points,
        history,
        None,
        False,
        "affine",
        5.0,
        fps=25.0,
        unsafe_speed_kmh=6.0,
    )[0]

    assert item["speed_average_sample_count"] == 2
    assert item["smoothed_speed_kmh"] == pytest.approx((4.0 + item["current_speed_kmh"]) / 2.0, abs=0.01)
    assert item["overspeed"] is False
    assert item["speed_status"] == "ok"


def test_alarm_uses_same_two_decimal_average_as_the_ui() -> None:
    def calculate(previous_speeds: list[float]) -> dict:
        history = defaultdict(lambda: deque(maxlen=30))
        history[1].extend(previous_speeds)
        return update_speeds(
            [{"track_id": 1, "box": [6, 0, 106, 200]}],
            {1: (50.0, 200.0)},
            history,
            None,
            False,
            "affine",
            5.0,
            fps=25.0,
            unsafe_speed_kmh=6.0,
        )[0]

    current_speed = calculate([])["current_speed_kmh"]
    prior_speed = (30.0 * 6.004 - current_speed) / 29.0
    item = calculate([prior_speed] * 29)

    assert item["smoothed_speed_kmh"] == 6.0
    assert item["decision_speed_kmh"] == 6.0
    assert item["overspeed"] is False
    assert item["speed_status"] == "ok"


def test_incident_tracker_reports_each_track_once_for_the_whole_task() -> None:
    tracker = HazardIncidentTracker()
    track_1 = [{"code": "NO_HELMET", "_incident_key": ("NO_HELMET", "person", "1")}]
    track_2 = [{"code": "NO_HELMET", "_incident_key": ("NO_HELMET", "person", "2")}]
    assert len(tracker.update(track_1)) == 1
    assert tracker.update(track_1) == []
    assert len(tracker.update([*track_1, *track_2])) == 1
    assert tracker.update([]) == []
    assert tracker.update(track_1) == []


def test_incident_tracker_reset_starts_a_new_task_identity_scope() -> None:
    tracker = HazardIncidentTracker()
    incident = [{"code": "NO_HELMET", "_incident_key": ("NO_HELMET", "person", "1")}]
    assert len(tracker.update(incident)) == 1
    tracker.reset()
    assert len(tracker.update(incident)) == 1


def test_forklift_work_without_forklift_or_violation_is_not_a_hazard() -> None:
    result = build_forklift_work_output("frame.jpg", [], [], "circle")
    assert result["result"] == "unknown"
    assert result["violations"] == []


def test_ground_point_work_area_rejects_fence_only_and_invalid_boxes() -> None:
    image_shape = (720, 1280, 3)

    assert generate_ground_point_work_area(None, [], []) is None
    assert generate_ground_point_work_area(
        image_shape,
        [],
        [[10, 10, 100, 100]],
    ) is None
    assert generate_ground_point_work_area(
        image_shape,
        [[float("nan"), 10, 100, 100]],
        [[10, 10, 100, 100]],
    ) is None

    valid = generate_ground_point_work_area(
        image_shape,
        [[200, 200, 400, 500]],
        [],
    )
    assert valid is not None
    assert valid["type"] == "forklift_ground_point_fallback_circle"


def test_invalid_detection_boxes_are_filtered_and_invalid_polygons_are_ignored() -> None:
    assert is_valid_box([1, 1, 10, 10]) is True
    assert is_valid_box([1, 1, 1, 10]) is False
    assert is_valid_box([1, 1, float("nan"), 10]) is False

    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

        def tolist(self):
            return self.values.tolist()

    class FakeBoxes:
        def __init__(self) -> None:
            self.xyxy = FakeTensor([[1, 1, 10, 10], [4, 4, 4, 12], [1, 1, np.nan, 8]])
            self.conf = FakeTensor([0.9, 0.8, 0.7])
            self.cls = FakeTensor([0, 0, 0])
            self.id = FakeTensor([1, 2, 3])

        def __len__(self) -> int:
            return 3

        def cpu(self):
            return self

    detections = parse_detections(type("FakeResult", (), {"boxes": FakeBoxes()})())
    assert len(detections) == 1
    assert detections[0]["track_id"] == 1

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    draw_transparent_polygon(image, [[1, 1]])
    assert not image.any()


def test_forklift_work_visualization_only_draws_radius_area() -> None:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    radius_polygon = [[120, 560], [280, 560], [280, 720], [120, 720]]
    hidden_polygon = [[650, 650], [750, 650], [750, 750], [650, 750]]
    hidden_mask = np.zeros((800, 800), dtype=np.uint8)
    hidden_mask[650:751, 650:751] = 255
    forklift_results = [
        {
            "forklift_id": 1,
            "forklift_box": [350, 350, 450, 500],
            "work_zone": [300, 300, 500, 550],
            "forklift_ground_point": [400, 500],
            "circle_work_zone": {
                "enabled": True,
                "center": [200, 640],
                "radius_x_px": 80,
                "radius_y_px": 80,
                "polygon": radius_polygon,
            },
            "fence_based_work_zone": hidden_polygon,
            "road_aligned_work_zone": {
                "success": True,
                "polygon": hidden_polygon,
                "aligned_area_ratio": 1.0,
                "used_fence_clip": True,
                "used_road_mask_clip": True,
                "fence_clipped_work_zone": {"success": True},
            },
            "nearby_persons": [],
            "excluded_persons": [],
            "nearby_fences": [],
            "has_person_nearby": False,
            "has_fence_nearby": False,
            "violations": [],
            "is_safe": False,
        }
    ]
    ground_area = {
        "enabled": True,
        "polygon": hidden_polygon,
        "ground_points": {"forklifts": [], "fences": []},
    }
    road_area = {"mask": hidden_mask, "polygons": [hidden_polygon]}

    draw_forklift_work_results(
        image,
        forklift_results,
        [],
        [],
        ground_point_work_area=ground_area,
        road_surface_mask=hidden_mask,
        road_constrained_work_area=road_area,
        nearby_mode="circle",
        draw_circle_work_zone=True,
    )

    assert image[640, 200].any()
    assert not image[700, 700].any()


def test_five_distinct_tracked_violations_create_five_images(tmp_path: Path) -> None:
    frame_result = {
        "frame_id": 100,
        "source_type": "video",
        "result": "unsafe",
        "results": {
            "person_helmet": {
                "persons": [
                    {"track_id": 1, "status": "no_helmet", "box": [1, 1, 5, 10]},
                    {"track_id": 2, "status": "no_helmet", "box": [6, 1, 10, 10]},
                ]
            },
            "driver_helmet": {
                "drivers": [
                    {"track_id": 3, "forklift_id": 20, "status": "no_helmet", "box": [11, 1, 15, 10]}
                ]
            },
            "forklift_proximity": {
                "forklifts": [
                    {
                        "forklift_id": 21,
                        "has_person_nearby": True,
                        "nearby_persons": [{"track_id": 4, "person_id": 4, "box": [16, 1, 20, 10]}],
                    }
                ]
            },
            "forklift_speed": {
                "forklifts": [{"track_id": 22, "overspeed": True, "box": [21, 1, 30, 10]}]
            },
        },
    }
    tracker = HazardIncidentTracker()
    incidents = tracker.update(build_hazard_instances("safety_combined", frame_result))
    assert len(incidents) == 5
    assert attach_hazard_event(
        frame_result,
        np.zeros((32, 32, 3), dtype=np.uint8),
        tmp_path / "result.mp4",
        "safety_combined",
        incidents,
        True,
        1,
    )
    assert frame_result["hazard_count"] == 5
    assert len(frame_result["hazard_image_paths"]) == 1
    assert all(Path(path).is_file() for path in frame_result["hazard_image_paths"])
    assert len({item["hazard_image_path"] for item in frame_result["hazards"]}) == 1


def test_event_cursor_retains_multiple_incident_batches() -> None:
    frame_1 = {"frame_id": 1, "hazard_event": {"event_id": "event-1"}}
    frame_3 = {"frame_id": 3, "hazard_event": {"event_id": "event-2"}}
    job = {
        "_hazard_event_results": [frame_1, frame_3],
        "frame_result": frame_3,
        "result_frame_id": 3,
    }
    event, updated = select_new_hazard_event(job, -1)
    assert updated is True and event == frame_1
    event, updated = select_new_hazard_event(job, 1)
    assert updated is True and event == frame_3
    event, updated = select_new_hazard_event(job, 3)
    assert updated is False and event is None


def test_job_status_without_cursor_preserves_final_summary() -> None:
    first_event = {"frame_id": 1, "hazard_event": {"event_id": "event-1"}}
    latest_event = {"frame_id": 3, "hazard_event": {"event_id": "event-2"}}
    final_summary = {"result": "unsafe", "hazard_event_count": 2}
    stored_job = {
        "status": "done",
        "result": final_summary,
        "frame_result": latest_event,
        "_hazard_event_results": [first_event, latest_event],
        "latest_frame_bytes": b"preview",
    }

    payload = build_job_status_payload(stored_job)

    assert payload["result"] is final_summary
    assert payload["frame_result"] == latest_event
    assert "result_updated" not in payload
    assert "annotated_updated" not in payload
    assert "_hazard_event_results" not in payload
    assert "latest_frame_bytes" not in payload


def test_job_status_with_cursor_returns_incremental_event() -> None:
    first_event = {"frame_id": 1, "hazard_event": {"event_id": "event-1"}}
    latest_event = {"frame_id": 3, "hazard_event": {"event_id": "event-2"}}
    stored_job = {
        "status": "running",
        "frame_result": latest_event,
        "result_frame_id": 3,
        "_hazard_event_results": [first_event, latest_event],
        "latest_frame_url": "/api/file?path=evidence.jpg",
        "annotated_frame_id": 3,
    }

    payload = build_job_status_payload(
        stored_job,
        {"after_frame_id": ["-1"], "after_annotated_frame_id": ["-1"]},
    )

    assert payload["frame_result"] == first_event
    assert payload["result_updated"] is True
    assert payload["annotated_updated"] is True


def test_job_state_is_atomic_and_terminal_state_rejects_late_updates() -> None:
    safety_web_server_module.JOBS.clear()
    try:
        create_job("job-1", {"ok": True, "job_id": "job-1", "status": "running", "progress": 0})
        assert update_job("job-1", progress=10) is True
        assert update_job("job-1", status="done", progress=100) is True
        assert update_job("job-1", progress=50) is False
        assert update_job("job-1", status="error", error="late callback") is False

        stored = safety_web_server_module.JOBS["job-1"]
        assert stored["status"] == "done"
        assert stored["progress"] == 100
        payload = build_job_status_payload(stored)
        assert all(not key.startswith("_") for key in payload)
    finally:
        safety_web_server_module.JOBS.clear()


def test_hazard_event_buffer_is_bounded(monkeypatch) -> None:
    safety_web_server_module.JOBS.clear()
    monkeypatch.setattr(safety_web_server_module, "MAX_RETAINED_HAZARD_EVENTS", 3)
    try:
        create_job("job-events", {"job_id": "job-events", "status": "running"})
        for frame_id in range(1, 8):
            update_safety_job_preview(
                "job-events",
                {
                    "frame_result": {
                        "frame_id": frame_id,
                        "hazard_event": {"event_id": f"event-{frame_id}"},
                    }
                },
            )
        retained = safety_web_server_module.JOBS["job-events"]["_hazard_event_results"]
        assert [item["frame_id"] for item in retained] == [5, 6, 7]
    finally:
        safety_web_server_module.JOBS.clear()


def test_new_hazard_event_is_durably_journaled_once(tmp_path: Path) -> None:
    safety_web_server_module.JOBS.clear()
    journal = tmp_path / "safety.events.jsonl"
    try:
        create_job(
            "job-journal",
            {
                "job_id": "job-journal",
                "status": "running",
                "_event_journal_path": str(journal),
            },
        )
        frame_result = {"frame_id": 9, "hazard_event": {"event_id": "event-9", "hazard_count": 1}}
        update_safety_job_preview("job-journal", {"frame_result": frame_result})
        update_safety_job_preview("job-journal", {"frame_result": frame_result})

        lines = journal.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["hazard_event"]["event_id"] == "event-9"
    finally:
        safety_web_server_module.JOBS.clear()


def test_finished_job_pruning_keeps_running_jobs(monkeypatch) -> None:
    safety_web_server_module.JOBS.clear()
    monkeypatch.setattr(safety_web_server_module, "MAX_RETAINED_JOBS", 2)
    monkeypatch.setattr(safety_web_server_module, "JOB_RETENTION_SECONDS", 1000.0)
    try:
        create_job("running", {"job_id": "running", "status": "running"})
        for index in range(4):
            create_job(f"done-{index}", {"job_id": f"done-{index}", "status": "running"})
            update_job(f"done-{index}", status="done")
            safety_web_server_module.JOBS[f"done-{index}"]["_finished_monotonic"] = float(index)

        assert prune_finished_jobs(now=10.0) == 2
        assert "running" in safety_web_server_module.JOBS
        assert {job_id for job_id in safety_web_server_module.JOBS if job_id.startswith("done-")} == {
            "done-2",
            "done-3",
        }
    finally:
        safety_web_server_module.JOBS.clear()


def test_atomic_json_result_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"partial":', encoding="utf-8")
    write_json_atomic(target, {"ok": True, "items": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True, "items": [1, 2, 3]}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_video_writer_publishes_only_after_release(tmp_path: Path) -> None:
    temporary = tmp_path / ".result.inprogress.mp4"
    final = tmp_path / "result.mp4"
    temporary.write_bytes(b"complete-mp4")

    class Writer:
        def __init__(self) -> None:
            self.released = 0

        def isOpened(self) -> bool:
            return True

        def write(self, frame) -> None:
            return None

        def release(self) -> None:
            self.released += 1

    writer = Writer()
    atomic = mcc.AtomicVideoWriter(writer, temporary, final)
    assert not final.exists()
    atomic.release()
    atomic.release()

    assert final.read_bytes() == b"complete-mp4"
    assert not temporary.exists()
    assert writer.released == 1


def test_managed_job_supervisor_records_unhandled_worker_failure() -> None:
    safety_web_server_module.JOBS.clear()
    try:
        create_job("job-failure", {"job_id": "job-failure", "status": "running"})

        def fail() -> None:
            raise RuntimeError("simulated worker failure")

        thread = start_managed_job("job-failure", fail, (), uses_algorithm_slot=False)
        thread.join(timeout=2.0)

        job = safety_web_server_module.JOBS["job-failure"]
        assert job["status"] == "error"
        assert "simulated worker failure" in job["error"]
        assert "job-failure" not in safety_web_server_module.JOB_THREADS
    finally:
        safety_web_server_module.JOBS.clear()


def test_live_algorithm_job_waits_for_first_frame_before_starting_model(monkeypatch) -> None:
    order: list[str] = []
    released: list[object] = []
    packet = SimpleNamespace(
        frame_id=17,
        frame=np.zeros((8, 12, 3), dtype=np.uint8),
        fps=25.0,
        generation=2,
    )

    class ReadyReader:
        error = ""
        last_failure = ""
        startup_timeout_seconds = 30.0

        @staticmethod
        def wait_for_frame(_after_frame_id: int, timeout: float = 0.5):
            assert timeout == 0.5
            order.append("first-frame")
            return packet

    reader = ReadyReader()
    monkeypatch.setattr(live_stream_module, "acquire_shared_live_reader", lambda _url: reader)
    monkeypatch.setattr(
        live_stream_module,
        "release_shared_live_reader",
        lambda _url, value: released.append(value),
    )
    monkeypatch.setattr(safety_web_server_module, "encode_preview", lambda _frame: b"jpeg")
    monkeypatch.setattr(
        safety_web_server_module,
        "_monitor_live_job",
        lambda _job_id, _url, _stop: order.append("health-monitor"),
    )
    create_job(
        "job-direct-live",
        {
            "job_id": "job-direct-live",
            "status": "running",
            "_live_stream_url": "http://example.invalid/index/api/whep?app=live&stream=test",
        },
    )

    def run_algorithm() -> None:
        order.append("model-and-algorithm")
        update_job("job-direct-live", status="done")

    try:
        thread = start_managed_job("job-direct-live", run_algorithm, (), uses_algorithm_slot=False)
        thread.join(timeout=2.0)

        job = safety_web_server_module.JOBS["job-direct-live"]
        assert job["status"] == "done"
        assert job["stream_ready"] is True
        assert job["raw_frame_id"] == 17
        assert job["latest_raw_frame_bytes"] == b"jpeg"
        assert order.index("first-frame") < order.index("model-and-algorithm")
        assert released == [reader]
    finally:
        safety_web_server_module.JOBS.pop("job-direct-live", None)


def test_live_job_monitor_cancels_stalled_stream_to_release_algorithm_slot(monkeypatch) -> None:
    class StalledReader:
        recovering = True
        connected = False
        opened_once = True
        startup_timeout_seconds = 30.0
        error = "test stream disconnected"
        last_failure = error

        @staticmethod
        def health_snapshot() -> dict[str, object]:
            return {"last_frame_age_seconds": 10.0}

    reader = StalledReader()
    released: list[object] = []
    monkeypatch.setattr(live_stream_module, "acquire_shared_live_reader", lambda _url: reader)
    monkeypatch.setattr(
        live_stream_module,
        "release_shared_live_reader",
        lambda _url, value: released.append(value),
    )
    monkeypatch.setattr(safety_web_server_module, "LIVE_STREAM_STALL_TIMEOUT_SECONDS", 0.01)
    create_job("job-stalled-stream", {"status": "running", "progress_text": "running"})
    try:
        safety_web_server_module._monitor_live_job(
            "job-stalled-stream",
            "http://example.invalid/index/api/whep?app=live&stream=test",
            safety_web_server_module.threading.Event(),
        )
        job = safety_web_server_module.JOBS["job-stalled-stream"]
        assert job["cancel_requested"] is True
        assert "释放算法资源" in job["progress_text"]
        assert released == [reader]
    finally:
        safety_web_server_module.JOBS.pop("job-stalled-stream", None)


def test_web_server_disallows_multiple_processes_on_same_port() -> None:
    assert safety_web_server_module.SafetyHTTPServer.allow_reuse_address is False


def test_hazard_frame_is_saved_with_timestamp_and_safe_frame_is_not(tmp_path: Path) -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    output_hint = tmp_path / "result.mp4"
    safe_result = {
        "frame_id": 1,
        "source_type": "video",
        "result": "safe",
        "persons": [],
    }
    assert attach_hazard_event(safe_result, frame, output_hint, "helmet", [], True, 1) is False
    assert list(tmp_path.glob("*.jpg")) == []

    unsafe_result = {
        "frame_id": 2,
        "source_type": "video",
        "result": "unsafe",
        "persons": [{"person_id": 7, "status": "no_helmet", "box": [1, 2, 10, 20]}],
    }
    incidents = HazardIncidentTracker().update(build_hazard_instances("helmet", unsafe_result))
    assert attach_hazard_event(unsafe_result, frame, output_hint, "helmet", incidents, True, 1) is True
    saved = list(tmp_path.glob("result_helmet_NO_HELMET_hazard_*_frame_2.jpg"))
    assert len(saved) == 1
    assert unsafe_result["hazard_event"]["hazard_image_path"] == str(saved[0].resolve())
    assert unsafe_result["hazards"][0]["code"] == "NO_HELMET"


def test_web_preview_keeps_display_status_separate_from_hazard_event(tmp_path: Path) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    evidence = tmp_path / "hazard_20260720_120000_000001.jpg"
    evidence.write_bytes(b"evidence")
    status = {
        "frame_id": 9,
        "result": "unsafe",
        "has_hazard": True,
        "hazard_event": {
            "frame_id": 9,
            "hazard_timestamp": "2026-07-20T12:00:00.000001+08:00",
            "hazard_image_path": str(evidence),
            "hazard_image_paths": [str(evidence)],
        },
        "hazards": [{"code": "NO_HELMET"}],
    }

    updates = build_safety_preview_updates("job-1", 9, frame, status, "helmet")
    assert updates["frame_status"]["result"] == "unsafe"
    assert "hazard_event" not in updates["frame_status"]
    assert updates["frame_result"]["hazard_event"]["frame_id"] == 9
    assert updates["result_frame_id"] == 9
    assert updates["annotated_stream_url"] == "/api/jobs/job-1/stream/annotated"
    assert updates["annotated_frame_id"] == 9
    assert updates["annotated_image_path"] == str(evidence.resolve())
    assert updates["latest_frame_url"].startswith("/api/file?path=")

    event, updated = select_new_hazard_event(updates, after_frame_id=-1)
    assert updated is True
    assert event is updates["frame_result"]
    event, updated = select_new_hazard_event(updates, after_frame_id=9)
    assert updated is False
    assert event is None
    assert is_annotated_image_updated(updates, after_annotated_frame_id=-1) is True
    assert is_annotated_image_updated(updates, after_annotated_frame_id=9) is False

    repeated_status = {"frame_id": 10, "result": "unsafe", "has_hazard": True}
    repeated_updates = build_safety_preview_updates(
        "job-1", 10, np.full((8, 8, 3), 255, dtype=np.uint8), repeated_status, "helmet"
    )
    assert "annotated_frame_id" not in repeated_updates
    assert "latest_frame_url" not in repeated_updates
    assert "frame_result" not in repeated_updates

    safe_updates = build_safety_preview_updates(
        "job-2", 11, frame, {"frame_id": 11, "result": "safe", "has_hazard": False}, "helmet"
    )
    assert "latest_frame_url" not in safe_updates
    assert "annotated_frame_id" not in safe_updates


def test_video_monitor_reports_each_new_track_once_but_displays_all_frames(
    tmp_path: Path, monkeypatch
) -> None:
    frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(3)]

    class FakeCapture:
        def __init__(self, _source: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == svm.cv2.CAP_PROP_FPS:
                return 25.0
            if prop == svm.cv2.CAP_PROP_FRAME_COUNT:
                return float(len(frames))
            return 0.0

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index].copy()
            self.index += 1
            return True, frame

        def release(self) -> None:
            pass

    def fake_analyze(_model, frame, frame_id, *_args, **_kwargs):
        track_id = 1 if frame_id < 3 else 2
        return frame.copy(), {
            "frame_id": frame_id,
            "result": "unsafe",
            "total_persons": 1,
            "no_helmet_count": 1,
            "persons": [{"person_id": track_id, "track_id": track_id, "status": "no_helmet", "box": [1, 1, 8, 10]}],
        }

    monkeypatch.setattr(svm.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(svm, "load_model", lambda _path: object())
    monkeypatch.setattr(svm, "resolve_device", lambda device: device)
    monkeypatch.setattr(svm, "analyze_helmet_frame", fake_analyze)

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_hint = tmp_path / "result.mp4"
    displayed = []
    result = svm.run_safety_monitor(
        mode="helmet",
        source_type="video",
        source_path=source,
        weights_path="unused.pt",
        output_media=output_hint,
        preview_callback=lambda _frame_id, _frame, status: displayed.append(dict(status)),
    )

    assert len(displayed) == 3
    assert ["hazard_event" in status for status in displayed] == [True, False, True]
    assert all(status["has_hazard"] for status in displayed)
    assert result["hazard_event_count"] == 2
    assert result["hazard_incident_count"] == 2
    assert result["hazard_event"]["frame_id"] == 3
    assert len(list(tmp_path.glob("result_helmet_NO_HELMET_hazard_*.jpg"))) == 2
    assert output_hint.exists() is False


def test_video_monitor_does_not_save_same_track_again_after_missed_frames(
    tmp_path: Path, monkeypatch
) -> None:
    frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(4)]

    class FakeCapture:
        def __init__(self, _source: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == svm.cv2.CAP_PROP_FPS:
                return 25.0
            if prop == svm.cv2.CAP_PROP_FRAME_COUNT:
                return float(len(frames))
            return 0.0

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index].copy()
            self.index += 1
            return True, frame

        def release(self) -> None:
            pass

    def fake_analyze(_model, frame, frame_id, *_args, **_kwargs):
        if frame_id == 2:
            return frame.copy(), {
                "frame_id": frame_id, "result": "unknown", "total_persons": 0,
                "no_helmet_count": 0, "persons": [],
            }
        return frame.copy(), {
            "frame_id": frame_id,
            "result": "unsafe",
            "total_persons": 1,
            "no_helmet_count": 1,
            "persons": [{
                "person_id": 7, "track_id": 7, "status": "no_helmet", "box": [1, 1, 8, 10]
            }],
        }

    monkeypatch.setattr(svm.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(svm, "load_model", lambda _path: object())
    monkeypatch.setattr(svm, "resolve_device", lambda device: device)
    monkeypatch.setattr(svm, "analyze_helmet_frame", fake_analyze)

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_hint = tmp_path / "result.mp4"
    displayed = []
    result = svm.run_safety_monitor(
        mode="helmet",
        source_type="video",
        source_path=source,
        weights_path="unused.pt",
        output_media=output_hint,
        preview_callback=lambda _frame_id, _frame, status: displayed.append(dict(status)),
    )

    assert ["hazard_event" in status for status in displayed] == [True, False, False, False]
    assert result["hazard_event_count"] == 1
    assert result["hazard_incident_count"] == 1
    assert len(list(tmp_path.glob("result_helmet_NO_HELMET_hazard_*.jpg"))) == 1
