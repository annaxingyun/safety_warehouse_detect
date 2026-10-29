from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
import warnings
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import INFERENCE_IMAGE_SIZE, NMS_IOU, load_model, model_class_names, resolve_device, save_json, yolo_detection_logging  # noqa: E402
from live_stream import acquire_shared_live_reader, release_shared_live_reader  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


BASE_TRACKER_YAML = PROJECT_ROOT / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"
WEB_TRACKER_YAML = PROJECT_ROOT / "runs" / "material_count_web" / "botsort_web.yaml"
# OBB 模型输出旋转框，普通检测模型输出轴对齐框；两者都可用于计数。
SUPPORTED_COUNT_TASKS = {"obb", "detect"}
EXPORT_VIDEO_FOURCC = "mp4v"
EXPORT_VIDEO_CODEC = "libx264"
EXPORT_VIDEO_CRF = 26
EXPORT_VIDEO_PRESET = "medium"
EXPORT_IMAGE_SEQUENCE_FPS = 10.0
VIDEO_FALLBACK_FPS = 30.0
PREVIEW_MAX_WIDTH = 1920
PREVIEW_MAX_HEIGHT = 1080
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/arial.ttf"),
)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class TrackerArgs:
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    track_buffer: int = 50
    match_thresh: float = 0.7
    with_reid: bool = True
    proximity_thresh: float = 0.6
    appearance_thresh: float = 0.85

    def __post_init__(self) -> None:
        for name in ("proximity_thresh", "appearance_thresh"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} 必须在 0 到 1 之间。")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if is_image_file(p)], key=lambda p: p.name)


def track_color(track_id: int) -> tuple[int, int, int]:
    return ((37 * track_id) % 255, (17 * track_id + 91) % 255, (29 * track_id + 53) % 255)


def load_text_font(size: int) -> ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=max(10, int(size)))
    return ImageFont.load_default()


def draw_unicode_text(frame: np.ndarray, text: str, origin: tuple[int, int], font_size: int, color: tuple[int, int, int]) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_text_font(font_size)
    draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def unicode_text_size(text: str, font_size: int) -> tuple[int, int]:
    font = load_text_font(font_size)
    bbox = font.getbbox(text)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


class PyAVH264Writer:
    """Small cv2.VideoWriter-compatible H.264 writer with controllable quality."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        crf: int = EXPORT_VIDEO_CRF,
        preset: str = EXPORT_VIDEO_PRESET,
    ) -> None:
        import av

        self._av = av
        self._closed = False
        self._width = width + (width % 2)
        self._height = height + (height % 2)
        rate = Fraction(max(1.0, float(fps))).limit_denominator(1000)
        self._container = av.open(output_path, mode="w", format="mp4", options={"movflags": "+faststart"})
        try:
            self._stream = self._container.add_stream(EXPORT_VIDEO_CODEC, rate=rate)
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": str(max(0, min(51, int(crf)))), "preset": str(preset)}
        except Exception:
            self._container.close()
            self._closed = True
            raise

    def isOpened(self) -> bool:
        return not self._closed

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("video writer is already closed")
        video_frame = self._av.VideoFrame.from_ndarray(frame, format="bgr24")
        if video_frame.width != self._width or video_frame.height != self._height:
            video_frame = video_frame.reformat(width=self._width, height=self._height, format="bgr24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def release(self) -> None:
        if self._closed:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._closed = True


class OpenCVVideoWriter:
    """Expose cv2's writer through the same small interface as the PyAV writer."""

    def __init__(self, writer: cv2.VideoWriter) -> None:
        self._writer = writer

    def isOpened(self) -> bool:
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


class AtomicVideoWriter:
    """Publish a video only after its encoder has flushed a valid trailer."""

    def __init__(self, writer: Any, temporary_path: Path, output_path: Path) -> None:
        self._writer = writer
        self._temporary_path = temporary_path
        self._output_path = output_path
        self._closed = False

    def isOpened(self) -> bool:
        return not self._closed and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("video writer is already closed")
        self._writer.write(frame)

    def release(self) -> None:
        if self._closed:
            return
        try:
            self._writer.release()
            if not self._temporary_path.is_file() or self._temporary_path.stat().st_size <= 0:
                raise RuntimeError(f"视频编码器未生成有效文件: {self._temporary_path}")
            os.replace(self._temporary_path, self._output_path)
        finally:
            self._closed = True


def create_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = EXPORT_VIDEO_CRF,
    preset: str = EXPORT_VIDEO_PRESET,
) -> Any:
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = final_path.suffix.lower()
    temporary_path = final_path.with_name(
        f".{final_path.stem}.{uuid.uuid4().hex}.inprogress{final_path.suffix}"
    )
    if suffix == ".mp4":
        try:
            writer = PyAVH264Writer(str(temporary_path), fps, width, height, crf=crf, preset=preset)
            return AtomicVideoWriter(writer, temporary_path, final_path)
        except (ImportError, ModuleNotFoundError) as exc:
            warnings.warn(f"PyAV 不可用，视频将回退到 OpenCV 编码器: {exc}", RuntimeWarning, stacklevel=2)
        except Exception as exc:
            warnings.warn(f"libx264 初始化失败，视频将回退到 OpenCV 编码器: {exc}", RuntimeWarning, stacklevel=2)
        temporary_path.unlink(missing_ok=True)
    fourcc_candidates = ("XVID",) if suffix == ".avi" else ("avc1", "H264", EXPORT_VIDEO_FOURCC)
    for fourcc_name in fourcc_candidates:
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*fourcc_name), max(1.0, fps), (width, height)
        )
        if writer.isOpened():
            return AtomicVideoWriter(OpenCVVideoWriter(writer), temporary_path, final_path)
        writer.release()
        temporary_path.unlink(missing_ok=True)
    raise RuntimeError(f"无法创建视频文件: {final_path}")


def make_preview_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(PREVIEW_MAX_WIDTH / max(1, w), PREVIEW_MAX_HEIGHT / max(1, h), 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def draw_labeled_box(frame: np.ndarray, xyxy: tuple[float, float, float, float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    line_thickness = max(2, int(round(min(frame.shape[:2]) / 360)))
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness, lineType=cv2.LINE_AA)
    draw_detection_label(frame, label, (x1, y1), color)


def draw_detection_label(
    frame: np.ndarray,
    label: str,
    anchor: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    x1, y1 = anchor
    font_size = max(16, int(round(max(0.5, min(frame.shape[:2]) / 1440) * 30)))
    tw, th = unicode_text_size(label, font_size)
    baseline = max(4, font_size // 5)
    label_x1 = max(0, min(int(x1), frame.shape[1] - 1))
    label_y1 = max(0, int(y1) - th - baseline - 8)
    label_y2 = max(th + baseline + 8, int(y1))
    label_x2 = min(frame.shape[1] - 1, label_x1 + tw + 8)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    draw_unicode_text(frame, label, (label_x1 + 4, label_y1 + 3), font_size, (255, 255, 255))


def draw_labeled_detection(
    frame: np.ndarray,
    points: list[list[float]],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw one OBB polygon or one axis-aligned box (as a 4-corner polygon)."""
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return
    line_thickness = max(2, int(round(min(frame.shape[:2]) / 360)))
    cv2.polylines(
        frame,
        [np.rint(polygon).astype(np.int32)],
        isClosed=True,
        color=color,
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )
    anchor = (float(np.min(polygon[:, 0])), float(np.min(polygon[:, 1])))
    draw_detection_label(frame, label, anchor, color)


def box_corners(box: list[float]) -> list[list[float]]:
    """Turn an xyxy box into the same 4-corner layout OBB results use."""
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _build_detections(
    points_rows: list,
    box_rows: list,
    confidences: list,
    class_ids: list,
    track_ids: list,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for points, box, confidence, class_id, track_id in zip(
        points_rows, box_rows, confidences, class_ids, track_ids
    ):
        try:
            normalized_points = [[float(point[0]), float(point[1])] for point in points]
            normalized_box = [float(value) for value in box]
            normalized_confidence = float(confidence)
            normalized_class_id = int(class_id)
            normalized_track_id = None if track_id is None else int(track_id)
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        values = [value for point in normalized_points for value in point] + normalized_box + [normalized_confidence]
        if len(normalized_points) != 4 or len(normalized_box) != 4 or not all(np.isfinite(values)):
            continue
        detections.append(
            {
                "class_id": normalized_class_id,
                "confidence": normalized_confidence,
                "track_id": normalized_track_id,
                "box": normalized_box,
                "points": normalized_points,
            }
        )
    return detections


def parse_material_detections(result: Any) -> list[dict[str, Any]]:
    """Normalize Ultralytics results for counting; accepts OBB and plain detect models."""
    if result is None:
        return []
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        obb = obb.cpu()
        class_ids = obb.cls.tolist()
        return _build_detections(
            obb.xyxyxyxy.tolist(),
            obb.xyxy.tolist(),
            obb.conf.tolist(),
            class_ids,
            obb.id.tolist() if getattr(obb, "id", None) is not None else [None] * len(class_ids),
        )
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = boxes.cpu()
    class_ids = boxes.cls.tolist()
    box_rows = [[float(value) for value in row] for row in boxes.xyxy.tolist()]
    return _build_detections(
        [box_corners(row) for row in box_rows],
        box_rows,
        boxes.conf.tolist(),
        class_ids,
        boxes.id.tolist() if getattr(boxes, "id", None) is not None else [None] * len(class_ids),
    )


def draw_info_panel(frame: np.ndarray, frame_idx: int, total_count: int, class_counts: dict[str, int]) -> np.ndarray:
    fh, fw = frame.shape[:2]
    font_size = max(14, int(round(max(0.45, min(fh, fw) / 1400) * 28)))
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"Frame : {frame_idx}", (0, 255, 0)),
        (f"Total : {total_count}", (0, 255, 0)),
    ]
    for cls_name, count in sorted(class_counts.items()):
        lines.append((f"  {cls_name}: {count}", (200, 200, 200)))
    pad = 6
    char_h = unicode_text_size("A", font_size)[1]
    line_h = int(char_h * 1.9)
    panel_h = line_h * len(lines) + pad * 2
    panel_w = max(180, max(unicode_text_size(text, font_size)[0] for text, _ in lines) + pad * 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, (text, color) in enumerate(lines):
        draw_unicode_text(frame, text, (pad, pad + i * line_h), font_size, color)
    return frame


def write_tracker_yaml(args: TrackerArgs) -> Path:
    if not BASE_TRACKER_YAML.is_file():
        raise FileNotFoundError(f"跟踪器配置不存在: {BASE_TRACKER_YAML}")
    data = YAML.load(BASE_TRACKER_YAML)
    data.update(
        {
            "track_high_thresh": float(args.track_high_thresh),
            "track_low_thresh": float(args.track_low_thresh),
            "new_track_thresh": float(args.new_track_thresh),
            "track_buffer": int(args.track_buffer),
            "match_thresh": float(args.match_thresh),
            "with_reid": bool(args.with_reid),
            "proximity_thresh": float(args.proximity_thresh),
            "appearance_thresh": float(args.appearance_thresh),
        }
    )
    WEB_TRACKER_YAML.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(WEB_TRACKER_YAML, data)
    return WEB_TRACKER_YAML


class MaterialCounter:
    def __init__(
        self,
        model_path: str,
        conf: float,
        tracker_args: TrackerArgs,
        device: str,
        stop_flag: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        preview_callback: Optional[Callable[[int, Any, dict[str, Any]], None]] = None,
        raw_preview_callback: Optional[Callable[[int, Any, dict[str, Any]], None]] = None,
        sample_interval: int = 5,
        target_classes: Optional[list[str]] = None,
        video_crf: int = EXPORT_VIDEO_CRF,
        video_preset: str = EXPORT_VIDEO_PRESET,
        detection_log_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.model_path = model_path
        self.conf = conf
        self.tracker_args = tracker_args
        self.device = device
        self.stop_flag = stop_flag
        self.progress_callback = progress_callback
        self.preview_callback = preview_callback
        self.raw_preview_callback = raw_preview_callback
        self.detection_log_callback = detection_log_callback
        self.sample_interval = max(1, int(sample_interval))
        self.requested_target_classes = target_classes
        self.video_crf = max(0, min(51, int(video_crf)))
        self.video_preset = str(video_preset)
        self.target_classes: list[str] = []
        self.target_class_ids: list[int] = []
        self.stopped_by_user = False
        self.class_counts: defaultdict[str, int] = defaultdict(int)
        self.counted_track_ids: set[tuple[int, int, int]] = set()
        self.counted_track_ordinals: dict[tuple[int, int, int], int] = {}
        self.track_generation = 0
        self.total_count = 0
        self.last_media_path = ""
        self.model_task = ""

    def run(
        self,
        source_type: str,
        source_path: Path | str,
        output_media: Optional[Path],
    ) -> dict[str, Any]:
        self.source_type = source_type
        self.source_name = "live_stream" if source_type == "live_stream" else (source_path.name if source_path.is_dir() else source_path.stem)
        detector = load_model(self.model_path)
        self.model_task = str(getattr(detector, "task", "") or "").lower()
        if self.model_task not in SUPPORTED_COUNT_TASKS:
            raise ValueError(
                f"物资计数模型必须是 OBB 或目标检测模型，当前模型 task={self.model_task or 'unknown'}。"
            )
        class_names = model_class_names(detector)
        self.target_class_ids, self.target_classes = resolve_target_classes(self.requested_target_classes, class_names)
        self.class_counts = defaultdict(int, {name: 0 for name in self.target_classes})
        if source_type == "single_image":
            assert isinstance(source_path, Path)
            self._run_single_image(detector, class_names, source_path, output_media)
        elif source_type == "image_sequence":
            assert isinstance(source_path, Path)
            self._run_image_sequence(detector, class_names, source_path, output_media)
        elif source_type == "live_stream":
            self._run_live_stream(detector, class_names, str(source_path), output_media)
        else:
            assert isinstance(source_path, Path)
            self._run_video(detector, class_names, source_path, output_media)
        return {
            "name": self.source_name,
            "source_type": source_type,
            "model_task": self.model_task,
            "sample_interval": self.sample_interval,
            "target_classes": list(self.target_classes),
            "total": self.total_count,
            "classes": dict(sorted(self.class_counts.items(), key=lambda x: -x[1])),
            "output_media_path": self.last_media_path,
            "stopped_by_user": bool(self.stopped_by_user),
            "finished_at": current_timestamp(),
        }

    def _should_stop(self) -> bool:
        if self.stop_flag is not None and self.stop_flag():
            self.stopped_by_user = True
            return True
        return False

    def _report_progress(self, processed: int, total: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(processed, total)

    def _report_preview(self, frame_idx: int, frame: np.ndarray, total: int) -> None:
        if self.preview_callback is not None:
            self.preview_callback(
                frame_idx,
                frame,
                {
                    "frame_id": frame_idx,
                    "name": getattr(self, "source_name", ""),
                    "source_type": getattr(self, "source_type", ""),
                    "model_task": self.model_task,
                    "sample_interval": self.sample_interval,
                    "target_classes": list(self.target_classes),
                    "total_frames": total,
                    "total": self.total_count,
                    "total_count": self.total_count,
                    "classes": dict(sorted(self.class_counts.items(), key=lambda x: -x[1])),
                    "stopped_by_user": bool(self.stopped_by_user),
                },
            )

    def _run_single_image(self, detector: YOLO, class_names: dict[int, str], image_path: Path, output_image: Optional[Path]) -> None:
        if not image_path.is_file() or not is_image_file(image_path):
            raise ValueError("单张图片模式下，输入必须是图片文件。")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"无法读取图片: {image_path}")
        painted, _ = self._process_single_image(frame, detector, class_names)
        painted = draw_info_panel(painted, 1, self.total_count, dict(self.class_counts))
        self._report_preview(1, painted.copy(), 1)
        self._report_progress(1, 1)
        if output_image is not None:
            output_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image), painted)
            self.last_media_path = str(output_image)

    def _run_image_sequence(self, detector: YOLO, class_names: dict[int, str], source_dir: Path, output_video: Optional[Path]) -> None:
        if not source_dir.is_dir():
            raise ValueError("图片序列模式下，输入必须是文件夹。")
        image_files = get_image_files(source_dir)
        if not image_files:
            raise ValueError("输入文件夹中没有可用图片。")
        tracker_yaml = write_tracker_yaml(self.tracker_args)
        writer: Optional[Any] = None
        writer_size: Optional[Tuple[int, int]] = None
        try:
            latest: Optional[np.ndarray] = None
            for frame_idx, image_path in enumerate(image_files, start=1):
                if self._should_stop():
                    break
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                painted, _ = self._process_frame(frame, frame_idx, detector, class_names, tracker_yaml)
                painted = draw_info_panel(painted, frame_idx, self.total_count, dict(self.class_counts))
                latest = painted
                self._report_preview(frame_idx, painted.copy(), len(image_files))
                self._report_progress(frame_idx, len(image_files))
                if output_video is not None:
                    export_frame = make_preview_frame(painted)
                    h, w = export_frame.shape[:2]
                    if writer is None:
                        writer = create_video_writer(
                            str(output_video), EXPORT_IMAGE_SEQUENCE_FPS, w, h, self.video_crf, self.video_preset
                        )
                        writer_size = (w, h)
                    elif writer_size != (w, h):
                        export_frame = cv2.resize(export_frame, writer_size, interpolation=cv2.INTER_AREA)
                    writer.write(export_frame)
            if output_video is not None and writer is not None:
                self.last_media_path = str(output_video)
            elif latest is not None:
                fallback = output_video.with_suffix(".jpg") if output_video is not None else source_dir / "material_count_result.jpg"
                cv2.imwrite(str(fallback), latest)
                self.last_media_path = str(fallback)
        finally:
            if writer is not None:
                writer.release()

    def _run_video(self, detector: YOLO, class_names: dict[int, str], video_path: Path, output_video: Optional[Path]) -> None:
        if not video_path.is_file():
            raise ValueError("视频模式下，输入必须是视频文件。")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        tracker_yaml = write_tracker_yaml(self.tracker_args)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or VIDEO_FALLBACK_FPS)
        writer: Optional[Any] = None
        writer_size: Optional[Tuple[int, int]] = None
        latest: Optional[np.ndarray] = None
        try:
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            while True:
                if self._should_stop():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % self.sample_interval != 0:
                    continue
                painted, _ = self._process_frame(frame, frame_idx, detector, class_names, tracker_yaml)
                painted = draw_info_panel(painted, frame_idx, self.total_count, dict(self.class_counts))
                latest = painted
                self._report_preview(frame_idx, painted.copy(), total_frames)
                self._report_progress(frame_idx, total_frames)
                if output_video is not None:
                    export_frame = make_preview_frame(painted)
                    h, w = export_frame.shape[:2]
                    if writer is None:
                        writer = create_video_writer(
                            str(output_video), max(1.0, fps / self.sample_interval), w, h,
                            self.video_crf, self.video_preset,
                        )
                        writer_size = (w, h)
                    elif writer_size != (w, h):
                        export_frame = cv2.resize(export_frame, writer_size, interpolation=cv2.INTER_AREA)
                    writer.write(export_frame)
            if output_video is not None and writer is not None:
                self.last_media_path = str(output_video)
            elif latest is not None:
                fallback = output_video.with_suffix(".jpg") if output_video is not None else video_path.with_name(f"{video_path.stem}_material_count.jpg")
                cv2.imwrite(str(fallback), latest)
                self.last_media_path = str(fallback)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    def _run_live_stream(
        self,
        detector: YOLO,
        class_names: dict[int, str],
        stream_url: str,
        output_video: Optional[Path],
    ) -> None:
        tracker_yaml = write_tracker_yaml(self.tracker_args)
        writer: Optional[Any] = None
        writer_size: Optional[Tuple[int, int]] = None
        last_seen_id = -1
        last_processed_id: Optional[int] = None
        live_generation: Optional[int] = None
        started_at = time.monotonic()

        def on_raw(packet) -> None:
            if self.raw_preview_callback is not None:
                self.raw_preview_callback(packet.frame_id, packet.frame, {"fps": packet.fps, "generation": packet.generation})

        reader = acquire_shared_live_reader(stream_url, frame_callback=on_raw)
        try:
            while not self._should_stop():
                packet = reader.wait_for_frame(last_seen_id, timeout=1.0)
                if packet is None:
                    if (
                        not reader.opened_once
                        and time.monotonic() - started_at > reader.startup_timeout_seconds
                    ):
                        raise RuntimeError(reader.last_failure or reader.error or "直播流连接超时。")
                    continue
                last_seen_id = packet.frame_id
                if live_generation is not None and packet.generation != live_generation:
                    predictor = getattr(detector, "predictor", None)
                    for tracker in getattr(predictor, "trackers", []) or []:
                        reset = getattr(tracker, "reset", None)
                        if callable(reset):
                            reset()
                    self.track_generation += 1
                    last_processed_id = None
                live_generation = packet.generation
                if last_processed_id is not None and packet.frame_id - last_processed_id < self.sample_interval:
                    continue
                painted, _ = self._process_frame(packet.frame, packet.frame_id, detector, class_names, tracker_yaml)
                painted = draw_info_panel(painted, packet.frame_id, self.total_count, dict(self.class_counts))
                self._report_preview(packet.frame_id, painted.copy(), 0)
                self._report_progress(packet.frame_id, 0)
                last_processed_id = packet.frame_id
                if output_video is not None:
                    export_frame = make_preview_frame(painted)
                    h, w = export_frame.shape[:2]
                    if writer is None:
                        writer = create_video_writer(
                            str(output_video), max(1.0, packet.fps / self.sample_interval), w, h,
                            self.video_crf, self.video_preset,
                        )
                        writer_size = (w, h)
                    elif writer_size != (w, h):
                        export_frame = cv2.resize(export_frame, writer_size, interpolation=cv2.INTER_AREA)
                    writer.write(export_frame)
            if output_video is not None and writer is not None:
                self.last_media_path = str(output_video)
        finally:
            release_shared_live_reader(stream_url, reader, frame_callback=on_raw)
            if writer is not None:
                writer.release()

    def _process_frame(self, frame: np.ndarray, frame_idx: int, detector: YOLO, class_names: dict[int, str], tracker_yaml: Path) -> tuple[np.ndarray, int]:
        with yolo_detection_logging(detector, frame_idx, self.detection_log_callback):
            results = detector.track(
                source=frame,
                imgsz=INFERENCE_IMAGE_SIZE,
                tracker=str(tracker_yaml),
                conf=self.conf,
                iou=NMS_IOU,
                end2end=False,
                agnostic_nms=False,
                device=self.device,
                persist=True,
                verbose=False,
                stream=False,
                save=False,
                classes=self.target_class_ids,
            )
        result = results[0] if results else None
        detections = parse_material_detections(result)
        active_tracks = 0
        for detection in detections:
            cls_id = int(detection["class_id"])
            track_id = detection.get("track_id")
            if cls_id not in class_names or cls_id not in self.target_class_ids or track_id is None:
                continue
            active_tracks += 1
            tid = int(track_id)
            pair = (self.track_generation, cls_id, tid)
            if pair not in self.counted_track_ids:
                self.counted_track_ids.add(pair)
                cls_name = class_names[cls_id]
                self.class_counts[cls_name] += 1
                self.total_count += 1
                self.counted_track_ordinals[pair] = self.class_counts[cls_name]
            ordinal = self.counted_track_ordinals.get(pair, 0)
            draw_labeled_detection(
                frame,
                detection["points"],
                f"{class_names[cls_id]} {ordinal}",
                track_color(tid),
            )
        if frame_idx % 30 == 0:
            print(f"[material-count] frame={frame_idx} tracks={active_tracks} count={self.total_count}", flush=True)
        return frame, active_tracks

    def _process_single_image(self, frame: np.ndarray, detector: YOLO, class_names: dict[int, str]) -> tuple[np.ndarray, int]:
        with yolo_detection_logging(detector, 1, self.detection_log_callback):
            results = detector.predict(
                source=frame,
                imgsz=INFERENCE_IMAGE_SIZE,
                conf=self.conf,
                iou=NMS_IOU,
                end2end=False,
                agnostic_nms=False,
                device=self.device,
                verbose=False,
                stream=False,
                save=False,
                classes=self.target_class_ids,
            )
        result = results[0] if results else None
        detections = parse_material_detections(result)
        detected_count = 0
        for det_idx, detection in enumerate(detections, start=1):
            cls_id = int(detection["class_id"])
            if cls_id not in class_names or cls_id not in self.target_class_ids:
                continue
            detected_count += 1
            cls_name = class_names[cls_id]
            self.class_counts[cls_name] += 1
            self.total_count += 1
            draw_labeled_detection(
                frame,
                detection["points"],
                f"{cls_name} {self.class_counts[cls_name]}",
                track_color(det_idx + cls_id * 97),
            )
        return frame, detected_count


def parse_target_classes(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_name in re.split(r"[\n,，]+", text):
        name = raw_name.strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def resolve_target_classes(requested: Optional[list[str]], class_names: dict[int, str]) -> tuple[list[int], list[str]]:
    if not class_names:
        raise ValueError("模型没有可用的类别名称。")
    if requested is None:
        class_ids = list(class_names)
        return class_ids, [class_names[class_id] for class_id in class_ids]
    if not requested:
        raise ValueError("请至少选择一个需要计数的物资类别。")

    by_name = {name.casefold(): (class_id, name) for class_id, name in class_names.items()}
    selected_ids: list[int] = []
    selected_names: list[str] = []
    unknown: list[str] = []
    for requested_name in requested:
        match = by_name.get(requested_name.casefold())
        if match is None:
            unknown.append(requested_name)
            continue
        class_id, canonical_name = match
        if class_id not in selected_ids:
            selected_ids.append(class_id)
            selected_names.append(canonical_name)
    if unknown:
        raise ValueError(f"模型不包含类别：{', '.join(unknown)}。可用类别：{', '.join(class_names.values())}")
    return selected_ids, selected_names


def normalize_expected_counts(expected: dict[str, int], target_classes: list[str]) -> dict[str, int]:
    target_by_name = {name.casefold(): name for name in target_classes}
    normalized: dict[str, int] = {}
    unknown: list[str] = []
    for name, count in expected.items():
        canonical_name = target_by_name.get(name.casefold())
        if canonical_name is None:
            unknown.append(name)
        else:
            normalized[canonical_name] = count
    if unknown:
        raise ValueError(f"实际数量中包含未勾选的类别：{', '.join(unknown)}")
    return normalized


def parse_expected_counts(text: str) -> dict[str, int]:
    expected: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "," in line:
            name, count = line.split(",", 1)
        elif "，" in line:
            name, count = line.split("，", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"实际数量格式错误: {line}")
            name, count = parts
        name = name.strip()
        count = count.strip()
        if not name:
            raise ValueError("类别名称不能为空")
        if not count.isdigit():
            raise ValueError(f"{name} 的实际数量必须是非负整数")
        expected[name] = int(count)
    return expected


def format_accuracy(detected: int, actual: Optional[int]) -> str:
    if actual is None:
        return "-"
    if actual == 0:
        return "100.00%" if detected == 0 else "无法计算"
    return f"{max(0.0, 1.0 - abs(detected - actual) / actual) * 100:.2f}%"


def build_accuracy_rows(classes: dict[str, int], expected: dict[str, int]) -> list[dict[str, Any]]:
    names = sorted(set(classes) | set(expected), key=lambda n: (n not in expected, -int(classes.get(n, 0)), n))
    rows = []
    for name in names:
        detected = int(classes.get(name, 0))
        actual = expected.get(name)
        rows.append({"class_name": name, "detected": detected, "actual": actual, "accuracy": format_accuracy(detected, actual)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Material counting CLI used by the web UI.")
    parser.add_argument("--source-type", choices=["single_image", "image_sequence", "video", "live_stream"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-media")
    parser.add_argument("--json", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--track-high-thresh", type=float, default=0.3)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.4)
    parser.add_argument("--track-buffer", type=int, default=50)
    parser.add_argument("--match-thresh", type=float, default=0.7)
    parser.add_argument("--with-reid", choices=["true", "false"], default="true")
    parser.add_argument("--proximity-thresh", type=float, default=0.6)
    parser.add_argument("--appearance-thresh", type=float, default=0.85)
    parser.add_argument("--target-classes", default="", help="逗号或换行分隔的目标类别；留空时兼容旧行为并检测全部类别")
    parser.add_argument("--expected-counts", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--video-crf", type=int, default=EXPORT_VIDEO_CRF, help="H.264 CRF quality (0-51; lower is larger/better)")
    parser.add_argument("--video-preset", default=EXPORT_VIDEO_PRESET, help="libx264 encoding preset")
    args = parser.parse_args()

    source = args.source if args.source_type == "live_stream" else Path(args.source)
    if args.source_type != "live_stream" and not source.exists():
        raise FileNotFoundError(f"输入路径不存在: {source}")
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    args.device = resolve_device(args.device)
    output_media = Path(args.output_media) if args.output_media else None
    tracker_args = TrackerArgs(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        with_reid=args.with_reid == "true",
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
    )
    requested_targets = parse_target_classes(args.target_classes) if args.target_classes.strip() else None
    counter = MaterialCounter(
        args.model,
        args.conf,
        tracker_args,
        args.device,
        sample_interval=args.sample_interval,
        target_classes=requested_targets,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
    )
    result = counter.run(args.source_type, source, output_media)
    expected = normalize_expected_counts(parse_expected_counts(args.expected_counts), result["target_classes"])
    result["expected_counts"] = expected
    result["accuracy_rows"] = build_accuracy_rows(result["classes"], expected)
    result.setdefault("finished_at", current_timestamp())
    result["created_at"] = result["finished_at"]
    json_path = Path(args.json)
    save_json(result, json_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
