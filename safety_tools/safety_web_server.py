from __future__ import annotations

import cgi
import copy
import csv
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from model_config import DEFAULT_MODEL_NAME, INFERENCE_IMAGE_SIZE, NMS_IOU

WEB_DIR = APP_DIR / "web"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "safety_web"
UPLOAD_DIR = DEFAULT_OUTPUT_DIR / "uploads"
DEFAULT_WEIGHTS = PROJECT_DIR / DEFAULT_MODEL_NAME
# 网页只保留物资计数，默认改用纯物资模型。
DEFAULT_MATERIAL_MODEL = PROJECT_DIR / "yolo26s-best.pt"
MATERIAL_CLASS_CANDIDATES = (
    "small cable",
    "large cable",
    "large distribution box",
    "rectangular transformer",
    "triangular transformer",
    "small rectangular transformer",
    "medium distribution box",
    "second medium distribution box",
    "small distribution box",
    "nano distribution box",
)
# 物资计数既支持 OBB 旋转框模型，也支持普通目标检测模型。
MATERIAL_SUPPORTED_TASKS = {"obb", "detect"}


def _env_number(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_THREADS: dict[str, threading.Thread] = {}
JOB_THREADS_LOCK = threading.Lock()
TERMINAL_JOB_STATES = {"done", "error", "canceled"}
MAX_RETAINED_HAZARD_EVENTS = int(_env_number("SAFETY_MAX_RETAINED_EVENTS", 256, 10))
MAX_RETAINED_DETECTION_LOGS = 200
MAX_RETAINED_JOBS = int(_env_number("SAFETY_MAX_RETAINED_JOBS", 100, 10))
JOB_RETENTION_SECONDS = _env_number("SAFETY_JOB_RETENTION_SECONDS", 21600, 300)
MAX_CONCURRENT_ALGORITHM_JOBS = int(_env_number("SAFETY_MAX_ALGORITHM_JOBS", 1, 1))
LIVE_STREAM_STALL_TIMEOUT_SECONDS = _env_number("SAFETY_LIVE_STREAM_STALL_TIMEOUT", 60, 5)
ALGORITHM_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_ALGORITHM_JOBS)
RUNTIME_STOP_EVENT = threading.Event()
SAFETY_CHECK_LABELS = {
    "person_helmet": "人员安全帽检测",
    "driver_helmet": "叉车司机安全帽检测",
    "forklift_proximity": "叉车作业半径站人检测",
    "forklift_fence": "叉车作业围蔽检测",
    "forklift_speed": "叉车速度检测",
    "crane_under_boom": "吊臂下方人员检测",
    "lifting_operation": "吊装作业围栏及安全监护检测",
}
MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript", ".ts", ".pb", ".tflite", ".xml"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".ts"}
GPU_INFO_CACHE: Optional[dict[str, Any]] = None


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_job(job_id: str, payload: dict[str, Any]) -> None:
    """Create a consistently timestamped in-memory job record."""
    now = time.monotonic()
    with JOBS_LOCK:
        JOBS[job_id] = {
            **payload,
            "_created_monotonic": now,
            "_updated_monotonic": now,
        }


def append_job_detection_log(job_id: str, frame_id: int, message: str) -> None:
    """Retain bounded per-frame logs with a cursor independent of hazard events."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.get("status") in TERMINAL_JOB_STATES:
            return
        entries = job.setdefault("_detection_logs", deque(maxlen=MAX_RETAINED_DETECTION_LOGS))
        log_id = int(job.get("_last_detection_log_id", 0)) + 1
        entries.append({"log_id": log_id, "frame_id": frame_id, "timestamp": current_timestamp(), "message": message})
        job["_last_detection_log_id"] = log_id
        job["_updated_monotonic"] = time.monotonic()


def update_job(job_id: str, **updates: Any) -> bool:
    """Apply one atomic job transition and protect terminal states from late callbacks."""
    clean_updates = {key: value for key, value in updates.items() if value is not None}
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return False
        current_status = str(job.get("status", ""))
        requested_status = clean_updates.get("status")
        if current_status in TERMINAL_JOB_STATES and requested_status != current_status:
            return False
        now = time.monotonic()
        job.update(clean_updates)
        job["_updated_monotonic"] = now
        if requested_status in TERMINAL_JOB_STATES:
            job.setdefault("_finished_monotonic", now)
            job.setdefault("finished_at", current_timestamp())
            job.pop("_live_stream_url", None)
        return True


def write_json_atomic(path: Path, payload: Any) -> None:
    """Durably replace a JSON result so readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Replace a binary preview without exposing a half-written image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file:
            file.write(payload)
            file.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl_durable(path: Path, payload: Any) -> None:
    """Append one safety event and flush it before acknowledging the event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.flush()
        os.fsync(file.fileno())


def prune_finished_jobs(now: Optional[float] = None) -> int:
    """Bound retained job memory while never deleting running work."""
    current = time.monotonic() if now is None else float(now)
    with JOBS_LOCK:
        terminal = [
            (job_id, job)
            for job_id, job in JOBS.items()
            if job.get("status") in TERMINAL_JOB_STATES
        ]
        expired = {
            job_id
            for job_id, job in terminal
            if current - float(job.get("_finished_monotonic", current)) >= JOB_RETENTION_SECONDS
        }
        survivors = sorted(
            ((job_id, job) for job_id, job in terminal if job_id not in expired),
            key=lambda item: float(item[1].get("_finished_monotonic", current)),
            reverse=True,
        )
        expired.update(job_id for job_id, _ in survivors[MAX_RETAINED_JOBS:])
        for job_id in expired:
            JOBS.pop(job_id, None)
    return len(expired)


def runtime_health_payload() -> dict[str, Any]:
    """Build a lightweight service health response without leaking stream URLs."""
    from live_stream import shared_live_reader_health

    with JOBS_LOCK:
        statuses = Counter(str(job.get("status", "unknown")) for job in JOBS.values())
        persistence_error_jobs = sum(bool(job.get("_persistence_error")) for job in JOBS.values())
    with JOB_THREADS_LOCK:
        worker_count = sum(thread.is_alive() for thread in JOB_THREADS.values())
    live_streams = shared_live_reader_health()
    stopping = RUNTIME_STOP_EVENT.is_set()
    degraded = persistence_error_jobs > 0 or int(live_streams["recovering_count"]) > 0
    return {
        "ok": not stopping,
        "ready": not stopping and not degraded,
        "status": "stopping" if stopping else ("degraded" if degraded else "running"),
        "jobs": dict(statuses),
        "active_worker_threads": worker_count,
        "persistence_error_jobs": persistence_error_jobs,
        "max_concurrent_algorithm_jobs": MAX_CONCURRENT_ALGORITHM_JOBS,
        "live_streams": live_streams,
    }


def _wait_for_live_job_first_frame(job_id: str, reader: Any) -> Any:
    """Make a live algorithm job own its cold-start handshake before model loading.

    The optional preview endpoint and the algorithm endpoint must be equivalent:
    either one may be the first consumer which creates the shared WHEP session.
    Waiting here also prevents model initialization from competing with SDP / ICE
    negotiation and keyframe decoding on resource-constrained machines.
    """
    started_at = time.monotonic()
    last_progress_text = ""
    update_job(
        job_id,
        progress_text="算法任务正在直接连接 WebRTC 直播流并等待首帧",
        stream_ready=False,
    )
    while not RUNTIME_STOP_EVENT.is_set() and not is_job_cancel_requested(job_id):
        packet = reader.wait_for_frame(-1, timeout=0.5)
        if hasattr(reader, "health_snapshot"):
            update_job(job_id, stream_diagnostics=reader.health_snapshot())
        if packet is not None:
            updates: dict[str, Any] = {
                "raw_frame_id": packet.frame_id,
                "raw_frame_status": {
                    "fps": packet.fps,
                    "generation": packet.generation,
                },
                "progress_text": "直播流首帧已就绪，正在加载模型并启动检测",
                "stream_ready": True,
            }
            try:
                updates["latest_raw_frame_bytes"] = encode_preview(packet.frame)
            except Exception:
                # A preview encoding failure must not invalidate a decoded frame
                # which is already usable by the algorithm.
                pass
            update_job(job_id, **updates)
            return packet

        progress_text = str(reader.error or reader.last_failure or "Python 后端正在初始化 WHEP 连接。")
        if progress_text and progress_text != last_progress_text:
            last_progress_text = progress_text
            update_job(job_id, progress_text=progress_text, stream_ready=False)
        if time.monotonic() - started_at >= float(reader.startup_timeout_seconds):
            detail = str(reader.last_failure or reader.error or "直播流连接超时。").strip()
            raise RuntimeError(detail or "直播流连接超时。")

    raise RuntimeError("任务在等待直播首帧时被停止")


def _managed_job_runner(
    job_id: str,
    target: Any,
    args: tuple[Any, ...],
    uses_algorithm_slot: bool,
) -> None:
    acquired = False
    startup_reader: Any = None
    health_stop = threading.Event()
    health_thread: Optional[threading.Thread] = None
    try:
        with JOBS_LOCK:
            live_stream_url = str(JOBS.get(job_id, {}).get("_live_stream_url", ""))
        if uses_algorithm_slot:
            update_job(job_id, progress_text="任务已进入执行队列，正在等待算法资源")
            while not RUNTIME_STOP_EVENT.is_set() and not is_job_cancel_requested(job_id):
                if ALGORITHM_JOB_SLOTS.acquire(timeout=0.25):
                    acquired = True
                    break
            if not acquired:
                update_job(job_id, status="canceled", error="任务在等待算法资源时被停止")
                return
            update_job(job_id, progress_text="算法资源已分配，正在启动任务")
        # A queued job must not keep an extra shared-reader reference alive.
        # Once the job owns its slot, complete the WHEP cold start before model
        # loading. This makes direct /start calls follow the same reliable path
        # as an optional preview test without requiring that test to be called.
        if live_stream_url:
            from live_stream import acquire_shared_live_reader

            startup_reader = acquire_shared_live_reader(live_stream_url)
            _wait_for_live_job_first_frame(job_id, startup_reader)
            health_thread = threading.Thread(
                target=_monitor_live_job,
                args=(job_id, live_stream_url, health_stop),
                name=f"live-health-{job_id[:8]}",
                daemon=True,
            )
            health_thread.start()
        target(*args)
        with JOBS_LOCK:
            still_running = JOBS.get(job_id, {}).get("status") == "running"
        if still_running:
            update_job(job_id, status="error", error="任务线程异常结束，未生成完成状态")
    except BaseException as exc:
        if is_job_cancel_requested(job_id) or RUNTIME_STOP_EVENT.is_set():
            update_job(job_id, status="canceled", error="任务已停止")
        else:
            update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        health_stop.set()
        if health_thread is not None:
            health_thread.join(timeout=2.0)
        if startup_reader is not None:
            try:
                from live_stream import release_shared_live_reader

                release_shared_live_reader(live_stream_url, startup_reader)
            except Exception:
                pass
        if acquired:
            ALGORITHM_JOB_SLOTS.release()
        with JOB_THREADS_LOCK:
            JOB_THREADS.pop(job_id, None)


def _monitor_live_job(job_id: str, stream_url: str, stop_event: threading.Event) -> None:
    """Mirror stream health and cancel a stalled job so its algorithm slot is released."""
    from live_stream import acquire_shared_live_reader, release_shared_live_reader

    try:
        reader = acquire_shared_live_reader(stream_url)
    except Exception as exc:
        update_job(job_id, progress_text=f"无法启动直播健康监测: {exc}", stream_ready=False)
        return
    previous_state = ""
    unavailable_since: Optional[float] = time.monotonic()
    try:
        while not stop_event.wait(0.5):
            if is_job_cancel_requested(job_id):
                return
            if hasattr(reader, "health_snapshot"):
                update_job(job_id, stream_diagnostics=reader.health_snapshot())
            if reader.recovering:
                state = "recovering"
            elif reader.connected:
                state = "connected"
            else:
                state = "connecting"
            now = time.monotonic()
            if state == "connected":
                unavailable_since = None
            else:
                if unavailable_since is None:
                    unavailable_since = now
                unavailable_seconds = now - unavailable_since
                timeout_seconds = LIVE_STREAM_STALL_TIMEOUT_SECONDS
                if not reader.opened_once:
                    timeout_seconds = max(timeout_seconds, float(reader.startup_timeout_seconds))
                if unavailable_seconds >= timeout_seconds:
                    reason = (
                        f"直播连续 {unavailable_seconds:.0f} 秒无可用画面，"
                        "任务已自动停止并释放算法资源"
                    )
                    detail = str(reader.error or reader.last_failure or "").strip()
                    update_job(
                        job_id,
                        cancel_requested=True,
                        progress_text=reason,
                        error=f"{reason}: {detail}" if detail else reason,
                    )
                    return
            if state == previous_state:
                continue
            if state == "recovering":
                update_job(
                    job_id,
                    progress_text=reader.error or "直播流暂时中断，正在自动恢复",
                    stream_ready=False,
                )
            elif state == "connected" and previous_state == "recovering":
                update_job(job_id, progress_text="直播流已恢复，监测任务继续运行", stream_ready=True)
            previous_state = state
    finally:
        try:
            release_shared_live_reader(stream_url, reader)
        except Exception:
            pass


def start_managed_job(
    job_id: str,
    target: Any,
    args: tuple[Any, ...],
    *,
    uses_algorithm_slot: bool = True,
) -> threading.Thread:
    """Start and register a supervised worker thread."""
    if RUNTIME_STOP_EVENT.is_set():
        update_job(job_id, status="error", error="服务正在停止，不能创建新任务。")
        raise RuntimeError("服务正在停止，不能创建新任务。")
    thread = threading.Thread(
        target=_managed_job_runner,
        args=(job_id, target, args, uses_algorithm_slot),
        name=f"safety-job-{job_id[:8]}",
        daemon=True,
    )
    with JOB_THREADS_LOCK:
        JOB_THREADS[job_id] = thread
    try:
        thread.start()
    except Exception as exc:
        with JOB_THREADS_LOCK:
            JOB_THREADS.pop(job_id, None)
        update_job(job_id, status="error", error=f"无法启动任务线程: {exc}")
        raise
    return thread


def job_janitor_loop(interval_seconds: float = 60.0) -> None:
    """Periodically remove old terminal jobs during long-running service use."""
    while not RUNTIME_STOP_EVENT.wait(max(5.0, float(interval_seconds))):
        prune_finished_jobs()


def shutdown_runtime(join_timeout: float = 8.0) -> None:
    """Cooperatively stop jobs and WebRTC readers before process exit."""
    RUNTIME_STOP_EVENT.set()
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("status") not in TERMINAL_JOB_STATES:
                job["cancel_requested"] = True
                job["progress_text"] = "服务正在停止，任务正在安全退出"
    from live_stream import stop_all_shared_live_readers

    stop_all_shared_live_readers(join_timeout=min(3.0, max(0.0, join_timeout)))
    deadline = time.monotonic() + max(0.0, float(join_timeout))
    while True:
        with JOB_THREADS_LOCK:
            threads = [thread for thread in JOB_THREADS.values() if thread.is_alive()]
        if not threads or time.monotonic() >= deadline:
            break
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=min(0.25, remaining))


def detect_gpu_info() -> dict[str, Any]:
    global GPU_INFO_CACHE
    if GPU_INFO_CACHE is not None:
        return dict(GPU_INFO_CACHE)

    info: dict[str, Any] = {"has_gpu": False, "gpu_count": 0, "error": ""}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            info["has_gpu"] = info["gpu_count"] > 0
    except Exception as exc:
        info["error"] = str(exc)
    GPU_INFO_CACHE = dict(info)
    return info


def default_device() -> str:
    return "0" if detect_gpu_info().get("has_gpu") else "cpu"


def resolve_device(requested: Optional[str]) -> str:
    raw = (requested or "").strip()
    gpu_info = detect_gpu_info()
    if not raw or raw.lower() in {"auto", "default"}:
        return default_device()
    normalized = raw.lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "0" if gpu_info.get("has_gpu") else "cpu"
    if normalized.startswith("cuda:"):
        return normalized.split(":", 1)[1] if gpu_info.get("has_gpu") else "cpu"
    if not gpu_info.get("has_gpu"):
        return "cpu"
    return raw


MODEL_INFO_CACHE: dict[str, dict[str, Any]] = {}
MODEL_INFO_LOCK = threading.Lock()
MODEL_INFO_DISK_LOCK = threading.Lock()
MODEL_INFO_CACHE_FILE = DEFAULT_OUTPUT_DIR / ".model_info_cache.json"
MODEL_INFO_DISK_LIMIT = 50


def resolve_model_path(model_text: str) -> Path:
    """Resolve a model path the same way the algorithm subprocesses do."""
    path = Path(str(model_text or "").strip())
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _load_model_cache_file() -> dict[str, Any]:
    try:
        data = json.loads(MODEL_INFO_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _remember_model_info(key: str, info: dict[str, Any]) -> None:
    """Cache in memory, and persist successful lookups so restarts stay fast."""
    with MODEL_INFO_LOCK:
        MODEL_INFO_CACHE[key] = info
    if not info.get("ok"):
        return
    with MODEL_INFO_DISK_LOCK:
        data = _load_model_cache_file()
        data.pop(key, None)
        data[key] = info
        while len(data) > MODEL_INFO_DISK_LIMIT:
            data.pop(next(iter(data)))
        try:
            ensure_dir(MODEL_INFO_CACHE_FILE.parent)
            MODEL_INFO_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def describe_model(model_text: str) -> dict[str, Any]:
    """Read a checkpoint's task and class names without keeping it loaded."""
    raw = str(model_text or "").strip()
    empty = {"ok": False, "path": raw, "task": "", "names": [], "error": "模型路径为空"}
    if not raw:
        return empty
    path = resolve_model_path(raw)
    try:
        stat = path.stat()
    except OSError:
        return {**empty, "error": f"模型文件不存在: {raw}"}
    key = f"{path.resolve()}|{int(stat.st_mtime)}|{int(stat.st_size)}"
    with MODEL_INFO_LOCK:
        cached = MODEL_INFO_CACHE.get(key)
    if cached is None:
        with MODEL_INFO_DISK_LOCK:
            cached = _load_model_cache_file().get(key)
        if cached is not None:
            with MODEL_INFO_LOCK:
                MODEL_INFO_CACHE[key] = cached
    if cached is not None:
        return dict(cached)
    info = read_model_info(path)
    _remember_model_info(key, info)
    return dict(info)


def read_model_info(path: Path) -> dict[str, Any]:
    """Run the inspection in a subprocess so the server never imports PyTorch."""
    rc, stdout, stderr = run_command([sys.executable, str(APP_DIR / "model_info.py"), str(path)])
    for line in reversed(stdout.strip().splitlines()):
        try:
            info = json.loads(line)
        except ValueError:
            continue
        if isinstance(info, dict) and "names" in info:
            return info
    detail = (stderr or stdout or "无法读取模型").strip()
    return {
        "ok": False,
        "path": str(path),
        "task": "",
        "names": [],
        "error": detail[-400:],
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_live_stream_url(value: str) -> str:
    """Reject accidental log/text input before it reaches aiortc or OpenCV."""
    stream_url = str(value).strip()
    if not stream_url:
        raise ValueError("直播地址不能为空。")
    if any(character in stream_url for character in "\r\n"):
        raise ValueError("直播地址格式无效：地址中不能包含换行或运行日志。")
    parsed = urlparse(stream_url)
    allowed_schemes = {"http", "https", "whep+http", "whep+https"}
    if parsed.scheme.lower() not in allowed_schemes or not parsed.netloc:
        raise ValueError(
            "直播地址格式无效；当前直播输入只支持 WebRTC/WHEP，请输入完整 URL，例如："
            "http://服务器/index/api/whep?app=live&stream=流名称"
        )
    return stream_url


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _helmet_status_value(person: dict[str, Any]) -> Optional[str]:
    if person.get("is_wearing_helmet") is True:
        return "helmet"
    if person.get("is_wearing_helmet") is False:
        return "no_helmet"
    status = person.get("helmet_status", person.get("status"))
    return str(status) if status in {"helmet", "no_helmet", "unknown"} else None


def _person_key(person: dict[str, Any]) -> Any:
    person_id = person.get("person_id")
    if person_id is not None:
        return ("id", person_id)
    box = person.get("box")
    if isinstance(box, list):
        return ("box", tuple(box))
    return None


def _collect_forklift_persons(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    persons: list[dict[str, Any]] = []
    for forklift in _as_list(data.get("forklifts")):
        if isinstance(forklift, dict):
            persons.extend([p for p in _as_list(forklift.get(key)) if isinstance(p, dict)])
    persons.extend([p for p in _as_list(data.get(key)) if isinstance(p, dict)])
    return persons


def count_unique_nearby_persons(data: dict[str, Any]) -> int:
    excluded = {_person_key(p) for p in _collect_forklift_persons(data, "excluded_persons")}
    excluded.discard(None)
    nearby = set()
    for person in _collect_forklift_persons(data, "nearby_persons"):
        key = _person_key(person)
        if key is not None and key not in excluded:
            nearby.add(key)
    return len(nearby)


def calculate_monitor_accuracy(expected_count: int, detected_count: int) -> float:
    expected = max(0, int(expected_count))
    detected = max(0, int(detected_count))
    denominator = max(expected, detected, 1)
    accuracy = (1.0 - abs(detected - expected) / denominator) * 100.0
    return max(0.0, min(100.0, accuracy))


def format_accuracy_statistics(expected_count: int, detected_count: int) -> str:
    accuracy = calculate_monitor_accuracy(expected_count, detected_count)
    return f"人工输入隐患数量：{expected_count}\n检测隐患数量：{detected_count}\n监测准确度：{accuracy:.2f}%"


def get_helmet_hazard_count(data: dict[str, Any]) -> int:
    if data.get("no_helmet_count") is not None:
        return _to_int(data.get("no_helmet_count"))
    return sum(1 for p in _as_list(data.get("persons")) if isinstance(p, dict) and _helmet_status_value(p) == "no_helmet")


def get_forklift_work_hazard_count(data: dict[str, Any]) -> int:
    violations = set(_as_list(data.get("violations")))
    if not violations:
        violations = {
            violation
            for forklift in _as_list(data.get("forklifts"))
            if isinstance(forklift, dict)
            for violation in _as_list(forklift.get("violations"))
        }
    nearby_count = count_unique_nearby_persons(data) if "PERSON_NEAR_FORKLIFT" in violations else 0
    no_fence_count = 1 if "NO_FENCE_NEAR_FORKLIFT" in violations else 0
    return nearby_count + no_fence_count + get_helmet_hazard_count(data)


def get_speed_hazard_count(data: dict[str, Any]) -> int:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    if _to_int(summary.get("overspeed_frame_count")) > 0:
        return 1
    return 1 if str(summary.get("result", "")).lower() == "unsafe" else 0


def format_helmet_statistics(data: dict[str, Any]) -> str:
    persons = _as_list(data.get("persons"))
    total = data.get("total_persons", len(persons))
    helmet_count = data.get("helmet_worn_count")
    no_helmet_count = data.get("no_helmet_count")
    unknown_count = data.get("unknown_count")
    if helmet_count is None:
        helmet_count = sum(1 for p in persons if isinstance(p, dict) and _helmet_status_value(p) == "helmet")
    if no_helmet_count is None:
        no_helmet_count = sum(1 for p in persons if isinstance(p, dict) and _helmet_status_value(p) == "no_helmet")
    if unknown_count is None:
        unknown_count = sum(1 for p in persons if isinstance(p, dict) and _helmet_status_value(p) == "unknown")
    return f"图中共 {total} 人，共 {helmet_count} 人佩戴安全帽，共 {no_helmet_count} 人未佩戴安全帽，{unknown_count} 人状态未知。"


def format_helmet_result(data: dict[str, Any]) -> str:
    lines = [
        f"总人数：{data.get('total_persons', 0)}",
        f"佩戴安全帽：{data.get('helmet_worn_count', 0)}",
        f"未佩戴安全帽：{data.get('no_helmet_count', 0)}",
        f"未知：{data.get('unknown_count', 0)}",
        "",
    ]
    for person in _as_list(data.get("persons")):
        if isinstance(person, dict):
            lines.append(f"人员 {person.get('person_id', '?')}: {person.get('status', 'unknown')} {person.get('box', [])}")
    return "\n".join(lines).strip()


def format_forklift_work_statistics(data: dict[str, Any]) -> str:
    person_count = data.get("person_count", data.get("total_persons"))
    if person_count is None:
        person_count = len(_as_list(data.get("persons")))
    forklifts = _as_list(data.get("forklifts"))
    forklift_count = data.get("forklift_count", len(forklifts))
    helmet_count = data.get("helmet_worn_count")
    if helmet_count is None:
        helmet_count = sum(1 for p in _as_list(data.get("persons")) if isinstance(p, dict) and _helmet_status_value(p) == "helmet")
    nearby_count = count_unique_nearby_persons(data)
    has_fence = bool(data.get("has_fence", data.get("has_fence_nearby", False)))
    return f"图中共 {person_count} 人，共 {helmet_count} 人佩戴安全帽，共 {forklift_count} 辆叉车，其中 {nearby_count} 人在作业半径内\n是否检测到围栏：{'是' if has_fence else '否'}"


def format_forklift_work_result(data: dict[str, Any]) -> str:
    lines = [
        f"作业结果：{data.get('result', 'unknown')}",
        f"nearby_mode：{data.get('nearby_mode', 'unknown')}",
        f"叉车数量：{data.get('forklift_count', len(_as_list(data.get('forklifts'))))}",
        f"人员数量：{data.get('total_persons', data.get('person_count', 0))}",
        f"违章：{', '.join(data.get('violations', [])) if data.get('violations') else '无'}",
    ]
    for forklift in _as_list(data.get("forklifts")):
        if isinstance(forklift, dict):
            lines.append(f"叉车 {forklift.get('forklift_id', '?')}: {forklift.get('message', forklift.get('is_safe', 'unknown'))}")
    return "\n".join(lines)


def format_speed_result(data: dict[str, Any]) -> str:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    return "\n".join(
        [
            f"标注限速：{summary.get('speed_limit_kmh', data.get('speed_limit_kmh', 5.0))} km/h",
            f"总帧数：{summary.get('total_frames', 0)}",
            f"超速帧数：{summary.get('overspeed_frame_count', 0)}",
            f"超速 track_id：{summary.get('overspeed_track_ids', [])}",
            f"最大估算速度：{summary.get('max_speed_kmh', None)} km/h",
            f"报警阈值：{summary.get('unsafe_speed_kmh', 6.0)} km/h（超过才判定不安全）",
            f"结果：{summary.get('result', 'unknown')}",
        ]
    )


def field_values(form: cgi.FieldStorage, name: str) -> list[str]:
    """Return all non-file values submitted for a multipart field."""
    if name not in form:
        return []
    item = form[name]
    items = item if isinstance(item, list) else [item]
    return [str(value.value) for value in items if not getattr(value, "filename", None) and value.value is not None]


def field_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    """Return the first text value for a form field, ignoring uploaded files."""
    values = field_values(form, name)
    return values[0] if values else default


def bool_field(form: cgi.FieldStorage, name: str, default: bool = False) -> bool:
    value = field_value(form, name, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "on"}


def upload_items(form: cgi.FieldStorage, field: str) -> list[cgi.FieldStorage]:
    if field not in form:
        return []
    item = form[field]
    items = item if isinstance(item, list) else [item]
    return [x for x in items if getattr(x, "filename", None)]


def save_upload(form: cgi.FieldStorage, field: str, fallback_path: str, prefix: str) -> Path:
    items = upload_items(form, field)
    if items:
        item = items[0]
        ensure_dir(UPLOAD_DIR)
        suffix = Path(item.filename).suffix or ".bin"
        path = UPLOAD_DIR / f"{prefix}_{int(time.time() * 1000)}{suffix}"
        data = item.file.read()
        path.write_bytes(data)
        return path
    fallback = fallback_path.strip()
    if not fallback:
        raise FileNotFoundError(f"未选择输入文件: {field}")
    path = Path(fallback)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    return path


def safe_upload_relative_path(filename: str) -> Path:
    raw = Path(filename.replace("\\", "/"))
    parts = [part for part in raw.parts if part not in {"", ".", "/", "\\"}]
    if not parts or any(part == ".." for part in parts):
        return Path(raw.name or f"upload_{uuid.uuid4().hex}")
    return Path(*parts)


def save_upload_directory(form: cgi.FieldStorage, field: str, fallback_path: str, prefix: str) -> Path:
    items = upload_items(form, field)
    if not items:
        fallback = fallback_path.strip()
        if not fallback:
            raise FileNotFoundError(f"未选择输入文件夹: {field}")
        path = Path(fallback)
        if not path.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {path}")
        return path

    base = ensure_dir(UPLOAD_DIR / f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")
    rel_paths: list[Path] = []
    for item in items:
        rel = safe_upload_relative_path(item.filename)
        rel_paths.append(rel)
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.file.read())

    first_parts = [rel.parts[0] for rel in rel_paths if len(rel.parts) > 1]
    if first_parts and len(first_parts) == len(rel_paths) and len(set(first_parts)) == 1:
        return base / first_parts[0]
    return base


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def run_command(command: list[str]) -> tuple[int, str, str]:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_DIR) if not existing_path else f"{PROJECT_DIR}{os.pathsep}{existing_path}"
    process = subprocess.run(command, cwd=str(PROJECT_DIR), text=True, capture_output=True, env=env)
    return process.returncode, process.stdout, process.stderr


def file_url(path: Path) -> str:
    return f"/api/file?path={quote(str(path.resolve()), safe='')}"


def select_path_dialog(kind: str, initial: str = "") -> dict[str, Any]:
    initial_path = Path(initial.strip()) if initial.strip() else PROJECT_DIR
    if initial_path.is_file():
        initial_path = initial_path.parent
    if not initial_path.exists():
        initial_path = PROJECT_DIR
    script = (
        "import json, pathlib, sys, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "kind, initial = sys.argv[1], sys.argv[2]\n"
        "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        "try:\n"
        "    if kind == 'dir':\n"
        "        selected = filedialog.askdirectory(initialdir=initial, title='选择输出目录')\n"
        "    else:\n"
        "        selected = filedialog.askopenfilename(initialdir=initial, title='选择文件', filetypes=[('Model files', '*.pt *.onnx *.engine *.torchscript *.ts *.pb *.tflite *.xml'), ('All files', '*.*')])\n"
        "finally:\n"
        "    root.destroy()\n"
        "print(json.dumps({'ok': bool(selected), 'path': str(pathlib.Path(selected).resolve()) if selected else '', 'cancelled': not bool(selected)}, ensure_ascii=False))\n"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", script, kind, str(initial_path)],
            text=True,
            capture_output=True,
            timeout=300,
        )
        if process.returncode != 0:
            return {"ok": False, "error": process.stderr.strip() or "目录选择对话框打开失败"}
        payload = json.loads(process.stdout.strip() or "{}")
        if not payload.get("ok"):
            return {"ok": False, "cancelled": True, "error": "已取消选择"}
        return {"ok": True, "path": str(Path(payload["path"]).resolve())}
    except Exception as exc:
        return {"ok": False, "error": f"无法打开系统选择对话框: {exc}"}


def drive_roots() -> list[str]:
    if os.name != "nt":
        return ["/"]
    roots = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if Path(root).exists():
            roots.append(root)
    return roots


def browse_path(kind: str, raw_path: str) -> dict[str, Any]:
    raw = raw_path.strip()
    target = Path(raw) if raw else PROJECT_DIR
    if raw and not target.is_absolute():
        target = PROJECT_DIR / target
    if not target.exists():
        target = target.parent if target.parent.exists() else PROJECT_DIR
    if target.is_file():
        target = target.parent
    target = target.resolve()

    dirs: list[dict[str, str]] = []
    files: list[dict[str, Any]] = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return {"ok": False, "error": f"无法读取目录: {exc}", "current": str(target)}
    for item in children:
        try:
            if item.is_dir():
                dirs.append({"name": item.name, "path": str(item.resolve())})
            elif kind == "file" and item.suffix.lower() in MODEL_EXTENSIONS:
                files.append({"name": item.name, "path": str(item.resolve()), "size": item.stat().st_size})
            elif kind == "video" and item.suffix.lower() in VIDEO_EXTENSIONS:
                files.append({"name": item.name, "path": str(item.resolve()), "size": item.stat().st_size})
        except OSError:
            continue

    parent = str(target.parent.resolve()) if target.parent != target else ""
    return {
        "ok": True,
        "kind": kind,
        "current": str(target),
        "parent": parent,
        "drives": drive_roots(),
        "dirs": dirs,
        "files": files,
    }


MOCK_STREAM_HOST = "127.0.0.1"
MOCK_STREAM_PORT = 8889
MOCK_STREAM_URL = f"http://{MOCK_STREAM_HOST}:{MOCK_STREAM_PORT}/whep"
MOCK_STREAM_PROCESS: Optional[subprocess.Popen] = None
MOCK_STREAM_VIDEO: Optional[str] = None
MOCK_STREAM_LOCK = threading.Lock()


def _mock_stream_port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((MOCK_STREAM_HOST, MOCK_STREAM_PORT)) == 0


def start_mock_stream(video_text: str) -> dict[str, Any]:
    """Launch the mock live camera for a server-local video and return its WHEP URL."""
    global MOCK_STREAM_PROCESS, MOCK_STREAM_VIDEO
    video = Path(str(video_text or "").strip())
    if not video.is_absolute():
        video = PROJECT_DIR / video
    if not video.is_file():
        return {"ok": False, "error": f"视频文件不存在: {video}"}
    with MOCK_STREAM_LOCK:
        if MOCK_STREAM_PROCESS is not None and MOCK_STREAM_PROCESS.poll() is None:
            return {"ok": True, "url": MOCK_STREAM_URL, "video": MOCK_STREAM_VIDEO, "running": True}
        if _mock_stream_port_in_use():
            MOCK_STREAM_PROCESS = None
            MOCK_STREAM_VIDEO = str(video)
            return {"ok": True, "url": MOCK_STREAM_URL, "video": str(video), "running": True, "reused": True}
        proc = subprocess.Popen(
            [sys.executable, str(APP_DIR / "mock_live_camera.py"), "--video", str(video)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        MOCK_STREAM_PROCESS = proc
        MOCK_STREAM_VIDEO = str(video)
    return {"ok": True, "url": MOCK_STREAM_URL, "video": str(video), "running": True}


def stop_mock_stream() -> dict[str, Any]:
    """Stop the mock live camera if it was started through this server."""
    global MOCK_STREAM_PROCESS, MOCK_STREAM_VIDEO
    with MOCK_STREAM_LOCK:
        proc = MOCK_STREAM_PROCESS
        MOCK_STREAM_PROCESS = None
        MOCK_STREAM_VIDEO = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return {"ok": True}


def build_response(kind: str, data: dict[str, Any], expected: int, output_media: Path, output_json: Path, command: list[str]) -> dict[str, Any]:
    data.setdefault("finished_at", current_timestamp())
    if kind == "helmet":
        detected = get_helmet_hazard_count(data)
        summary = format_helmet_result(data)
        statistics = format_helmet_statistics(data)
    elif kind == "forklift_work":
        detected = get_forklift_work_hazard_count(data)
        summary = format_forklift_work_result(data)
        statistics = format_forklift_work_statistics(data)
        result = str(data.get("result", "unknown")).lower()
        if result == "safe":
            statistics += "\n叉车作业状态：安全"
        elif result == "unsafe":
            statistics += "\n叉车作业状态：不安全"
    else:
        detected = get_speed_hazard_count(data)
        summary = format_speed_result(data)
        statistics = summary
    accuracy = format_accuracy_statistics(expected, detected)
    hazard_image_path = data.get("hazard_image_path")
    output_media_url = file_url(Path(hazard_image_path)) if hazard_image_path else ""
    return {
        "ok": True,
        "finished_at": data.get("finished_at"),
        "command": " ".join(f'"{x}"' if " " in x else x for x in command),
        "summary": f"{summary}\n{accuracy}",
        "statistics": f"{statistics}\n{accuracy}",
        "accuracy": calculate_monitor_accuracy(expected, detected),
        "detected_hazards": detected,
        "expected_hazards": expected,
        "has_hazard": bool(data.get("has_hazard")),
        "hazard_count": _to_int(data.get("hazard_count")),
        "hazards": _as_list(data.get("hazards")),
        "hazard_event": data.get("hazard_event"),
        "hazard_image_url": file_url(Path(hazard_image_path)) if hazard_image_path else None,
        "json": data,
        "json_text": json.dumps(data, ensure_ascii=False, indent=2),
        "output_media_url": output_media_url,
        "output_json_url": file_url(output_json),
    }


def build_unified_response(
    data: dict[str, Any], expected: int, output_media: Path, output_json: Path, command: list[str]
) -> dict[str, Any]:
    data.setdefault("finished_at", current_timestamp())
    results = data.get("results", {}) if isinstance(data.get("results"), dict) else {}
    selected = [str(item) for item in data.get("selected_checks", [])]
    hazard_counts = {
        "person_helmet": _to_int(results.get("person_helmet", {}).get("no_helmet_count")),
        "driver_helmet": _to_int(results.get("driver_helmet", {}).get("no_helmet_count")),
        "forklift_proximity": _to_int(results.get("forklift_proximity", {}).get("nearby_person_count")),
        "forklift_fence": sum(
            1 for item in _as_list(results.get("forklift_fence", {}).get("forklifts"))
            if (
                isinstance(item, dict)
                and item.get("fence_check_applicable", True)
                and not item.get("has_fence_nearby")
            )
        ),
        "forklift_speed": 1 if results.get("forklift_speed", {}).get("result") == "unsafe" else 0,
        "crane_under_boom": _to_int(results.get("crane_under_boom", {}).get("danger_person_count")),
        "lifting_operation": (
            _to_int(results.get("lifting_operation", {}).get("missing_safety_officer_count"))
            + _to_int(results.get("lifting_operation", {}).get("missing_fence_count"))
        ),
    }
    detected = sum(hazard_counts.get(name, 0) for name in selected)
    lines = [f"综合安全状态：{data.get('result', 'unknown')}", f"处理帧数：{data.get('processed_frames', 0)}"]
    for name in selected:
        item = results.get(name, {})
        lines.append(f"{SAFETY_CHECK_LABELS.get(name, name)}：{item.get('result', 'unknown')}，隐患数 {hazard_counts.get(name, 0)}")
    accuracy = format_accuracy_statistics(expected, detected)
    summary = "\n".join(lines)
    hazard_image_path = data.get("hazard_image_path")
    output_media_url = file_url(Path(hazard_image_path)) if hazard_image_path else ""
    return {
        "ok": True,
        "finished_at": data.get("finished_at"),
        "command": " ".join(f'"{x}"' if " " in x else x for x in command),
        "summary": f"{summary}\n{accuracy}",
        "statistics": f"{summary}\n{accuracy}",
        "accuracy": calculate_monitor_accuracy(expected, detected),
        "detected_hazards": detected,
        "hazards_by_check": {name: hazard_counts.get(name, 0) for name in selected},
        "expected_hazards": expected,
        "has_hazard": bool(data.get("has_hazard")),
        "hazard_count": _to_int(data.get("hazard_count")),
        "hazards": _as_list(data.get("hazards")),
        "hazard_event": data.get("hazard_event"),
        "hazard_image_url": file_url(Path(hazard_image_path)) if hazard_image_path else None,
        "json": data,
        "json_text": json.dumps(data, ensure_ascii=False, indent=2),
        "output_media_url": output_media_url,
        "output_json_url": file_url(output_json),
    }


def write_material_count_csv(csv_path: Path, data: dict[str, Any]) -> None:
    """从物资计数结果生成「类别/检测数量/实际数量/准确度」CSV。"""
    rows = _as_list(data.get("accuracy_rows"))
    if not rows:
        classes = data.get("classes", {}) if isinstance(data.get("classes"), dict) else {}
        rows = [{"class_name": name, "detected": count, "actual": None, "accuracy": ""} for name, count in classes.items()]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["类别", "检测数量", "实际数量", "准确度"])
        for row in rows:
            if not isinstance(row, dict):
                continue
            actual = row.get("actual")
            writer.writerow([
                row.get("class_name", ""),
                row.get("detected", 0),
                "" if actual is None else actual,
                row.get("accuracy", ""),
            ])


def write_safety_csv(csv_path: Path, data: dict[str, Any]) -> None:
    """从安全监测结果生成「违规事件」CSV。"""
    events = _as_list(data.get("hazard_events"))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["序号", "时间", "帧号", "违规类型", "违规描述", "证据图路径"])
        for event in events:
            if not isinstance(event, dict):
                continue
            hazards = _as_list(event.get("hazards"))
            codes = "、".join(str(h.get("code", "")) for h in hazards if isinstance(h, dict))
            messages = "、".join(str(h.get("message", "")) for h in hazards if isinstance(h, dict))
            writer.writerow([
                event.get("event_sequence", ""),
                event.get("hazard_timestamp", ""),
                event.get("frame_id", ""),
                codes,
                messages,
                event.get("hazard_image_path", ""),
            ])


def format_material_count_summary(data: dict[str, Any]) -> str:
    lines = [f"累计计数：{data.get('total', 0)}", "类别计数："]
    classes = data.get("classes", {}) if isinstance(data.get("classes"), dict) else {}
    if classes:
        for name, count in classes.items():
            lines.append(f"{name}: {count}")
    else:
        lines.append("无")
    rows = _as_list(data.get("accuracy_rows"))
    if rows:
        lines.append("")
        lines.append("抽样物资实际数量与准确度：")
        for row in rows:
            if isinstance(row, dict):
                actual = row.get("actual")
                actual_text = "-" if actual is None else str(actual)
                lines.append(
                    f"{row.get('class_name')}: 检测 {row.get('detected', 0)} / 实际 {actual_text}，准确度 {row.get('accuracy', '-')}"
                )
    return "\n".join(lines)


def create_speed_command(
    weights: Path,
    video: Path | str,
    output_video: Path,
    output_json: Path,
    speed_limit: str,
    conf: str,
    speed_window: str,
    min_bg_matches: str,
    motion_model: str,
    device: str,
    source_type: str = "video",
    sample_interval: int = 5,
    camera_mode: str = "oblique",
) -> list[str]:
    return [
        sys.executable,
        str(APP_DIR / "forklift_speed_check.py"),
        "--weights",
        str(weights),
        "--video",
        str(video),
        "--source-type",
        source_type,
        "--sample-interval",
        str(max(1, int(sample_interval))),
        "--camera-mode",
        camera_mode,
        "--output",
        str(output_video),
        "--json",
        str(output_json),
        "--speed-limit-kmh",
        speed_limit,
        "--unsafe-speed-kmh",
        "6.0",
        "--conf",
        conf,
        "--speed-window",
        speed_window,
        "--min-bg-matches",
        min_bg_matches,
        "--motion-model",
        motion_model,
        "--device",
        device,
        "--quiet",
    ]


def is_job_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("cancel_requested"))


def encode_preview(frame: Any, quality: int = 82) -> bytes:
    if frame is None:
        return b""
    try:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))]
        )
        return encoded.tobytes() if ok else b""
    except (cv2.error, TypeError, ValueError):
        return b""


def build_safety_preview_updates(
    job_id: str,
    frame_id: int,
    annotated_frame: Any,
    frame_status: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    """Keep display state current while publishing only explicit hazard events."""
    display_status = dict(frame_status)
    for key in (
        "hazard_event",
        "hazards",
        "hazard_count",
        "hazard_timestamp",
        "hazard_image_path",
        "hazard_image_paths",
        "hazard_image_url",
        "is_first_hazard",
    ):
        display_status.pop(key, None)
    updates: dict[str, Any] = {
        "frame_id": frame_id,
        "latest_frame_bytes": encode_preview(annotated_frame),
        "annotated_stream_url": f"/api/jobs/{job_id}/stream/annotated",
        "frame_status": display_status,
        "algorithm": algorithm,
        "progress_text": f"当前帧：{frame_id}",
    }
    event = frame_status.get("hazard_event")
    if not isinstance(event, dict):
        return updates

    image_paths = [str(path) for path in event.get("hazard_image_paths", []) if path]
    image_path = image_paths[-1] if image_paths else event.get("hazard_image_path")
    if image_path:
        resolved_path = Path(str(image_path)).resolve()
        image_url = file_url(resolved_path)
        event["hazard_image_url"] = image_url
        frame_status["hazard_image_url"] = image_url
        updates.update(
            hazard_image_url=image_url,
            latest_frame_url=image_url,
            annotated_image_path=str(resolved_path),
            annotated_frame_id=frame_id,
            annotated_timestamp=event.get("hazard_timestamp") or current_timestamp(),
        )
    updates.update(
        frame_result=frame_status,
        result_frame_id=frame_id,
        result_timestamp=event.get("hazard_timestamp") or current_timestamp(),
    )
    return updates


def update_safety_job_preview(job_id: str, updates: dict[str, Any]) -> None:
    """Store display state and retain every new incident event until consumers advance their cursor."""
    journal_entry: Optional[dict[str, Any]] = None
    journal_path: Optional[Path] = None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.get("status") in TERMINAL_JOB_STATES:
            return
        frame_result = updates.get("frame_result")
        if isinstance(frame_result, dict):
            event_id = (frame_result.get("hazard_event") or {}).get("event_id")
            retained = job.get("_hazard_event_results")
            if not isinstance(retained, deque):
                retained = deque(retained or [], maxlen=MAX_RETAINED_HAZARD_EVENTS)
                job["_hazard_event_results"] = retained
            if not event_id or all((item.get("hazard_event") or {}).get("event_id") != event_id for item in retained):
                journal_entry = copy.deepcopy(frame_result)
                retained.append(journal_entry)
                configured_journal = str(job.get("_event_journal_path", ""))
                journal_path = Path(configured_journal) if configured_journal else None
        job.update({key: value for key, value in updates.items() if value is not None})
        job["_updated_monotonic"] = time.monotonic()
    if journal_entry is not None and journal_path is not None:
        try:
            append_jsonl_durable(journal_path, journal_entry)
        except Exception as exc:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["_persistence_error"] = str(exc)
            print(f"[event-journal] {job_id} 告警事件持久化失败: {exc}", flush=True)


def select_new_hazard_event(job: dict[str, Any], after_frame_id: int) -> tuple[Optional[dict[str, Any]], bool]:
    """Return a hazard event only when its cursor has not been consumed."""
    retained = job.get("_hazard_event_results", [])
    if isinstance(retained, (list, deque)):
        for frame_result in retained:
            if isinstance(frame_result, dict) and _to_int(frame_result.get("frame_id"), -1) > int(after_frame_id):
                return frame_result, True
    frame_result = job.get("frame_result")
    result_frame_id = _to_int(job.get("result_frame_id"), -1)
    updated = bool(isinstance(frame_result, dict) and result_frame_id > int(after_frame_id))
    return (frame_result if updated else None), updated


def is_annotated_image_updated(job: dict[str, Any], after_annotated_frame_id: int) -> bool:
    """Check whether a newer persisted violation annotation is available."""
    annotated_frame_id = _to_int(job.get("annotated_frame_id"), -1)
    return bool(job.get("latest_frame_url") and annotated_frame_id > int(after_annotated_frame_id))


def build_job_status_payload(
    stored_job: dict[str, Any],
    query: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """Return incremental detection logs and optionally cursor-filtered hazard events."""
    job = dict(stored_job)
    job.pop("latest_frame_bytes", None)
    job.pop("latest_raw_frame_bytes", None)
    after_log_id = max(0, _to_int((query or {}).get("after_log_id", ["0"])[0], 0))
    entries = list(job.get("_detection_logs", []))
    job["detection_logs"] = [entry for entry in entries if entry["log_id"] > after_log_id]
    job["detection_logs_dropped"] = max(0, entries[0]["log_id"] - after_log_id - 1) if entries else 0
    cursor_requested = bool(
        query and ("after_frame_id" in query or "after_annotated_frame_id" in query)
    )
    if cursor_requested:
        after_frame_id = _to_int((query or {}).get("after_frame_id", ["-1"])[0], -1)
        after_annotated_frame_id = _to_int(
            (query or {}).get("after_annotated_frame_id", ["-1"])[0],
            -1,
        )
        frame_result, result_updated = select_new_hazard_event(job, after_frame_id)
        job["frame_result"] = frame_result
        job["result_updated"] = result_updated
        if frame_result is not None:
            job["result_frame_id"] = _to_int(frame_result.get("frame_id"), -1)
            event = frame_result.get("hazard_event") if isinstance(frame_result.get("hazard_event"), dict) else {}
            job["result_timestamp"] = event.get("hazard_timestamp") or job.get("result_timestamp")
        job["annotated_updated"] = is_annotated_image_updated(job, after_annotated_frame_id)
    for key in [name for name in job if name.startswith("_")]:
        job.pop(key, None)
    return job


def run_live_test_job(job_id: str, stream_url: str) -> None:
    from live_stream import acquire_shared_live_reader, release_shared_live_reader

    started_at = time.monotonic()
    last_preview_at = 0.0
    last_progress_text = ""
    first_frame_logged = False

    def log_progress(message: str) -> None:
        print(f"[live-test] {job_id} {message}", flush=True)

    def on_raw(packet) -> None:
        nonlocal first_frame_logged, last_preview_at
        now = time.monotonic()
        if now - last_preview_at < 0.08:
            return
        last_preview_at = now
        frame_bytes = encode_preview(packet.frame)
        update_job(
            job_id,
            raw_frame_id=packet.frame_id,
            latest_raw_frame_bytes=frame_bytes,
            progress_text=f"直播测试正常，当前帧：{packet.frame_id}",
            stream_ready=True,
        )
        if not first_frame_logged:
            first_frame_logged = True
            log_progress(f"已收到并解码第一帧（frame_id={packet.frame_id}）。")

    log_progress("任务已启动，准备由 Python 后端直接连接 WebRTC。")
    reader = acquire_shared_live_reader(stream_url, frame_callback=on_raw)
    final_status = "canceled"
    final_message = "直播测试已停止"
    try:
        while not is_job_cancel_requested(job_id):
            if reader.recovering:
                progress_text = reader.error or "直播流暂时中断，Python 后端正在自动恢复。"
                if progress_text != last_progress_text:
                    last_progress_text = progress_text
                    log_progress(progress_text)
                    update_job(job_id, progress_text=progress_text, stream_ready=False)
            elif not reader.opened_once:
                progress_text = reader.error or reader.last_failure or "Python 后端正在初始化 WHEP 连接。"
                if progress_text != last_progress_text:
                    last_progress_text = progress_text
                    log_progress(progress_text)
                    update_job(job_id, progress_text=progress_text, stream_ready=False)
            if not reader.opened_once and time.monotonic() - started_at > reader.startup_timeout_seconds:
                raise RuntimeError(reader.last_failure or reader.error or "直播流连接超时。")
            time.sleep(0.1)
    except Exception as exc:
        final_status = "error"
        final_message = str(exc)
    finally:
        # Keep the underlying WebRTC session briefly available so an algorithm
        # started from this preview can acquire the exact same reader without
        # repeating SDP / ICE negotiation.
        with JOBS_LOCK:
            release_immediately = bool(JOBS.get(job_id, {}).get("_release_stream_immediately", False))
        release_shared_live_reader(
            stream_url,
            reader,
            frame_callback=on_raw,
            idle_seconds=0.0 if release_immediately else 10.0,
        )
        updates: dict[str, Any] = {
            "status": final_status,
            "progress_text": final_message,
            "stream_ready": False,
        }
        if final_status == "error":
            updates["error"] = final_message
        update_job(job_id, **updates)
        if final_status == "error":
            log_progress(f"失败：{final_message}")
        else:
            log_progress("直播测试订阅已释放；共享 WebRTC 连接可由算法继续复用。")


def run_safety_monitor_job(
    job_id: str,
    mode: str,
    source_type: str,
    source: Path | str,
    weights: Path,
    output_media: Path,
    output_json: Path,
    expected: int,
    conf: float,
    iou: float,
    device: str,
    work_options: dict[str, Any],
    command: list[str],
    sample_interval: int = 5,
) -> None:
    last_raw_preview_at = 0.0

    def set_job(**updates: Any) -> None:
        update_job(job_id, **updates)

    def preview_callback(frame_id: int, annotated_frame: Any, frame_status: dict[str, Any]) -> None:
        update_safety_job_preview(
            job_id, build_safety_preview_updates(job_id, frame_id, annotated_frame, frame_status, mode)
        )

    def raw_preview_callback(frame_id: int, raw_frame: Any, frame_status: dict[str, Any]) -> None:
        nonlocal last_raw_preview_at
        now = time.monotonic()
        if now - last_raw_preview_at < 0.08:
            return
        last_raw_preview_at = now
        set_job(
            raw_frame_id=frame_id,
            latest_raw_frame_bytes=encode_preview(raw_frame),
            raw_frame_status=frame_status,
        )

    def progress_callback(processed: int, total: int) -> None:
        progress = int(processed * 100 / total) if total > 0 else 0
        set_job(progress=progress, progress_text=f"进度：{processed}/{total if total > 0 else '?'}")

    try:
        from safety_video_monitor import run_safety_monitor

        result = run_safety_monitor(
            mode=mode,
            source_type=source_type,
            source_path=source,
            weights_path=str(weights),
            output_media=output_media,
            conf=conf,
            iou=iou,
            device=device,
            work_options=work_options,
            preview_callback=preview_callback,
            progress_callback=progress_callback,
            stop_flag=lambda: is_job_cancel_requested(job_id),
            sample_interval=sample_interval,
            raw_preview_callback=raw_preview_callback,
            detection_log_callback=lambda frame_id, message: append_job_detection_log(job_id, frame_id, message),
        )
        write_json_atomic(output_json, result)
        payload = build_response(mode, result, expected, output_media, output_json, command)
        payload["logs"] = ""
        stopped = bool(result.get("stopped_by_user")) or is_job_cancel_requested(job_id)
        status = "canceled" if stopped else "done"
        set_job(
            status=status,
            progress=100 if status == "done" else None,
            result=payload,
            output_media_url=payload["output_media_url"],
        )
    except Exception as exc:
        if is_job_cancel_requested(job_id):
            set_job(status="canceled", error="任务已停止")
        else:
            set_job(status="error", error=str(exc))


def run_unified_safety_job(
    job_id: str,
    enabled_checks: set[str],
    source_type: str,
    source: Path | str,
    weights: Path,
    output_media: Path,
    output_json: Path,
    expected: int,
    conf: float,
    iou: float,
    device: str,
    work_options: dict[str, Any],
    speed_options: dict[str, Any],
    command: list[str],
    sample_interval: int = 5,
    crane_options: Optional[dict[str, Any]] = None,
) -> None:
    last_raw_preview_at = 0.0

    def set_job(**updates: Any) -> None:
        update_job(job_id, **updates)

    def preview_callback(frame_id: int, annotated_frame: Any, frame_status: dict[str, Any]) -> None:
        update_safety_job_preview(
            job_id,
            build_safety_preview_updates(job_id, frame_id, annotated_frame, frame_status, "safety_combined"),
        )

    def raw_preview_callback(frame_id: int, raw_frame: Any, frame_status: dict[str, Any]) -> None:
        nonlocal last_raw_preview_at
        now = time.monotonic()
        if now - last_raw_preview_at < 0.08:
            return
        last_raw_preview_at = now
        set_job(raw_frame_id=frame_id, latest_raw_frame_bytes=encode_preview(raw_frame), raw_frame_status=frame_status)

    def progress_callback(processed: int, total: int) -> None:
        progress = int(processed * 100 / total) if total > 0 else 0
        set_job(progress=progress, progress_text=f"进度：{processed}/{total if total > 0 else '?'}")

    try:
        from safety_video_monitor import run_unified_safety_monitor

        result = run_unified_safety_monitor(
            enabled_checks=enabled_checks,
            source_type=source_type,
            source_path=source,
            weights_path=str(weights),
            output_media=output_media,
            conf=conf,
            iou=iou,
            device=device,
            work_options=work_options,
            speed_options=speed_options,
            crane_options=crane_options or {},
            preview_callback=preview_callback,
            progress_callback=progress_callback,
            stop_flag=lambda: is_job_cancel_requested(job_id),
            sample_interval=sample_interval,
            raw_preview_callback=raw_preview_callback,
            detection_log_callback=lambda frame_id, message: append_job_detection_log(job_id, frame_id, message),
        )
        write_json_atomic(output_json, result)
        output_csv = output_json.with_suffix(".csv")
        write_safety_csv(output_csv, result)
        payload = build_unified_response(result, expected, output_media, output_json, command)
        payload["output_csv_url"] = file_url(output_csv)
        payload["logs"] = ""
        stopped = bool(result.get("stopped_by_user")) or is_job_cancel_requested(job_id)
        status = "canceled" if stopped else "done"
        set_job(
            status=status,
            progress=100 if status == "done" else None,
            result=payload,
            output_media_url=payload["output_media_url"],
        )
    except Exception as exc:
        if is_job_cancel_requested(job_id):
            set_job(status="canceled", error="任务已停止")
        else:
            set_job(status="error", error=str(exc))


def run_speed_job(
    job_id: str,
    weights: Path,
    video: Path | str,
    output_video: Path,
    output_json: Path,
    expected: int,
    speed_limit: float,
    conf: float,
    speed_window: int,
    min_bg_matches: int,
    motion_model: str,
    device: str,
    command: list[str],
    source_type: str = "video",
    sample_interval: int = 5,
    camera_mode: str = "oblique",
) -> None:
    last_raw_preview_at = 0.0

    def set_job(**updates: Any) -> None:
        update_job(job_id, **updates)

    def preview_callback(frame_id: int, annotated_frame: Any, frame_status: dict[str, Any]) -> None:
        update_safety_job_preview(
            job_id,
            build_safety_preview_updates(job_id, frame_id, annotated_frame, frame_status, "forklift_speed"),
        )

    def raw_preview_callback(frame_id: int, raw_frame: Any, frame_status: dict[str, Any]) -> None:
        nonlocal last_raw_preview_at
        now = time.monotonic()
        if now - last_raw_preview_at < 0.08:
            return
        last_raw_preview_at = now
        set_job(
            raw_frame_id=frame_id,
            latest_raw_frame_bytes=encode_preview(raw_frame),
            raw_frame_status=frame_status,
        )

    def progress_callback(processed: int, total: int) -> None:
        progress = int(processed * 100 / total) if total > 0 else 0
        set_job(progress=progress, progress_text=f"进度：{processed}/{total if total > 0 else '?'}")

    try:
        from forklift_speed_check import process_forklift_speed_video

        result = process_forklift_speed_video(
            weights_path=str(weights),
            video_path=str(video),
            output_path=str(output_video),
            json_path=str(output_json),
            speed_limit_kmh=speed_limit,
            unsafe_speed_kmh=6.0,
            preview_callback=preview_callback,
            conf=conf,
            speed_window=speed_window,
            min_bg_matches=min_bg_matches,
            motion_model=motion_model,
            device=device,
            stop_flag=lambda: is_job_cancel_requested(job_id),
            progress_callback=progress_callback,
            quiet=True,
            source_type=source_type,
            sample_interval=sample_interval,
            raw_preview_callback=raw_preview_callback,
            detection_log_callback=lambda frame_id, message: append_job_detection_log(job_id, frame_id, message),
            camera_mode=camera_mode,
        )
        data = load_json(output_json) if output_json.exists() else result
        payload = build_response("speed", data, expected, output_video, output_json, command)
        payload["logs"] = ""
        stopped = bool(data.get("summary", {}).get("stopped_by_user")) if isinstance(data.get("summary"), dict) else False
        status = "canceled" if stopped or is_job_cancel_requested(job_id) else "done"
        updates = {"status": status, "result": payload, "output_media_url": payload["output_media_url"]}
        if status == "done":
            updates["progress"] = 100
        set_job(**updates)
    except Exception as exc:
        if is_job_cancel_requested(job_id):
            set_job(status="canceled", error="任务已停止")
        else:
            set_job(status="error", error=str(exc))


def run_material_count_job(
    job_id: str,
    source_type: str,
    source: Path | str,
    model: Path,
    output_media: Path,
    output_json: Path,
    tracker_args: dict[str, Any],
    expected_counts: str,
    target_classes: str,
    conf: float,
    device: str,
    command: list[str],
    sample_interval: int = 5,
) -> None:
    latest_frame = DEFAULT_OUTPUT_DIR / f"material_count_preview_{job_id}.jpg"
    last_raw_preview_at = 0.0
    last_disk_preview_at = 0.0

    def set_job(**updates: Any) -> None:
        update_job(job_id, **updates)

    def progress_callback(processed: int, total: int) -> None:
        # 直播流没有总帧数，此时不下发百分比，避免前端显示误导性的 0%。
        if total > 0:
            set_job(progress=int(processed * 100 / total), progress_text=f"已处理 {processed}/{total} 帧")
        else:
            set_job(progress=None, progress_text=f"已处理 {processed} 帧")

    def preview_callback(frame_id: int, annotated_frame: Any, frame_status: dict[str, Any]) -> None:
        nonlocal last_disk_preview_at
        frame_bytes = encode_preview(annotated_frame)
        now = time.monotonic()
        if frame_bytes and now - last_disk_preview_at >= 1.0:
            write_bytes_atomic(latest_frame, frame_bytes)
            last_disk_preview_at = now
        set_job(
            frame_id=frame_id,
            latest_frame_bytes=frame_bytes,
            latest_frame_url=file_url(latest_frame),
            frame_status=frame_status,
            frame_result=frame_status,
            result_timestamp=current_timestamp(),
            algorithm="material_count",
        )

    def raw_preview_callback(frame_id: int, raw_frame: Any, frame_status: dict[str, Any]) -> None:
        nonlocal last_raw_preview_at
        now = time.monotonic()
        if now - last_raw_preview_at < 0.08:
            return
        last_raw_preview_at = now
        set_job(
            raw_frame_id=frame_id,
            latest_raw_frame_bytes=encode_preview(raw_frame),
            raw_frame_status=frame_status,
        )

    try:
        from material_count_cli import (
            MaterialCounter,
            TrackerArgs,
            build_accuracy_rows,
            normalize_expected_counts,
            parse_expected_counts,
            parse_target_classes,
        )

        args = TrackerArgs(**tracker_args)
        counter = MaterialCounter(
            str(model),
            float(conf),
            args,
            str(device),
            stop_flag=lambda: is_job_cancel_requested(job_id),
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            raw_preview_callback=raw_preview_callback,
            detection_log_callback=lambda frame_id, message: append_job_detection_log(job_id, frame_id, message),
            sample_interval=sample_interval,
            target_classes=parse_target_classes(target_classes),
        )
        result = counter.run(source_type, source, output_media)
        expected = normalize_expected_counts(parse_expected_counts(expected_counts), result["target_classes"])
        result["expected_counts"] = expected
        result["accuracy_rows"] = build_accuracy_rows(result["classes"], expected)
        result.setdefault("finished_at", current_timestamp())
        result["created_at"] = result["finished_at"]
        write_json_atomic(output_json, result)
        output_csv = output_json.with_suffix(".csv")
        write_material_count_csv(output_csv, result)
        summary = format_material_count_summary(result)
        payload = {
            "ok": True,
            "finished_at": result.get("finished_at"),
            "command": " ".join(f'"{x}"' if " " in x else x for x in command),
            "summary": summary,
            "statistics": summary,
            "json": result,
            "json_text": json.dumps(result, ensure_ascii=False, indent=2),
            "output_media_url": file_url(Path(result.get("output_media_path") or output_media)),
            "output_json_url": file_url(output_json),
            "output_csv_url": file_url(output_csv),
            "logs": "",
        }
        status = "canceled" if result.get("stopped_by_user") or is_job_cancel_requested(job_id) else "done"
        set_job(status=status, progress=100 if status == "done" else None, result=payload, output_media_url=payload["output_media_url"])
    except Exception as exc:
        if is_job_cancel_requested(job_id):
            set_job(status="canceled", error="任务已停止")
        else:
            set_job(status="error", error=str(exc))


class SafetyWebHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30.0)

    def log_message(self, fmt: str, *args: Any) -> None:
        request_line = str(args[0]) if args else ""
        status = str(args[1]) if len(args) > 1 else ""
        if request_line.startswith("GET /api/jobs/") and status == "200":
            return
        print(f"[web] {self.address_string()} {fmt % args}")

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
            pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.serve_static(WEB_DIR / "index.html")
        if parsed.path.startswith("/web/"):
            return self.serve_static(WEB_DIR / unquote(parsed.path.removeprefix("/web/")))
        if parsed.path == "/api/config":
            gpu_info = detect_gpu_info()
            return self.send_json(
                {
                    "default_weights": str(DEFAULT_WEIGHTS),
                    "default_material_model": str(DEFAULT_MATERIAL_MODEL),
                    "inference_imgsz": INFERENCE_IMAGE_SIZE,
                    "material_class_candidates": list(MATERIAL_CLASS_CANDIDATES),
                    "default_output_dir": str(DEFAULT_OUTPUT_DIR),
                    "default_device": default_device(),
                    "has_gpu": bool(gpu_info.get("has_gpu")),
                    "gpu_count": int(gpu_info.get("gpu_count", 0) or 0),
                    "gpu_detection_error": gpu_info.get("error", ""),
                }
            )
        if parsed.path == "/api/health":
            return self.send_json(runtime_health_payload())
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            target = Path(query.get("path", [""])[0])
            return self.serve_file(target)
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            kind = query.get("kind", ["file"])[0]
            if kind not in {"file", "dir", "video"}:
                return self.send_json({"ok": False, "error": f"不支持的选择类型: {kind}"}, 400)
            return self.send_json(browse_path(kind, query.get("path", [""])[0]))
        if parsed.path == "/api/select-path":
            query = parse_qs(parsed.query)
            kind = query.get("kind", ["dir"])[0]
            if kind not in {"file", "dir"}:
                return self.send_json({"ok": False, "error": f"不支持的选择类型: {kind}"}, 400)
            return self.send_json(select_path_dialog(kind, query.get("initial", [""])[0]))
        if parsed.path == "/api/model-classes":
            query = parse_qs(parsed.query)
            info = describe_model(query.get("path", [""])[0])
            info["supported"] = bool(info.get("ok")) and info.get("task") in MATERIAL_SUPPORTED_TASKS
            return self.send_json(info)
        if parsed.path.startswith("/api/jobs/") and "/stream" in parsed.path:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 4 and parts[0:2] == ["api", "jobs"] and parts[3] == "stream":
                variant = parts[4] if len(parts) > 4 else "annotated"
                if variant in {"raw", "annotated"}:
                    return self.handle_job_stream(parts[2], variant)
        if parsed.path.startswith("/api/jobs/"):
            return self.handle_job_status(parsed.path.rsplit("/", 1)[-1], parse_qs(parsed.query))
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/detect/safety/start":
                return self.handle_unified_safety()
            if parsed.path == "/api/detect/helmet":
                return self.handle_helmet()
            if parsed.path == "/api/detect/forklift-work":
                return self.handle_forklift_work()
            if parsed.path == "/api/detect/forklift-speed":
                return self.handle_forklift_speed()
            if parsed.path == "/api/detect/forklift-speed/start":
                return self.handle_forklift_speed_start()
            if parsed.path == "/api/detect/material-count":
                return self.handle_material_count()
            if parsed.path == "/api/detect/material-count-live":
                return self.handle_material_count_live()
            if parsed.path == "/api/model-classes":
                return self.handle_model_classes_upload()
            if parsed.path == "/api/mock-stream/start":
                return self.handle_mock_stream_start()
            if parsed.path == "/api/mock-stream/stop":
                return self.handle_mock_stream_stop()
            if parsed.path == "/api/live-stream/test":
                return self.handle_live_stream_test()
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                return self.handle_job_cancel(parsed.path.split("/")[-2], parse_qs(parsed.query))
        except (ValueError, FileNotFoundError) as exc:
            return self.send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:
            return self.send_json({"ok": False, "error": str(exc)}, 500)
        self.send_error(404)

    def parse_form(self) -> cgi.FieldStorage:
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})

    def common_paths(self, form: cgi.FieldStorage, stem: str, media_suffix: str) -> tuple[Path, Path, Path, int]:
        weights = save_upload(form, "weights_file", field_value(form, "weights", str(DEFAULT_WEIGHTS)), "weights")
        if not weights.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights}")
        output_dir = ensure_dir(Path(field_value(form, "output_dir", str(DEFAULT_OUTPUT_DIR)).strip()))
        token = int(time.time() * 1000)
        output_media = output_dir / f"{stem}_{token}{media_suffix}"
        output_json = output_dir / f"{stem}_{token}.json"
        expected = _to_int(field_value(form, "expected_hazards", "0"), 0)
        return weights, output_media, output_json, expected

    def handle_unified_safety(self) -> None:
        form = self.parse_form()
        allowed_checks = set(SAFETY_CHECK_LABELS)
        enabled_checks = {str(item) for item in form.getlist("checks") if str(item) in allowed_checks}
        if not enabled_checks:
            raise ValueError("请至少勾选一个安全监测功能。")
        source_type = field_value(form, "source_type", "single_image")
        if source_type not in {"single_image", "image_sequence", "video", "live_stream"}:
            raise ValueError(f"不支持的输入类型: {source_type}")
        if source_type == "single_image" and "forklift_speed" in enabled_checks:
            raise ValueError("叉车速度检测需要视频、直播流或连续图片序列，不能使用单张图片。")

        suffix = ".jpg" if source_type == "single_image" else ".mp4"
        weights, output_media, output_json, expected = self.common_paths(form, "safety_combined_result", suffix)
        source_path = field_value(form, "source_path")
        sample_interval = max(1, _to_int(field_value(form, "sample_interval", "5"), 5))
        if source_type == "live_stream":
            source: Path | str = validate_live_stream_url(field_value(form, "stream_url"))
        elif source_type == "image_sequence":
            source = save_upload_directory(form, "source_file", source_path, "safety_combined_sequence")
            if not source.is_dir():
                raise FileNotFoundError(f"图片序列文件夹不存在: {source}")
        else:
            source = save_upload(form, "source_file", source_path, "safety_combined")

        device = resolve_device(field_value(form, "device", default_device()))
        camera_mode = field_value(form, "camera_mode", "oblique")
        if camera_mode not in {"oblique", "overhead"}:
            raise ValueError("camera_mode 必须是 oblique 或 overhead。")
        work_options = {
            "nearby_mode": field_value(form, "nearby_mode", "circle"),
            "camera_mode": camera_mode,
            "forklift_radius_scale": float(field_value(form, "forklift_radius_scale", "1.2")),
            "min_forklift_radius_px": float(field_value(form, "min_forklift_radius_px", "180")),
            "max_forklift_radius_px": float(field_value(form, "max_forklift_radius_px", "450")),
            "fence_iou_thres": float(field_value(form, "fence_iou_thres", "0.01")),
            "draw_circle_work_zone": bool_field(form, "draw_circle_work_zone", True),
        }
        speed_options = {
            "speed_limit": float(field_value(form, "speed_limit", "5.0")),
            "unsafe_speed_limit": 6.0,
            "movement_threshold": float(field_value(form, "movement_threshold", "1.0")),
            "speed_window": int(field_value(form, "speed_window", "30")),
            "min_bg_matches": int(field_value(form, "min_bg_matches", "20")),
            "motion_model": field_value(form, "motion_model", "affine"),
            "camera_mode": camera_mode,
        }
        crane_options = {
            "boom_conf": float(field_value(form, "boom_conf", "0.40")),
            "person_conf": float(field_value(form, "crane_person_conf", "0.40")),
            "line_margin": float(field_value(form, "crane_line_margin", "5")),
            "bottom_margin": float(field_value(form, "crane_bottom_margin", "5")),
            "range_extend_ratio": float(field_value(form, "crane_range_extend_ratio", "0.05")),
            "enter_window": int(field_value(form, "crane_enter_window", "5")),
            "enter_hits": int(field_value(form, "crane_enter_hits", "3")),
            "exit_safe_frames": int(field_value(form, "crane_exit_safe_frames", "5")),
            "track_ttl_frames": int(field_value(form, "crane_track_ttl_frames", "30")),
        }
        if not 0.0 <= crane_options["boom_conf"] <= 1.0 or not 0.0 <= crane_options["person_conf"] <= 1.0:
            raise ValueError("吊臂和人员置信度必须在 0 到 1 之间。")
        if crane_options["line_margin"] < 0 or crane_options["bottom_margin"] < 0:
            raise ValueError("吊臂几何容差不能为负数。")
        if crane_options["range_extend_ratio"] < 0:
            raise ValueError("吊臂长度扩展比例不能为负数。")
        if crane_options["enter_window"] < 1 or not 1 <= crane_options["enter_hits"] <= crane_options["enter_window"]:
            raise ValueError("吊臂时序进入阈值必须介于 1 和窗口帧数之间。")
        if crane_options["exit_safe_frames"] < 1 or crane_options["track_ttl_frames"] < 1:
            raise ValueError("吊臂时序退出帧数和轨迹保留帧数必须大于 0。")
        selected_order = [name for name in SAFETY_CHECK_LABELS if name in enabled_checks]
        command = [
            sys.executable,
            str(APP_DIR / "safety_video_monitor.py"),
            "--checks",
            ",".join(selected_order),
            "--source-type",
            source_type,
            "--source",
            "[直播地址已隐藏]" if source_type == "live_stream" else str(source),
            "--weights",
            str(weights),
            "--output",
            str(output_media),
            "--json",
            str(output_json),
            "--conf",
            field_value(form, "conf", "0.4"),
            "--iou",
            str(NMS_IOU),
            "--device",
            device,
            "--sample-interval",
            str(sample_interval),
            "--camera-mode",
            camera_mode,
            "--speed-limit",
            str(speed_options["speed_limit"]),
            "--unsafe-speed-limit",
            str(speed_options["unsafe_speed_limit"]),
            "--movement-threshold",
            str(speed_options["movement_threshold"]),
            "--speed-window",
            str(speed_options["speed_window"]),
            "--min-bg-matches",
            str(speed_options["min_bg_matches"]),
            "--motion-model",
            str(speed_options["motion_model"]),
        ]
        if "crane_under_boom" in enabled_checks:
            command.extend(
                [
                    "--boom-conf",
                    str(crane_options["boom_conf"]),
                    "--crane-person-conf",
                    str(crane_options["person_conf"]),
                    "--crane-line-margin",
                    str(crane_options["line_margin"]),
                    "--crane-bottom-margin",
                    str(crane_options["bottom_margin"]),
                    "--crane-range-extend-ratio",
                    str(crane_options["range_extend_ratio"]),
                    "--crane-enter-window",
                    str(crane_options["enter_window"]),
                    "--crane-enter-hits",
                    str(crane_options["enter_hits"]),
                    "--crane-exit-safe-frames",
                    str(crane_options["exit_safe_frames"]),
                    "--crane-track-ttl-frames",
                    str(crane_options["track_ttl_frames"]),
                ]
            )
        job_id = uuid.uuid4().hex
        command_text = " ".join(f'"{item}"' if " " in item else item for item in command)
        create_job(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "progress": 0,
                "progress_text": f"任务已启动（{'斜拍模式' if camera_mode == 'oblique' else '俯拍模式'}），正在加载统一模型",
                "command": command_text,
                "algorithm": "safety_combined",
                "selected_checks": selected_order,
                "_live_stream_url": str(source) if source_type == "live_stream" else "",
                "_event_journal_path": str(output_json.with_suffix(".events.jsonl")),
            },
        )
        start_managed_job(
            job_id,
            run_unified_safety_job,
            (
                job_id,
                enabled_checks,
                source_type,
                source,
                weights,
                output_media,
                output_json,
                expected,
                float(field_value(form, "conf", "0.4")),
                NMS_IOU,
                device,
                work_options,
                speed_options,
                command,
                sample_interval,
                crane_options,
            ),
        )
        self.send_json({"ok": True, "job_id": job_id, "command": command_text, "selected_checks": selected_order})

    def handle_helmet(self) -> None:
        self.start_safety_monitor("helmet")

    def handle_forklift_work(self) -> None:
        self.start_safety_monitor("forklift_work")

    def start_safety_monitor(self, mode: str) -> None:
        form = self.parse_form()
        source_type = field_value(form, "source_type", "single_image")
        if source_type not in {"single_image", "image_sequence", "video", "live_stream"}:
            raise ValueError(f"不支持的输入类型: {source_type}")
        suffix = ".jpg" if source_type == "single_image" else ".mp4"
        stem = "helmet_result" if mode == "helmet" else "forklift_work_result"
        weights, output_media, output_json, expected = self.common_paths(form, stem, suffix)
        source_path = field_value(form, "source_path", field_value(form, "image_path"))
        sample_interval = max(1, _to_int(field_value(form, "sample_interval", "5"), 5))
        if source_type == "live_stream":
            source = validate_live_stream_url(field_value(form, "stream_url"))
        elif source_type == "image_sequence":
            source = save_upload_directory(form, "source_file", source_path, f"{mode}_sequence")
            if not source.is_dir():
                raise FileNotFoundError(f"图片序列文件夹不存在: {source}")
        else:
            source = save_upload(form, "source_file", source_path, mode)
        device = resolve_device(field_value(form, "device", default_device()))
        command = [
            sys.executable,
            str(APP_DIR / "safety_video_monitor.py"),
            "--mode",
            mode,
            "--source-type",
            source_type,
            "--source",
            "[直播地址已隐藏]" if source_type == "live_stream" else str(source),
            "--weights",
            str(weights),
            "--output",
            str(output_media),
            "--json",
            str(output_json),
            "--conf",
            field_value(form, "conf", "0.4"),
            "--iou",
            str(NMS_IOU),
            "--device",
            device,
            "--sample-interval",
            str(sample_interval),
        ]
        work_options: dict[str, Any] = {}
        if mode == "forklift_work":
            work_options = {
                "nearby_mode": field_value(form, "nearby_mode", "circle"),
                "camera_mode": field_value(form, "camera_mode", "oblique"),
                "forklift_radius_scale": float(field_value(form, "forklift_radius_scale", "1.2")),
                "min_forklift_radius_px": float(field_value(form, "min_forklift_radius_px", "180")),
                "max_forklift_radius_px": float(field_value(form, "max_forklift_radius_px", "450")),
                "fence_iou_thres": float(field_value(form, "fence_iou_thres", "0.01")),
                "draw_circle_work_zone": bool_field(form, "draw_circle_work_zone", True),
            }
            command.extend(
                [
                    "--nearby-mode",
                    str(work_options["nearby_mode"]),
                    "--camera-mode",
                    str(work_options["camera_mode"]),
                    "--forklift-radius-scale",
                    str(work_options["forklift_radius_scale"]),
                    "--min-forklift-radius-px",
                    str(work_options["min_forklift_radius_px"]),
                    "--max-forklift-radius-px",
                    str(work_options["max_forklift_radius_px"]),
                    "--fence-iou-thres",
                    str(work_options["fence_iou_thres"]),
                ]
            )
        job_id = uuid.uuid4().hex
        command_text = " ".join(f'"{item}"' if " " in item else item for item in command)
        create_job(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "progress": 0,
                "progress_text": "任务已启动，正在加载模型",
                "command": command_text,
                "algorithm": mode,
                "_live_stream_url": str(source) if source_type == "live_stream" else "",
                "_event_journal_path": str(output_json.with_suffix(".events.jsonl")),
            },
        )
        start_managed_job(
            job_id,
            run_safety_monitor_job,
            (
                job_id,
                mode,
                source_type,
                source,
                weights,
                output_media,
                output_json,
                expected,
                float(field_value(form, "conf", "0.4")),
                NMS_IOU,
                device,
                work_options,
                command,
                sample_interval,
            ),
        )
        self.send_json({"ok": True, "job_id": job_id, "command": command_text})

    def handle_forklift_speed(self) -> None:
        form = self.parse_form()
        weights, output_video, output_json, expected = self.common_paths(form, "forklift_speed_result", ".mp4")
        video = save_upload(form, "video", field_value(form, "video_path"), "forklift_speed")
        device = resolve_device(field_value(form, "device", default_device()))
        camera_mode = field_value(form, "camera_mode", "oblique")
        if camera_mode not in {"oblique", "overhead"}:
            raise ValueError("camera_mode 必须是 oblique 或 overhead。")
        command = create_speed_command(
            weights,
            video,
            output_video,
            output_json,
            field_value(form, "speed_limit", "5.0"),
            field_value(form, "conf", "0.4"),
            field_value(form, "speed_window", "30"),
            field_value(form, "min_bg_matches", "20"),
            field_value(form, "motion_model", "affine"),
            device,
            camera_mode=camera_mode,
        )
        rc, stdout, stderr = run_command(command)
        if rc != 0:
            return self.send_json({"ok": False, "error": stderr or stdout or "脚本运行失败", "command": " ".join(command)}, 500)
        data = load_json(output_json)
        payload = build_response("speed", data, expected, output_video, output_json, command)
        payload["logs"] = stdout + stderr
        self.send_json(payload)

    def handle_forklift_speed_start(self) -> None:
        form = self.parse_form()
        weights, output_video, output_json, expected = self.common_paths(form, "forklift_speed_result", ".mp4")
        source_type = field_value(form, "source_type", "video")
        if source_type not in {"video", "live_stream"}:
            raise ValueError(f"不支持的输入类型: {source_type}")
        sample_interval = max(1, _to_int(field_value(form, "sample_interval", "5"), 5))
        if source_type == "live_stream":
            video: Path | str = validate_live_stream_url(field_value(form, "stream_url"))
        else:
            video = save_upload(form, "video", field_value(form, "video_path"), "forklift_speed")
        speed_limit_text = field_value(form, "speed_limit", "5.0")
        conf_text = field_value(form, "conf", "0.4")
        speed_window_text = field_value(form, "speed_window", "30")
        min_bg_matches_text = field_value(form, "min_bg_matches", "20")
        motion_model = field_value(form, "motion_model", "affine")
        camera_mode = field_value(form, "camera_mode", "oblique")
        if camera_mode not in {"oblique", "overhead"}:
            raise ValueError("camera_mode 必须是 oblique 或 overhead。")
        device = resolve_device(field_value(form, "device", default_device()))
        command = create_speed_command(
            weights,
            video,
            output_video,
            output_json,
            speed_limit_text,
            conf_text,
            speed_window_text,
            min_bg_matches_text,
            motion_model,
            device,
            source_type,
            sample_interval,
            camera_mode,
        )
        if source_type == "live_stream":
            command[command.index("--video") + 1] = "[直播地址已隐藏]"
        job_id = uuid.uuid4().hex
        command_text = " ".join(f'"{x}"' if " " in x else x for x in command)
        create_job(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "progress": 0,
                "progress_text": "任务已启动",
                "command": command_text,
                "algorithm": "forklift_speed",
                "_live_stream_url": str(video) if source_type == "live_stream" else "",
                "_event_journal_path": str(output_json.with_suffix(".events.jsonl")),
            },
        )
        start_managed_job(
            job_id,
            run_speed_job,
            (
                job_id,
                weights,
                video,
                output_video,
                output_json,
                expected,
                float(speed_limit_text),
                float(conf_text),
                int(speed_window_text),
                int(min_bg_matches_text),
                motion_model,
                device,
                command,
                source_type,
                sample_interval,
                camera_mode,
            ),
        )
        self.send_json({"ok": True, "job_id": job_id, "command": command_text})

    def handle_job_status(self, job_id: str, query: Optional[dict[str, list[str]]] = None) -> None:
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
            if "_detection_logs" in job:
                job["_detection_logs"] = list(job["_detection_logs"])
        if not job:
            return self.send_json({"ok": False, "error": "任务不存在"}, 404)
        self.send_json(build_job_status_payload(job, query))

    def handle_job_cancel(
        self,
        job_id: str,
        query: Optional[dict[str, list[str]]] = None,
    ) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return self.send_json({"ok": False, "error": "任务不存在"}, 404)
            if job.get("status") in TERMINAL_JOB_STATES:
                return self.send_json({"ok": True, "job_id": job_id, "status": job.get("status")})
            job["cancel_requested"] = True
            if query and query.get("release_stream", [""])[0].lower() in {"1", "true", "yes"}:
                job["_release_stream_immediately"] = True
            job["progress_text"] = "正在停止任务..."
            job["_updated_monotonic"] = time.monotonic()
        self.send_json({"ok": True, "job_id": job_id, "status": "canceling"})

    def handle_job_stream(self, job_id: str, variant: str = "annotated") -> None:
        with JOBS_LOCK:
            exists = job_id in JOBS
        if not exists:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        last_frame_id = -1
        while True:
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                break
            id_key = "raw_frame_id" if variant == "raw" else "frame_id"
            bytes_key = "latest_raw_frame_bytes" if variant == "raw" else "latest_frame_bytes"
            frame_id = int(job.get(id_key, 0) or 0)
            frame_bytes = job.get(bytes_key, b"")
            if frame_bytes and frame_id != last_frame_id:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                    break
                last_frame_id = frame_id
            if job.get("status") in TERMINAL_JOB_STATES:
                break
            # When the browser opens the MJPEG URL before the first frame is
            # available, yield between checks.  Without this sleep the HTTP
            # handler spins at full speed while repeatedly taking JOBS_LOCK,
            # starving the WebRTC negotiation / decoder threads and making
            # the page-only test time out even though direct capture works.
            time.sleep(0.08)

    def handle_live_stream_test(self) -> None:
        form = self.parse_form()
        stream_url = validate_live_stream_url(field_value(form, "stream_url"))
        # The UI owns one preview subscription. Cancel abandoned preview jobs
        # before registering the replacement while leaving algorithm jobs and
        # their shared WebRTC reader untouched.
        with JOBS_LOCK:
            for existing in JOBS.values():
                if existing.get("algorithm") == "live_test" and existing.get("status") == "running":
                    existing["cancel_requested"] = True
                    existing["progress_text"] = "已由新的直播测试替换"
        job_id = uuid.uuid4().hex
        create_job(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "progress": 0,
                "progress_text": "正在连接直播流...",
                "stream_ready": False,
                "algorithm": "live_test",
            },
        )
        start_managed_job(job_id, run_live_test_job, (job_id, stream_url), uses_algorithm_slot=False)
        self.send_json({"ok": True, "job_id": job_id})

    def handle_model_classes_upload(self) -> None:
        form = self.parse_form()
        path_text = field_value(form, "model", "")
        if upload_items(form, "model_file"):
            model_path = save_upload(form, "model_file", path_text, "material_model")
        else:
            if not path_text.strip():
                raise ValueError("请指定模型文件。")
            model_path = resolve_model_path(path_text)
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        info = describe_model(str(model_path))
        info["supported"] = bool(info.get("ok")) and info.get("task") in MATERIAL_SUPPORTED_TASKS
        self.send_json(info)

    def handle_mock_stream_start(self) -> None:
        form = self.parse_form()
        result = start_mock_stream(field_value(form, "video", ""))
        if not result.get("ok"):
            self.send_json(result, 400)
            return
        self.send_json(result)

    def handle_mock_stream_stop(self) -> None:
        self.send_json(stop_mock_stream())

    def handle_material_count(self) -> None:
        self.start_material_count(False)

    def handle_material_count_live(self) -> None:
        self.start_material_count(True)

    def start_material_count(self, live_only: bool = False) -> None:
        form = self.parse_form()
        source_type = "live_stream" if live_only else field_value(form, "source_type", "single_image")
        if source_type not in {"single_image", "image_sequence", "video", "live_stream"}:
            raise ValueError(f"不支持的输入类型: {source_type}")
        sample_interval = max(1, _to_int(field_value(form, "sample_interval", "5"), 5))
        proximity_thresh = float(field_value(form, "proximity_thresh", "0.6"))
        appearance_thresh = float(field_value(form, "appearance_thresh", "0.85"))
        if not 0.0 <= proximity_thresh <= 1.0 or not 0.0 <= appearance_thresh <= 1.0:
            raise ValueError("ReID 空间接近阈值和外观相似度阈值必须在 0 到 1 之间。")
        # Java clients may submit each selected class as a separate multipart part.
        # Combine them so both that form and the web UI's single text field work.
        target_classes = "\n".join(field_values(form, "target_classes")).strip()
        if not target_classes:
            raise ValueError("请至少勾选一个需要计数的物资类别。")
        requested_classes = [name.strip() for name in re.split(r"[\n,，]+", target_classes) if name.strip()]

        model = save_upload(form, "model_file", field_value(form, "model", str(DEFAULT_MATERIAL_MODEL)), "material_model")
        if not model.exists():
            raise FileNotFoundError(f"模型文件不存在: {model}")
        # 类别以模型自身的 names 为准；模型读不出来时才退回内置候选表，
        # 这样 all_best.pt 之外的模型（例如普通检测模型）也能按自己的类别计数。
        model_info = describe_model(str(model))
        if model_info.get("ok"):
            if model_info.get("task") and model_info["task"] not in MATERIAL_SUPPORTED_TASKS:
                raise ValueError(
                    f"物资计数模型必须是 OBB 或目标检测模型，当前模型 task={model_info['task']}。"
                )
            available_classes = list(model_info.get("names") or [])
        else:
            available_classes = list(MATERIAL_CLASS_CANDIDATES)
        supported_by_key = {name.casefold(): name for name in available_classes}
        unsupported_classes = [name for name in requested_classes if name.casefold() not in supported_by_key]
        if unsupported_classes:
            raise ValueError(
                f"模型不包含物资类别：{', '.join(unsupported_classes)}。"
                f"模型可选类别：{', '.join(available_classes)}"
            )

        output_dir = ensure_dir(Path(field_value(form, "output_dir", str(DEFAULT_OUTPUT_DIR)).strip()))
        token = int(time.time() * 1000)
        if source_type == "live_stream":
            source: Path | str = validate_live_stream_url(field_value(form, "stream_url"))
            output_media = output_dir / f"material_count_live_{token}.mp4"
        elif source_type == "single_image":
            source = save_upload(form, "source_file", field_value(form, "source_path"), "material_image")
            output_media = output_dir / f"material_count_{token}.jpg"
        elif source_type == "video":
            source = save_upload(form, "source_file", field_value(form, "source_path"), "material_video")
            output_media = output_dir / f"material_count_{token}.mp4"
        else:
            source = save_upload_directory(form, "source_file", field_value(form, "source_path"), "material_sequence")
            if not source.is_dir():
                raise FileNotFoundError(f"图片序列文件夹不存在: {source}")
            output_media = output_dir / f"material_count_{token}.mp4"
        output_json = output_dir / f"material_count_{token}.json"
        device = resolve_device(field_value(form, "device", "0"))
        command = [
            sys.executable,
            str(APP_DIR / "material_count_cli.py"),
            "--source-type",
            source_type,
            "--source",
            "[直播地址已隐藏]" if source_type == "live_stream" else str(source),
            "--model",
            str(model),
            "--output-media",
            str(output_media),
            "--json",
            str(output_json),
            "--conf",
            field_value(form, "conf", "0.5"),
            "--track-high-thresh",
            field_value(form, "track_high_thresh", "0.3"),
            "--track-low-thresh",
            field_value(form, "track_low_thresh", "0.1"),
            "--new-track-thresh",
            field_value(form, "new_track_thresh", "0.4"),
            "--track-buffer",
            field_value(form, "track_buffer", "50"),
            "--match-thresh",
            field_value(form, "match_thresh", "0.25"),
            "--with-reid",
            "true" if bool_field(form, "with_reid", True) else "false",
            "--proximity-thresh",
            str(proximity_thresh),
            "--appearance-thresh",
            str(appearance_thresh),
            "--expected-counts",
            field_value(form, "expected_counts", ""),
            "--target-classes",
            target_classes,
            "--device",
            device,
            "--sample-interval",
            str(sample_interval),
        ]
        job_id = uuid.uuid4().hex
        tracker_args = {
            "track_high_thresh": float(field_value(form, "track_high_thresh", "0.3")),
            "track_low_thresh": float(field_value(form, "track_low_thresh", "0.1")),
            "new_track_thresh": float(field_value(form, "new_track_thresh", "0.4")),
            "track_buffer": int(field_value(form, "track_buffer", "50")),
            "match_thresh": float(field_value(form, "match_thresh", "0.25")),
            "with_reid": bool_field(form, "with_reid", True),
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": appearance_thresh,
        }
        command_text = " ".join(f'"{x}"' if " " in x else x for x in command)
        create_job(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "progress": None if source_type == "live_stream" else 0,
                "progress_text": "任务已启动",
                "command": command_text,
                "algorithm": "material_count",
                "_live_stream_url": str(source) if source_type == "live_stream" else "",
            },
        )
        start_managed_job(
            job_id,
            run_material_count_job,
            (
                job_id,
                source_type,
                source,
                model,
                output_media,
                output_json,
                tracker_args,
                field_value(form, "expected_counts", ""),
                target_classes,
                float(field_value(form, "conf", "0.5")),
                device,
                command,
                sample_interval,
            ),
        )
        self.send_json({"ok": True, "job_id": job_id, "command": command_text})

    def serve_static(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(WEB_DIR.resolve())) or not path.exists() or not path.is_file():
            self.send_error(404)
            return
        self.serve_file(path)

    def serve_file(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(PROJECT_DIR.resolve())) or not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200

        if range_header.startswith("bytes="):
            requested = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            start_text, _, end_text = requested.partition("-")
            try:
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else file_size - 1
                elif end_text:
                    suffix_len = int(end_text)
                    start = max(0, file_size - suffix_len)
                    end = file_size - 1
                if start < 0 or end < start or start >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                start = 0
                end = file_size - 1
                status = 200

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class SafetyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # In-memory jobs and semaphores cannot be shared by multiple processes.
    # Refuse a second server on the same port instead of routing requests to
    # independent queues and leaving jobs apparently stuck.
    allow_reuse_address = False
    request_queue_size = 64


def main() -> None:
    host = os.environ.get("SAFETY_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("SAFETY_WEB_PORT", "7860"))
    ensure_dir(DEFAULT_OUTPUT_DIR)
    ensure_dir(UPLOAD_DIR)
    RUNTIME_STOP_EVENT.clear()
    janitor = threading.Thread(target=job_janitor_loop, name="safety-job-janitor", daemon=True)
    janitor.start()
    server = SafetyHTTPServer((host, port), SafetyWebHandler)
    print(f"Safety web UI: http://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[web] 收到停止信号，正在安全关闭任务和直播连接...", flush=True)
    finally:
        server.server_close()
        shutdown_runtime()


if __name__ == "__main__":
    main()
