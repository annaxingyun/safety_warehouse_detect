"""Low-latency, reconnecting video-stream reader shared by WebUI jobs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    frame: np.ndarray
    captured_at: float
    fps: float
    generation: int


def is_whep_source(source: str) -> bool:
    """Return whether a URL is a WHEP-compatible WebRTC playback endpoint."""
    raw = str(source).strip().lower()
    if raw.startswith(("whep+http://", "whep+https://")):
        return True
    parsed = urlparse(raw)
    path_parts = parsed.path.rstrip("/").split("/")
    if "whep" in path_parts:
        return True

    # ZLMediaKit also exposes its WHEP-compatible player at
    # /index/api/webrtc?...&type=play. It is an SDP/aiortc endpoint, not a
    # media file URL, so OpenCV/FFmpeg cannot open it directly.
    query = parse_qs(parsed.query)
    request_types = {item.lower() for item in query.get("type", [])}
    return bool(path_parts and path_parts[-1] == "webrtc" and "play" in request_types)


def normalize_whep_url(source: str) -> str:
    """Convert the explicit whep+http(s) scheme to a request URL."""
    raw = str(source).strip()
    if raw.startswith("whep+https://"):
        return "https://" + raw.removeprefix("whep+https://")
    if raw.startswith("whep+http://"):
        return "http://" + raw.removeprefix("whep+http://")
    return raw


def preferred_whep_video_codecs(codecs: list[object]) -> list[object]:
    """Prefer H.264, retain RTX recovery, then keep VP8 as a fallback."""
    h264 = [codec for codec in codecs if str(getattr(codec, "mimeType", "")).lower() == "video/h264"]
    rtx = [codec for codec in codecs if str(getattr(codec, "mimeType", "")).lower() == "video/rtx"]
    vp8 = [codec for codec in codecs if str(getattr(codec, "mimeType", "")).lower() == "video/vp8"]
    return [*h264, *rtx, *vp8]


def sdp_video_source_ids(sdp: str) -> set[int]:
    """Find primary video SSRCs so an initial PLI need not wait for the first RTP packet."""
    sources: set[int] = set()
    retransmissions: set[int] = set()
    in_video = False
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
        if not in_video:
            continue
        try:
            if line.startswith("a=ssrc:"):
                sources.add(int(line.split(":", 1)[1].split()[0]))
            elif line.startswith("a=ssrc-group:FID "):
                primary, retransmission = map(int, line.split()[1:3])
                sources.add(primary)
                retransmissions.add(retransmission)
        except (ValueError, IndexError):
            continue
    return sources - retransmissions


def sdp_video_codec_names(sdp: str) -> list[str]:
    """Return supported primary codec names advertised in the SDP video section."""
    names: list[str] = []
    in_video = False
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
        if not in_video or not line.lower().startswith("a=rtpmap:"):
            continue
        codec_parts = line.split(None, 1)
        if len(codec_parts) != 2:
            continue
        codec_name = codec_parts[1].split("/", 1)[0].upper()
        if codec_name in {"H264", "VP8"} and codec_name not in names:
            names.append(codec_name)
    return names


def parse_webrtc_answer(body: str) -> tuple[str, str]:
    """Parse a standard raw-SDP answer or ZLMediaKit's JSON answer wrapper."""
    text = str(body).strip()
    if not text:
        raise RuntimeError("WebRTC 服务未返回 SDP answer")
    if text.startswith("v="):
        return text, "answer"
    if not text.startswith("{"):
        raise RuntimeError(f"WebRTC 服务返回了无法识别的应答：{text[:200]}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"WebRTC 服务返回的 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("WebRTC 服务返回的 JSON 不是对象。")

    code = payload.get("code", 0)
    if code not in {0, "0", None}:
        message = payload.get("msg") or payload.get("message") or "未知错误"
        raise RuntimeError(f"ZLMediaKit WebRTC 协商失败（code={code}）：{message}")
    sdp = str(payload.get("sdp") or "").strip()
    if not sdp:
        raise RuntimeError("ZLMediaKit WebRTC 应答中缺少 sdp 字段。")
    answer_type = str(payload.get("type") or "answer").strip().lower()
    if answer_type != "answer":
        raise RuntimeError(f"ZLMediaKit WebRTC 返回了非 answer 类型：{answer_type}")
    return sdp, answer_type


def strip_auxiliary_video_codecs(sdp: str, keep_rtx: bool = False) -> str:
    """Remove RTX / RED / ULPFEC payloads from the video section of an SDP answer.

    Some WHEP servers answer with retransmission codecs even when the client is
    receive-only. Older aiortc versions may then try to construct a decoder for
    ``video/rtx``. Keeping only primary video payloads makes those packets
    ignorable while preserving negotiated H.264 or VP8 primary streams.
    """
    newline = "\r\n" if "\r\n" in sdp else "\n"
    lines = sdp.replace("\r\n", "\n").split("\n")
    auxiliary_payloads: set[str] = set()
    in_video = False
    for line in lines:
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
        if not in_video or not line.lower().startswith("a=rtpmap:"):
            continue
        payload_and_codec = line[len("a=rtpmap:"):].split(None, 1)
        if len(payload_and_codec) != 2:
            continue
        codec_name = payload_and_codec[1].split("/", 1)[0].lower()
        if codec_name in ({"red", "ulpfec", "flexfec-03"} if keep_rtx else {"rtx", "red", "ulpfec", "flexfec-03"}):
            auxiliary_payloads.add(payload_and_codec[0])

    if not auxiliary_payloads:
        return sdp

    filtered: list[str] = []
    in_video = False
    for line in lines:
        if line.startswith("m="):
            in_video = line.startswith("m=video ")
            if in_video:
                parts = line.split()
                line = " ".join([*parts[:3], *(part for part in parts[3:] if part not in auxiliary_payloads)])
        if in_video and line.startswith(("a=rtpmap:", "a=fmtp:", "a=rtcp-fb:")):
            payload = line.split(":", 1)[1].split(None, 1)[0]
            if payload in auxiliary_payloads:
                continue
        filtered.append(line)
    return newline.join(filtered)


async def request_whep_keyframe(receiver: object, source_ids: Optional[set[int]] = None) -> bool:
    """Best-effort PLI request for a WHEP receiver which joined mid-stream.

    aiortc has no public keyframe-request API. Its receiver does expose the
    synchronization sources publicly and uses ``_send_rtcp_pli`` internally
    when its jitter buffer detects picture loss, so use the same path when no
    decodable first frame has arrived yet.
    """
    send_pli = getattr(receiver, "_send_rtcp_pli", None)
    get_sources = getattr(receiver, "getSynchronizationSources", None)
    if not callable(send_pli):
        return False
    try:
        active_sources = {
            int(source.source)
            for source in (get_sources() if callable(get_sources) else [])
            if getattr(source, "source", None) is not None
        }
        active_sources.update(source_ids or set())
        for source_id in active_sources:
            await send_pli(source_id)
        return bool(active_sources)
    except Exception:
        # This is a compatibility aid for aiortc's private PLI method. A
        # version mismatch must not terminate the otherwise valid session.
        return False


def install_aiortc_rtx_compatibility() -> bool:
    """Backport aiortc's RTX codec reassignment fix for releases before 1.11.

    aiortc 1.10 unwraps an RTX packet but leaves its codec set to video/rtx,
    which can feed raw RTP payload into H.264 and later tries to create an RTX
    decoder. Newer aiortc assigns the associated primary codec after unwrap.
    """
    import aiortc
    from aiortc import RTCRtpReceiver
    from aiortc.codecs import is_rtx
    from aiortc.rtp import unwrap_rtx

    version_parts = []
    for part in str(getattr(aiortc, "__version__", "0")).split(".")[:2]:
        try:
            version_parts.append(int(part))
        except ValueError:
            version_parts.append(0)
    if tuple((version_parts + [0, 0])[:2]) >= (1, 11):
        return False
    if getattr(RTCRtpReceiver, "_safety_rtx_compat_installed", False):
        return True

    original_handler = RTCRtpReceiver._handle_rtp_packet

    async def handle_rtp_packet_with_primary_codec(self, packet, arrival_time_ms):
        codecs = getattr(self, "_RTCRtpReceiver__codecs", {})
        codec = codecs.get(packet.payload_type)
        if codec is not None and is_rtx(codec):
            rtx_ssrc = getattr(self, "_RTCRtpReceiver__rtx_ssrc", {})
            original_ssrc = rtx_ssrc.get(packet.ssrc)
            apt = codec.parameters.get("apt")
            if (
                original_ssrc is not None
                and len(packet.payload) >= 2
                and isinstance(apt, int)
                and apt in codecs
            ):
                packet = unwrap_rtx(packet, payload_type=apt, ssrc=original_ssrc)
        return await original_handler(self, packet, arrival_time_ms)

    RTCRtpReceiver._handle_rtp_packet = handle_rtp_packet_with_primary_codec
    RTCRtpReceiver._safety_rtx_compat_installed = True
    return True


def install_aiortc_decoder_resilience() -> bool:
    """Keep aiortc's decoder worker alive after a recoverable bad access unit.

    aiortc 1.10 lets every exception raised by ``decoder.decode`` escape its
    worker thread.  The RTP / ICE connection then remains connected while the
    remote track can never produce another frame, which looks like a frozen
    stream and eventually causes an unnecessary WHEP reconnect.  A malformed
    or incomplete H.264 access unit is expected during packet loss, so discard
    that unit and retain the decoder (including its cached SPS/PPS) for the next
    keyframe instead of terminating the worker.
    """
    import aiortc.rtcrtpreceiver as receiver_module

    if getattr(receiver_module, "_safety_decoder_resilience_installed", False):
        return True

    def resilient_decoder_worker(loop, input_q, output_q) -> None:
        codec_name = None
        decoder = None

        while True:
            task = input_q.get()
            if task is None:
                asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
                break
            codec, encoded_frame = task

            try:
                input_q._safety_decode_attempt_count = int(getattr(input_q, "_safety_decode_attempt_count", 0)) + 1
                if codec.name != codec_name:
                    decoder = receiver_module.get_decoder(codec)
                    codec_name = codec.name
                frames = decoder.decode(encoded_frame)
            except Exception:
                # Do not recreate the H.264 decoder here: the current context
                # may be the only place holding SPS/PPS from the last IDR. RTP
                # loss is handled by NACK/RTX and periodic PLI in the session
                # loop, so the next complete keyframe can recover naturally.
                input_q._safety_decode_error_count = int(
                    getattr(input_q, "_safety_decode_error_count", 0)
                ) + 1
                continue

            if not frames:
                # PyAV/aiortc's H.264 decoder can swallow a codec error and
                # return [] instead of raising; include these units in diagnostics.
                input_q._safety_empty_decode_count = int(getattr(input_q, "_safety_empty_decode_count", 0)) + 1
            for frame in frames:
                asyncio.run_coroutine_threadsafe(output_q.put(frame), loop)

        if decoder is not None:
            del decoder

    receiver_module.decoder_worker = resilient_decoder_worker
    receiver_module._safety_decoder_resilience_installed = True
    return True


def configure_webrtc_video_jitter_buffer(receiver: object, capacity: int = 4096) -> int:
    """Increase aiortc's video RTP buffer for large H.264 IDR access units.

    aiortc 1.10 uses a 128-packet video buffer. A high-resolution IDR frame can
    exceed that size, causing the FU-A start packet to be discarded and PyAV to
    receive an access unit without an Annex-B start code. The receiver has not
    started when this helper is called, so replacing the empty buffer is safe.
    """
    from aiortc.jitterbuffer import JitterBuffer

    requested = max(128, int(capacity))
    target_capacity = 1 << (requested - 1).bit_length()
    attribute = "_RTCRtpReceiver__jitter_buffer"
    current = getattr(receiver, attribute, None)
    current_capacity = int(getattr(current, "capacity", 0) or 0)
    if current_capacity >= target_capacity:
        return current_capacity
    setattr(receiver, attribute, JitterBuffer(capacity=target_capacity, is_video=True))
    return target_capacity


def _observe_asyncio_task_exception(task: asyncio.Task) -> None:
    """Mark a background receive exception as observed without consuming its result."""
    if task.cancelled():
        return
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


def reconnect_backoff_seconds(failure_streak: int, base_delay: float = 1.0, max_delay: float = 8.0) -> float:
    """Return a bounded exponential delay for consecutive startup failures."""
    base = max(0.1, float(base_delay))
    maximum = max(base, float(max_delay))
    exponent = min(max(0, int(failure_streak) - 1), 6)
    return min(maximum, base * (2**exponent))


async def receiver_packet_counts(receiver: object) -> tuple[int, int]:
    """Return aggregate received/lost RTP packet counters without breaking capture."""
    get_stats = getattr(receiver, "getStats", None)
    if not callable(get_stats):
        return 0, 0
    try:
        report = await get_stats()
        values = report.values() if hasattr(report, "values") else report
        received = 0
        lost = 0
        for stat in values:
            if getattr(stat, "type", "") != "inbound-rtp" or getattr(stat, "kind", "video") != "video":
                continue
            received += max(0, int(getattr(stat, "packetsReceived", 0) or 0))
            lost += max(0, int(getattr(stat, "packetsLost", 0) or 0))
        return received, lost
    except Exception:
        # Statistics are diagnostic only and must never interrupt the stream.
        return 0, 0


class LiveStreamReader:
    """Continuously keep only the newest frame so slow inference cannot build a queue."""

    def __init__(
        self,
        source: str,
        frame_callback: Optional[Callable[[FramePacket], None]] = None,
        reconnect_delay: float = 1.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        self.source = str(source).strip()
        if not self.source:
            raise ValueError("直播地址不能为空。")
        self.frame_callback = frame_callback
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.open_timeout_ms = max(100, int(open_timeout_ms))
        self.read_timeout_ms = max(100, int(read_timeout_ms))
        self._condition = threading.Condition()
        self._frame_callbacks: list[Callable[[FramePacket], None]] = []
        if frame_callback is not None:
            self._frame_callbacks.append(frame_callback)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[FramePacket] = None
        self._error = ""
        self._last_failure = ""
        self._opened_once = False
        self._connected = False
        self._recovering = False
        self._reconnect_count = 0
        self._last_frame_at = 0.0
        self._whep_frame_id = 0
        self._transport_diagnostics: dict[str, object] = {}

    @property
    def error(self) -> str:
        with self._condition:
            return self._error

    @property
    def last_failure(self) -> str:
        """Return the latest completed connection failure, excluding progress text."""
        with self._condition:
            return self._last_failure

    @property
    def startup_timeout_seconds(self) -> float:
        """Maximum time consumers should allow for the first decoded frame.

        WebRTC startup includes ICE gathering, HTTP negotiation, video-track setup
        and a keyframe wait.  It therefore needs more time than a regular
        media input and must not be cut off by the old fixed 12-second timeout.
        """
        transport_timeout = max(self.open_timeout_ms, self.read_timeout_ms) / 1000.0
        return max(60.0, transport_timeout * 8.0 + 30.0)

    @property
    def opened_once(self) -> bool:
        with self._condition:
            return self._opened_once

    @property
    def connected(self) -> bool:
        """Whether the current WebRTC generation is actively producing frames."""
        with self._condition:
            return self._connected

    @property
    def recovering(self) -> bool:
        with self._condition:
            return self._recovering

    @property
    def reconnect_count(self) -> int:
        with self._condition:
            return self._reconnect_count

    @property
    def last_frame_age(self) -> Optional[float]:
        with self._condition:
            captured_at = self._last_frame_at
        if captured_at <= 0:
            return None
        return max(0.0, time.monotonic() - captured_at)

    def health_snapshot(self) -> dict[str, object]:
        """Return sanitized runtime health without exposing the stream URL."""
        with self._condition:
            age = None if self._last_frame_at <= 0 else max(0.0, time.monotonic() - self._last_frame_at)
            return {
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "opened_once": self._opened_once,
                "connected": self._connected,
                "recovering": self._recovering,
                "frame_id": self._whep_frame_id,
                "last_frame_age_seconds": round(age, 3) if age is not None else None,
                "reconnect_count": self._reconnect_count,
                **self._transport_diagnostics,
            }

    def start(self) -> "LiveStreamReader":
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="live-stream-reader", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._connected = False
            self._recovering = False
            self._condition.notify_all()

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))

    def add_frame_callback(self, callback: Optional[Callable[[FramePacket], None]]) -> None:
        if callback is None:
            return
        with self._condition:
            if callback not in self._frame_callbacks:
                self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Optional[Callable[[FramePacket], None]]) -> None:
        if callback is None:
            return
        with self._condition:
            self._frame_callbacks = [item for item in self._frame_callbacks if item is not callback]

    def wait_for_frame(self, after_frame_id: int = -1, timeout: float = 1.0) -> Optional[FramePacket]:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._stop_event.is_set():
                if self._latest is not None and self._latest.frame_id > after_frame_id:
                    packet = self._latest
                    return FramePacket(
                        packet.frame_id,
                        packet.frame.copy(),
                        packet.captured_at,
                        packet.fps,
                        packet.generation,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def _publish_frame(self, frame_id: int, frame: np.ndarray, fps: float, generation: int) -> None:
        packet = FramePacket(frame_id, frame, time.monotonic(), fps, generation)
        with self._condition:
            self._latest = packet
            self._error = ""
            self._last_failure = ""
            self._connected = True
            self._recovering = False
            self._last_frame_at = packet.captured_at
            callbacks = list(self._frame_callbacks)
            self._condition.notify_all()
        for callback in callbacks:
            try:
                callback(packet)
            except Exception:
                # A preview consumer must never terminate capture.
                pass

    def _set_progress(self, message: str) -> None:
        """Expose the current connection stage without marking it as a failure."""
        with self._condition:
            self._error = message
            self._condition.notify_all()

    def _mark_recovering(self, message: str) -> None:
        with self._condition:
            self._connected = False
            self._recovering = True
            self._error = message
            self._condition.notify_all()

    async def _run_whep_session(self, generation: int) -> None:
        """Receive a WHEP video track and publish its newest decoded frame."""
        with self._condition:
            self._transport_diagnostics = {}
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpReceiver
        except ImportError as exc:
            raise RuntimeError("WHEP 拉流需要安装 aiortc：pip install -r requirements.txt") from exc

        rtx_compatibility_enabled = install_aiortc_rtx_compatibility()
        decoder_resilience_enabled = install_aiortc_decoder_resilience()

        endpoint = normalize_whep_url(self.source)
        timeout_seconds = max(1.0, self.open_timeout_ms / 1000.0)
        import certifi

        configured_ca = str(os.environ.get("SAFETY_CA_BUNDLE", "")).strip()
        ca_bundle = configured_ca if configured_ca and os.path.isfile(configured_ca) else certifi.where()
        ssl_context = ssl.create_default_context(cafile=ca_bundle)

        async def whep_request(method: str, url: str, body: Optional[bytes] = None) -> tuple[int, str, str]:
            def send() -> tuple[int, str, str]:
                private_zlm_api = urlparse(url).path.rstrip("/").endswith("/webrtc")
                headers = {
                    "Accept": "application/json, application/sdp, text/plain",
                    "User-Agent": "safety-monitor/1.0",
                }
                if body is not None:
                    headers["Content-Type"] = "text/plain;charset=UTF-8" if private_zlm_api else "application/sdp"
                request = urllib.request.Request(url, data=body, headers=headers, method=method)
                try:
                    with urllib.request.urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                        return int(response.status), response.read().decode("utf-8"), response.headers.get("Location", "")
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"WebRTC 信令 HTTP {method} 请求在 {timeout_seconds:g} 秒内未收到响应；"
                        "请检查 ZLMediaKit HTTP 端口、app/stream 名称及 on_play 等 Hook 是否阻塞。"
                    ) from exc
                except urllib.error.HTTPError as exc:
                    return int(exc.code), exc.read().decode("utf-8", errors="replace"), exc.headers.get("Location", "")

            return await asyncio.to_thread(send)

        pc = RTCPeerConnection()
        video_tracks = []
        video_ready = asyncio.Event()
        session_url = ""

        def connection_diagnostic() -> str:
            return (
                f"connection={getattr(pc, 'connectionState', 'unknown')}, "
                f"ice={getattr(pc, 'iceConnectionState', 'unknown')}, "
                f"signaling={getattr(pc, 'signalingState', 'unknown')}"
            )

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "video":
                video_tracks.append(track)
                video_ready.set()

        try:
            # Keep RTX enabled so NACK recovery can repair missing H.264 FU-A
            # fragments. The compatibility patch above makes aiortc 1.10 map
            # retransmissions back to their associated primary codec.
            transceiver = pc.addTransceiver("video", direction="recvonly")
            receiver = transceiver.receiver
            jitter_capacity = configure_webrtc_video_jitter_buffer(receiver)
            preferred_codecs = preferred_whep_video_codecs(RTCRtpReceiver.getCapabilities("video").codecs)
            if not preferred_codecs:
                raise RuntimeError("本机 aiortc 未提供 H.264 或 VP8 解码能力，无法连接 WHEP 直播流。")
            transceiver.setCodecPreferences(preferred_codecs)
            self._set_progress(
                f"Python 后端正在创建 WHEP SDP Offer（RTP 缓冲 {jitter_capacity} 包，"
                f"RTX 恢复{'已兼容' if rtx_compatibility_enabled else '已启用'}，"
                f"坏帧隔离{'已启用' if decoder_resilience_enabled else '可用'}）。"
            )
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # Submit a fully gathered offer. This works with standard one-shot WHEP
            # endpoints and avoids requiring a separate trickle-ICE implementation.
            gathering_deadline = time.monotonic() + timeout_seconds
            while pc.iceGatheringState != "complete" and time.monotonic() < gathering_deadline:
                await asyncio.sleep(0.05)

            local = pc.localDescription
            if local is None:
                raise RuntimeError("无法创建 WHEP SDP offer")
            self._set_progress("Python 后端已生成 SDP Offer，正在请求 WHEP 服务。")
            status, answer_sdp, location = await whep_request("POST", endpoint, local.sdp.encode("utf-8"))
            if status not in {200, 201}:
                raise RuntimeError(f"WHEP 协商失败（HTTP {status}）：{answer_sdp[:300]}")
            answer_sdp, answer_type = parse_webrtc_answer(answer_sdp)
            # Retain RTX for packet-loss recovery, but remove unsupported FEC
            # payloads which do not have decoders in this receive pipeline.
            answer_sdp = strip_auxiliary_video_codecs(answer_sdp, keep_rtx=True)
            video_source_ids = sdp_video_source_ids(answer_sdp)
            negotiated_codecs = sdp_video_codec_names(answer_sdp)
            if not negotiated_codecs:
                raise RuntimeError("WHEP 服务端未协商可解码的 H.264 或 VP8 视频流。")
            codec_text = "/".join(negotiated_codecs)
            session_url = urljoin(endpoint, location) if location else ""

            self._set_progress(f"WHEP SDP 协商成功（{codec_text}），正在建立视频轨。")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type=answer_type))
            try:
                await asyncio.wait_for(video_ready.wait(), timeout=timeout_seconds)
            except TimeoutError as exc:
                raise RuntimeError("WHEP 协商成功，但未收到视频轨") from exc

            track = video_tracks[0]
            first_frame_received = False
            # A video track is signaled before ICE/DTLS is necessarily connected.
            # Give transport setup and first-keyframe decode separate budgets.
            transport_deadline = time.monotonic() + max(15.0, timeout_seconds * 3.0)
            first_frame_deadline: Optional[float] = None
            last_frame_at = time.monotonic()
            # Give RTX/NACK and one requested IDR enough time to repair a short
            # disturbance before rebuilding ICE / DTLS from scratch.
            stalled_frame_timeout = max(15.0, self.read_timeout_ms / 1000.0 * 3.0)
            disconnected_since: Optional[float] = None
            last_keyframe_request_at = 0.0
            last_stats_at = 0.0
            last_packets_lost: Optional[int] = None
            last_decoder_errors = 0
            pending_loss_burst = 0
            consecutive_bad_frames = 0
            recv_task: Optional[asyncio.Task] = None
            keyframe_requests = 0
            self._set_progress(f"WHEP 视频轨已协商（{codec_text}），正在建立 ICE/DTLS 视频传输。")

            async def request_keyframe_if_due(force: bool = False) -> bool:
                nonlocal last_keyframe_request_at, keyframe_requests
                now = time.monotonic()
                # Repeated PLIs generate large IDRs and can make congestion
                # worse, so rate-limit them while allowing an immediate first
                # request after joining mid-stream.
                if not force and now - last_keyframe_request_at < 2.0:
                    return False
                requested = await request_whep_keyframe(receiver, video_source_ids)
                if requested:
                    last_keyframe_request_at = now
                    keyframe_requests += 1
                return requested

            await request_keyframe_if_due(force=True)

            try:
                while not self._stop_event.is_set():
                    if recv_task is None:
                        recv_task = asyncio.create_task(track.recv())
                        recv_task.add_done_callback(_observe_asyncio_task_exception)
                    try:
                        done, _ = await asyncio.wait(
                            {recv_task}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                        )
                        if recv_task not in done:
                            now = time.monotonic()
                            decoder_thread = getattr(receiver, "_RTCRtpReceiver__decoder_thread", None)
                            if decoder_thread is not None and not decoder_thread.is_alive():
                                raise RuntimeError("WebRTC 解码线程异常退出，正在重建解码通道")
                            connection_state = str(getattr(pc, "connectionState", "unknown"))
                            ice_state = str(getattr(pc, "iceConnectionState", "unknown"))
                            if connection_state in {"failed", "closed"} or ice_state in {"failed", "closed"}:
                                raise RuntimeError(f"WebRTC 传输已失效（{connection_diagnostic()}）")
                            if connection_state == "disconnected" or ice_state == "disconnected":
                                if disconnected_since is None:
                                    disconnected_since = now
                                elif now - disconnected_since >= 3.0:
                                    raise RuntimeError(f"WebRTC 传输断开超过 3 秒（{connection_diagnostic()}）")
                            else:
                                disconnected_since = None

                            await request_keyframe_if_due()
                            received, lost = await receiver_packet_counts(receiver)
                            decoder_queue = getattr(receiver, "_RTCRtpReceiver__decoder_queue", None)
                            errors = int(getattr(decoder_queue, "_safety_decode_error_count", 0))
                            empty_units = int(getattr(decoder_queue, "_safety_empty_decode_count", 0))
                            diagnostic = (f"connection={connection_state}, ice={ice_state}, "
                                          f"RTP received={received}, lost={lost}, "
                                          f"decode_errors={errors}, decode_empty={empty_units}, PLI={keyframe_requests}")
                            with self._condition:
                                self._transport_diagnostics = {
                                    "connection_state": connection_state, "ice_state": ice_state,
                                    "rtp_packets_received": received, "rtp_packets_lost": lost,
                                    "decode_errors": errors, "decode_empty_units": empty_units,
                                    "keyframe_requests": keyframe_requests,
                                }
                            if not first_frame_received:
                                if connection_state != "connected":
                                    self._set_progress(f"WHEP 正在建立 ICE/DTLS 视频传输（{diagnostic}）。")
                                    if now >= transport_deadline:
                                        raise RuntimeError(f"WHEP 信令成功，但 ICE/DTLS 连接超时；请检查媒体服务器公告 IP 和 UDP 端口（{diagnostic}）。")
                                    continue
                                if first_frame_deadline is None:
                                    first_frame_deadline = now + max(30.0, timeout_seconds * 4.0)
                                if received == 0:
                                    self._set_progress(f"WHEP 传输已连接，但未收到视频 RTP 包；请检查推流和媒体端口（{diagnostic}）。")
                                else:
                                    self._set_progress(f"WHEP 已收到视频 RTP，正在等待可解码关键帧（{diagnostic}）。")
                                if now >= first_frame_deadline:
                                    raise RuntimeError(
                                        f"WHEP 已建立 {codec_text} 视频轨，但等待可解码关键帧超时；"
                                        + ("未收到视频 RTP，请检查媒体服务器公告 IP、UDP 端口和推流状态。" if received == 0 else
                                           "请让推流端响应 RTCP PLI；H.264还需在IDR帧前重复发送SPS/PPS。")
                                        + f"（{diagnostic}）"
                                    )
                            elif now - last_frame_at >= stalled_frame_timeout:
                                raise RuntimeError(
                                    f"WHEP {codec_text} 已连续 {stalled_frame_timeout:g} 秒未收到视频帧，"
                                    "已请求关键帧但尚未恢复。"
                                )
                            continue
                        completed_task = recv_task
                        recv_task = None
                        frame = completed_task.result()
                        last_frame_at = time.monotonic()
                    except Exception as exc:
                        raise RuntimeError(
                            f"WHEP {codec_text} 视频轨结束，正在重连（{connection_diagnostic()}）："
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    now = time.monotonic()
                    if now - last_stats_at >= 1.0:
                        _, packets_lost = await receiver_packet_counts(receiver)
                        decoder_queue = getattr(receiver, "_RTCRtpReceiver__decoder_queue", None)
                        decoder_errors = int(getattr(decoder_queue, "_safety_decode_error_count", 0) or 0)
                        if last_packets_lost is not None and packets_lost > last_packets_lost:
                            pending_loss_burst += packets_lost - last_packets_lost
                        last_packets_lost = packets_lost
                        last_stats_at = now
                        # NACK/RTX repairs isolated losses. A burst is more
                        # likely to leave H.264 reference pictures damaged, so
                        # request one fresh IDR without restarting the session.
                        if pending_loss_burst >= 3 and await request_keyframe_if_due():
                            pending_loss_burst = 0
                        if decoder_errors > last_decoder_errors:
                            # A malformed H.264 access unit was isolated by the
                            # resilient worker. Ask for a clean IDR while
                            # preserving this WHEP / ICE session and decoder
                            # context, rather than freezing or reconnecting.
                            await request_keyframe_if_due()
                        last_decoder_errors = decoder_errors

                    try:
                        image = frame.to_ndarray(format="bgr24")
                        if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
                            raise ValueError(f"无效视频帧尺寸: {image.shape}")
                        consecutive_bad_frames = 0
                    except Exception as exc:
                        consecutive_bad_frames += 1
                        await request_keyframe_if_due()
                        if consecutive_bad_frames >= 5:
                            raise RuntimeError(f"连续 {consecutive_bad_frames} 帧转换失败: {exc}") from exc
                        continue

                    if not first_frame_received:
                        first_frame_received = True
                        with self._condition:
                            # A negotiated track is not considered open until
                            # the decoder has produced a real frame.
                            self._opened_once = True
                            self._error = ""
                            self._condition.notify_all()
                    self._whep_frame_id += 1
                    self._publish_frame(self._whep_frame_id, image, 25.0, generation)
            finally:
                if recv_task is not None:
                    if not recv_task.done():
                        recv_task.cancel()
                    # A remote track can finish with MediaStreamError in the
                    # small window between done() and cancellation. Always
                    # await the task once so asyncio considers its exception
                    # retrieved and no orphan task leaks into the next session.
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await recv_task

        finally:
            # Always release the server-side WHEP resource, including decoder,
            # timeout and cancellation failures. Leaked sessions make repeated
            # tests progressively less reliable on resource-constrained ZLM.
            if session_url:
                with contextlib.suppress(Exception):
                    await whep_request("DELETE", session_url)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pc.close(), timeout=3.0)
            # Let aiortc's receiver/decoder callbacks observe their terminal
            # exceptions before asyncio tears down this session loop.
            await asyncio.sleep(0)

    def _run_whep(self) -> None:
        generation = 0
        failure_streak = 0
        while not self._stop_event.is_set():
            generation += 1
            frame_id_before_session = self._whep_frame_id
            try:
                asyncio.run(self._run_whep_session(generation))
            except Exception as exc:
                produced_frames = self._whep_frame_id > frame_id_before_session
                failure_streak = 0 if produced_frames else failure_streak + 1
                delay = reconnect_backoff_seconds(failure_streak, self.reconnect_delay)
                with self._condition:
                    self._connected = False
                    self._recovering = True
                    self._reconnect_count += 1
                    self._error = f"WHEP 直播流暂时中断，{delay:g} 秒后自动重连: {exc}"
                    self._last_failure = self._error
                    self._condition.notify_all()
            else:
                delay = self.reconnect_delay
            if not self._stop_event.is_set():
                self._stop_event.wait(delay)

    def _run(self) -> None:
        # Every live input in this application is a WebRTC playback endpoint.
        # Never fall back to OpenCV/FFmpeg based on URL spelling.
        self._run_whep()


@dataclass
class _SharedLiveReaderEntry:
    reader: LiveStreamReader
    references: int = 0
    idle_timer: Optional[threading.Timer] = None
    callback_leases: set[int] = field(default_factory=set)
    anonymous_references: int = 0


_SHARED_LIVE_READERS: dict[str, _SharedLiveReaderEntry] = {}
_SHARED_LIVE_READERS_LOCK = threading.Lock()


def _shared_stream_key(source: str) -> str:
    return normalize_whep_url(str(source).strip())


def acquire_shared_live_reader(
    source: str,
    frame_callback: Optional[Callable[[FramePacket], None]] = None,
) -> LiveStreamReader:
    """Acquire one process-wide WebRTC reader shared by preview and algorithms."""
    key = _shared_stream_key(source)
    with _SHARED_LIVE_READERS_LOCK:
        entry = _SHARED_LIVE_READERS.get(key)
        if entry is None:
            entry = _SharedLiveReaderEntry(reader=LiveStreamReader(key))
            _SHARED_LIVE_READERS[key] = entry
            entry.reader.start()
        else:
            snapshot = entry.reader.health_snapshot() if hasattr(entry.reader, "health_snapshot") else {}
            if snapshot.get("thread_alive") is False:
                # Recover a registry entry whose capture thread terminated
                # unexpectedly instead of handing every new consumer a dead
                # reader until the idle cleanup timer eventually removes it.
                entry.reader.start()
        if entry.idle_timer is not None:
            entry.idle_timer.cancel()
            entry.idle_timer = None
        if frame_callback is None:
            entry.anonymous_references += 1
            entry.references += 1
        else:
            callback_id = id(frame_callback)
            if callback_id not in entry.callback_leases:
                entry.callback_leases.add(callback_id)
                entry.references += 1
        entry.reader.add_frame_callback(frame_callback)
        return entry.reader


def release_shared_live_reader(
    source: str,
    reader: LiveStreamReader,
    frame_callback: Optional[Callable[[FramePacket], None]] = None,
    idle_seconds: float = 10.0,
) -> None:
    """Release a shared reader after a grace period to support seamless handoff."""
    key = _shared_stream_key(source)

    def stop_if_still_idle() -> None:
        should_stop = False
        with _SHARED_LIVE_READERS_LOCK:
            current = _SHARED_LIVE_READERS.get(key)
            if current is entry and current.references == 0:
                _SHARED_LIVE_READERS.pop(key, None)
                current.idle_timer = None
                should_stop = True
        if should_stop:
            reader.stop()
            reader.join(timeout=max(8.0, reader.open_timeout_ms / 1000.0 + 3.0))

    remove_callback = False
    with _SHARED_LIVE_READERS_LOCK:
        entry = _SHARED_LIVE_READERS.get(key)
        if entry is None or entry.reader is not reader:
            return
        if frame_callback is None:
            if entry.anonymous_references <= 0:
                return
            entry.anonymous_references -= 1
        else:
            callback_id = id(frame_callback)
            if callback_id not in entry.callback_leases:
                return
            entry.callback_leases.remove(callback_id)
            remove_callback = True
        entry.references -= 1
        if entry.references > 0:
            schedule_cleanup = False
        else:
            entry.references = 0
            if entry.idle_timer is not None:
                entry.idle_timer.cancel()
            entry.idle_timer = threading.Timer(max(0.0, float(idle_seconds)), stop_if_still_idle)
            entry.idle_timer.daemon = True
            entry.idle_timer.start()
            schedule_cleanup = True
    if remove_callback:
        reader.remove_frame_callback(frame_callback)
    if not schedule_cleanup:
        return


def shared_live_reader_health() -> dict[str, object]:
    """Return aggregate process-wide reader health for service diagnostics."""
    with _SHARED_LIVE_READERS_LOCK:
        entries = list(_SHARED_LIVE_READERS.values())
        snapshots = [
            {
                **entry.reader.health_snapshot(),
                "references": entry.references,
                "idle_cleanup_pending": entry.idle_timer is not None,
            }
            for entry in entries
        ]
    return {
        "reader_count": len(snapshots),
        "connected_count": sum(bool(item["connected"]) for item in snapshots),
        "recovering_count": sum(bool(item["recovering"]) for item in snapshots),
        "readers": snapshots,
    }


def shared_live_reader_is_ready(source: str, max_frame_age: float = 3.0) -> bool:
    """Return whether an active shared reader is producing recent frames."""
    key = _shared_stream_key(source)
    with _SHARED_LIVE_READERS_LOCK:
        entry = _SHARED_LIVE_READERS.get(key)
        if entry is None or entry.references <= 0:
            return False
        snapshot = entry.reader.health_snapshot()
    age = snapshot.get("last_frame_age_seconds")
    return bool(
        snapshot.get("thread_alive")
        and snapshot.get("opened_once")
        and snapshot.get("connected")
        and not snapshot.get("recovering")
        and age is not None
        and float(age) <= max(0.1, float(max_frame_age))
    )


def stop_all_shared_live_readers(join_timeout: float = 8.0) -> int:
    """Stop and remove every shared reader during controlled service shutdown."""
    with _SHARED_LIVE_READERS_LOCK:
        entries = list(_SHARED_LIVE_READERS.values())
        _SHARED_LIVE_READERS.clear()
        for entry in entries:
            if entry.idle_timer is not None:
                entry.idle_timer.cancel()
                entry.idle_timer = None
    for entry in entries:
        entry.reader.stop()
    for entry in entries:
        entry.reader.join(timeout=max(0.0, float(join_timeout)))
    return len(entries)


__all__ = [
    "FramePacket",
    "LiveStreamReader",
    "acquire_shared_live_reader",
    "configure_webrtc_video_jitter_buffer",
    "is_whep_source",
    "install_aiortc_rtx_compatibility",
    "install_aiortc_decoder_resilience",
    "normalize_whep_url",
    "preferred_whep_video_codecs",
    "parse_webrtc_answer",
    "receiver_packet_counts",
    "reconnect_backoff_seconds",
    "request_whep_keyframe",
    "release_shared_live_reader",
    "shared_live_reader_health",
    "shared_live_reader_is_ready",
    "sdp_video_codec_names",
    "sdp_video_source_ids",
    "strip_auxiliary_video_codecs",
    "stop_all_shared_live_readers",
]
