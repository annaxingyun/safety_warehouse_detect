#!/usr/bin/env python3
"""用本地视频文件模拟一路 WHEP 直播流，供安全监测的直播输入拉流检测。

视频文件在这里扮演摄像头：脚本按原始帧率解码、循环播放，并通过 WebRTC
（WHEP 信令）输出 H.264 或 VP8 视频轨。安全监测后台仍然走真实的
SDP/ICE/DTLS/RTP 拉流链路，和接入真实摄像头、NVR 或流媒体服务器完全一致。

不依赖 ffmpeg 或 MediaMTX，只使用项目已有的 PyAV、aiortc 和 aiohttp。

用法：

    # 1. 启动模拟摄像头（默认监听 127.0.0.1:8889，地址 /whep）
    python safety_tools/mock_live_camera.py --video demo.mp4

    # 2. WebUI 选择“直播流”，地址填：
    #    http://127.0.0.1:8889/whep
    #    先点“测试 Python 后端拉流”，画面连续后再“开始检测”

可选：

    --host 0.0.0.0          允许局域网其他电脑拉流
    --port 8889             监听端口
    --path whep             WHEP 路径，最终地址为 http://host:port/<path>
    --fps 25                覆盖视频原始帧率
    --width 1280 --height 720
                            输出画面的最大尺寸（等比缩放，默认不超过 720p）
    --no-resize             保持视频原始分辨率
    --codec vp8             改用 VP8（默认 H.264）
    --bitrate 2000000       目标码率，单位 bit/s
    --no-loop               播放一次后结束（默认循环）
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

try:
    import av
    from aiortc import (
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
        RTCRtpSender,
        VideoStreamTrack,
    )
    from aiortc.mediastreams import MediaStreamError
    from aiohttp import web
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "模拟直播需要 PyAV、aiortc 和 aiohttp，请先执行：\n"
        "    pip install -r requirements.txt\n"
        f"缺失依赖：{exc}"
    ) from exc


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8889
DEFAULT_PATH = "whep"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS_FALLBACK = 25.0
MAX_FPS = 60.0
ICE_GATHER_TIMEOUT = 2.0

# 会话看门狗：拉流端进程被强杀时 UDP 端口会直接消失，aiortc 未必及时把
# connectionState 置为 failed，编码线程仍会按帧率空转。用 RTCP 接收者报告
# 的到达时间判断对端是否还在：正常客户端每 0.5~1.5 秒回一次 RR。
WATCHDOG_INTERVAL = 5.0
WATCHDOG_GRACE = 15.0
RTCP_TIMEOUT = 15.0
DISCONNECTED_TIMEOUT = 10.0

LOGGER = logging.getLogger("mock-live-camera")


def _even(value: float) -> int:
    """Return the largest even integer not greater than value (minimum 2)."""
    return max(2, int(value) // 2 * 2)


def fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Fit within a bounding box, preserving aspect ratio and even dimensions."""
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / float(width))
    if max_height > 0:
        scale = min(scale, max_height / float(height))
    return _even(width * scale), _even(height * scale)


def codec_preferences(mime: str) -> list[object]:
    """Put the requested primary codec first, then RTX, then the rest."""
    capabilities = list(RTCRtpSender.getCapabilities("video").codecs)
    primary = [codec for codec in capabilities if codec.mimeType.lower() == mime]
    rtx = [codec for codec in capabilities if codec.mimeType.lower() == "video/rtx"]
    rest = [codec for codec in capabilities if codec not in primary and codec not in rtx]
    ordered = [*primary, *rtx, *rest]
    return ordered or capabilities


def configure_bitrate(bitrate: int) -> None:
    """Override aiortc's default encoder bitrate before sessions are created."""
    import aiortc.codecs.h264 as h264
    import aiortc.codecs.vpx as vpx

    for module in (h264, vpx):
        if hasattr(module, "DEFAULT_BITRATE"):
            module.DEFAULT_BITRATE = int(bitrate)


@dataclass
class VideoProbeResult:
    fps: float
    width: int
    height: int


def probe_video(path: Path, fps_override: Optional[float]) -> VideoProbeResult:
    """Read basic stream metadata without keeping the container open."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("文件不包含视频轨")
            stream = container.streams.video[0]
            rate = fps_override or stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else DEFAULT_FPS_FALLBACK
            return VideoProbeResult(
                fps=max(1.0, min(MAX_FPS, fps)),
                width=int(stream.width),
                height=int(stream.height),
            )
    except Exception as exc:
        raise ValueError(f"无法读取视频文件 {path}：{exc}") from exc


class VideoFileTrack(VideoStreamTrack):
    """A video track that loops a decoded file at a real-time pace."""

    kind = "video"

    def __init__(
        self,
        path: Path,
        fps: float,
        max_width: int,
        max_height: int,
        loop: bool,
        on_eof=None,
    ) -> None:
        super().__init__()
        self.path = path
        self.fps = fps
        self.loop = loop
        self._on_eof = on_eof
        self._loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mock-live-decode")
        self._container = None
        self._frames = None
        self._next_due = 0.0
        self._target_size: Optional[tuple[int, int]] = None
        self._max_width = max_width
        self._max_height = max_height
        self._last_error = ""
        self.frames_sent = 0
        self.started_at = time.monotonic()

    # ------------------------------------------------------------------ decode
    def _ensure_container(self) -> None:
        if self._container is None:
            container = av.open(str(self.path))
            if not container.streams.video:
                container.close()
                raise ValueError("文件不包含视频轨")
            container.streams.video[0].thread_type = "AUTO"
            self._container = container
            self._frames = container.decode(container.streams.video[0])

    def _reopen(self) -> None:
        self._close_container()
        self._ensure_container()

    def _close_container(self) -> None:
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()
        self._container = None
        self._frames = None

    def _decode_next_frame(self) -> av.VideoFrame:
        self._ensure_container()
        assert self._frames is not None
        try:
            frame = next(self._frames)
        except StopIteration:
            if not self.loop:
                raise
            self._reopen()
            frame = next(self._frames)
        if self._target_size is None:
            self._target_size = fit_size(
                frame.width, frame.height, self._max_width, self._max_height
            )
        target_width, target_height = self._target_size
        if frame.width != target_width or frame.height != target_height:
            frame = frame.reformat(width=target_width, height=target_height, format="yuv420p")
        elif frame.format.name != "yuv420p":
            frame = frame.reformat(format="yuv420p")
        return frame

    # ------------------------------------------------------------------ pacing
    def _pace_delay(self, now: float) -> float:
        if self._next_due <= 0:
            self._next_due = now
        delay = self._next_due - now
        self._next_due += 1.0 / self.fps
        # Fallen far behind (slow encode / long pause): resync instead of bursting.
        if self._next_due < now - 1.0:
            self._next_due = now
        return delay

    async def recv(self) -> av.VideoFrame:
        delay = self._pace_delay(time.monotonic())
        if delay > 0:
            await asyncio.sleep(delay)
        timestamp, time_base = await self.next_timestamp()
        try:
            frame = await self._loop.run_in_executor(self._executor, self._decode_next_frame)
        except StopIteration:
            if self._on_eof is not None:
                with contextlib.suppress(Exception):
                    self._on_eof()
            self.stop()
            raise MediaStreamError
        except Exception as exc:
            self._last_error = str(exc)
            self.stop()
            raise MediaStreamError from exc
        frame.pts = timestamp
        frame.time_base = time_base
        self.frames_sent += 1
        return frame

    def close(self) -> None:
        self.stop()
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.submit(self._close_container).result(timeout=3)
            self._executor.shutdown(wait=False)
            self._executor = None

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        return self._target_size


@dataclass
class _Session:
    pc: RTCPeerConnection
    track: VideoFileTrack
    created_at: float
    peer: str = ""
    disconnected_since: float = 0.0


class MockLiveCameraServer:
    def __init__(
        self,
        video: Path,
        probe: VideoProbeResult,
        *,
        loop: bool,
        max_width: int,
        max_height: int,
        codec: str,
    ) -> None:
        self.video = video
        self.probe = probe
        self.loop = loop
        self.max_width = max_width
        self.max_height = max_height
        self.codec = codec
        self.mime = "video/vp8" if codec == "vp8" else "video/h264"
        self.sessions: dict[str, _Session] = {}
        self.route = ""
        self.started_at = time.monotonic()
        self._watchdog_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ routes
    def build_app(self, route_path: str) -> web.Application:
        self.route = "/" + route_path.strip("/")
        app = web.Application()
        app.router.add_post(self.route, self.handle_whep)
        app.router.add_delete(self.route + "/{session_id}", self.handle_delete)
        app.router.add_get("/healthz", self.handle_health)
        app.router.add_get("/", self.handle_index)
        app.on_startup.append(self._start_watchdog)
        app.on_cleanup.append(self._cleanup)
        return app

    async def handle_index(self, request: web.Request) -> web.Response:
        text = (
            "模拟直播摄像头\n"
            f"视频: {self.video}\n"
            f"WHEP 地址: {request.scheme}://{request.host}{self.route}\n"
            "在安全监测 WebUI 中选择“直播流”并填入上面的地址。\n"
        )
        return web.Response(text=text, content_type="text/plain")

    async def handle_health(self, request: web.Request) -> web.Response:
        active_resolution = None
        for session in self.sessions.values():
            if session.track.resolution is not None:
                active_resolution = list(session.track.resolution)
                break
        return web.json_response(
            {
                "video": str(self.video),
                "fps": self.probe.fps,
                "source_size": [self.probe.width, self.probe.height],
                "output_size": active_resolution or [],
                "codec": self.mime,
                "loop": self.loop,
                "sessions": len(self.sessions),
                "frames_sent": sum(session.track.frames_sent for session in self.sessions.values()),
                "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            }
        )

    async def handle_whep(self, request: web.Request) -> web.Response:
        offer_sdp = await request.text()
        if not offer_sdp.strip():
            raise web.HTTPBadRequest(text="SDP offer 不能为空")

        session_id = uuid.uuid4().hex
        # 模拟器只服务本机/局域网：禁用 STUN，避免等待不可达的公共 STUN 服务器
        # 拖长 SDP answer 的生成时间（超过拉流端 5 秒信令超时）。
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = VideoFileTrack(
            self.video,
            self.probe.fps,
            self.max_width,
            self.max_height,
            self.loop,
            on_eof=lambda sid=session_id: self._schedule_close(sid),
        )
        transceiver = pc.addTransceiver(track, direction="sendonly")
        try:
            transceiver.setCodecPreferences(codec_preferences(self.mime))
        except Exception as exc:  # pragma: no cover - aiortc version differences
            LOGGER.warning("设置编码偏好失败，将使用 aiortc 默认协商：%s", exc)

        session = _Session(pc=pc, track=track, created_at=time.monotonic(), peer=request.remote or "")
        self.sessions[session_id] = session

        @pc.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            state = pc.connectionState
            if state in ("failed", "closed"):
                self._schedule_close(session_id)
            elif state == "disconnected" and session.disconnected_since <= 0:
                session.disconnected_since = time.monotonic()
            elif state == "connected":
                session.disconnected_since = 0.0

        try:
            started = time.monotonic()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
            LOGGER.debug("setRemoteDescription %.3fs", time.monotonic() - started)
            answer = await pc.createAnswer()
            LOGGER.debug("createAnswer %.3fs", time.monotonic() - started)
            await pc.setLocalDescription(answer)
            LOGGER.debug("setLocalDescription %.3fs", time.monotonic() - started)
            await self._wait_for_ice_gathering(pc)
            LOGGER.debug("iceGathering %s %.3fs", pc.iceGatheringState, time.monotonic() - started)
        except Exception:
            await self._close_session(session_id)
            raise

        local = pc.localDescription
        if local is None:
            await self._close_session(session_id)
            raise web.HTTPInternalServerError(text="无法生成 WHEP answer")

        negotiated = self._negotiated_codec(local.sdp)
        LOGGER.info(
            "拉流已连接 session=%s 对端=%s 编码=%s",
            session_id[:8],
            request.remote or "-",
            negotiated or self.mime,
        )
        return web.Response(
            status=201,
            text=local.sdp,
            headers={
                "Content-Type": "application/sdp",
                "Location": f"{self.route}/{session_id}",
            },
        )

    async def handle_delete(self, request: web.Request) -> web.Response:
        await self._close_session(request.match_info["session_id"])
        return web.Response(status=200, text="")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    async def _wait_for_ice_gathering(pc: RTCPeerConnection) -> None:
        deadline = time.monotonic() + ICE_GATHER_TIMEOUT
        while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    @staticmethod
    def _negotiated_codec(sdp: str) -> str:
        for line in sdp.replace("\r\n", "\n").split("\n"):
            if line.startswith("a=rtpmap:") and ("H264" in line or "VP8" in line):
                return line.split(" ", 1)[1]
        return ""

    def _schedule_close(self, session_id: str) -> None:
        try:
            asyncio.get_running_loop().create_task(self._close_session(session_id))
        except RuntimeError:
            # Event loop already shutting down; cleanup happens in _cleanup.
            pass

    async def _close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        with contextlib.suppress(Exception):
            await session.pc.close()
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(None, session.track.close)
        LOGGER.info(
            "拉流已断开 session=%s 共发送 %d 帧",
            session_id[:8],
            session.track.frames_sent,
        )

    # --------------------------------------------------------------- watchdog
    async def _start_watchdog(self, app: web.Application) -> None:
        self._watchdog_task = asyncio.ensure_future(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """回收没有对端响应的会话，避免断开后仍然满速编码。"""
        try:
            while True:
                await asyncio.sleep(WATCHDOG_INTERVAL)
                for session_id, session in list(self.sessions.items()):
                    reason = ""
                    with contextlib.suppress(Exception):
                        reason = await self._stale_reason(session)
                    if reason:
                        LOGGER.info(
                            "回收无响应会话 session=%s 原因=%s 已发送 %d 帧",
                            session_id[:8],
                            reason,
                            session.track.frames_sent,
                        )
                        await self._close_session(session_id)
        except asyncio.CancelledError:
            raise

    async def _stale_reason(self, session: _Session) -> str:
        age = time.monotonic() - session.created_at
        state = session.pc.connectionState
        if state == "closed":
            return "PeerConnection 已关闭"
        if state == "failed":
            return "连接失败"
        if session.disconnected_since > 0:
            if time.monotonic() - session.disconnected_since > DISCONNECTED_TIMEOUT:
                return "连接中断超时"
            return ""

        # 还没送出任何帧时给足协商时间，长时间不出画面才回收。
        if session.track.frames_sent <= 0:
            if age > WATCHDOG_GRACE + RTCP_TIMEOUT:
                return "始终没有发送视频帧"
            return ""
        if age < WATCHDOG_GRACE:
            return ""

        last_rtcp = await self._last_rtcp_time(session.pc)
        if last_rtcp is None:
            if age > WATCHDOG_GRACE + RTCP_TIMEOUT:
                return "未收到任何 RTCP 接收者报告"
            return ""
        silent = (datetime.datetime.now(datetime.timezone.utc) - last_rtcp).total_seconds()
        if silent > RTCP_TIMEOUT:
            return f"RTCP 已静默 {silent:.0f} 秒"
        return ""

    @staticmethod
    async def _last_rtcp_time(pc: RTCPeerConnection) -> Optional[datetime.datetime]:
        report = await pc.getStats()
        newest: Optional[datetime.datetime] = None
        for stat in report.values():
            if getattr(stat, "type", "") != "remote-inbound-rtp":
                continue
            stamp = getattr(stat, "timestamp", None)
            if isinstance(stamp, datetime.datetime) and (newest is None or stamp > newest):
                newest = stamp
        return newest

    async def _cleanup(self, app: web.Application) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._watchdog_task
            self._watchdog_task = None
        for session_id in list(self.sessions):
            await self._close_session(session_id)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本地视频文件模拟 WHEP 直播流（循环播放，供安全监测拉流检测）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="要循环播放的视频文件路径")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    parser.add_argument("--path", default=DEFAULT_PATH, help="WHEP 路径，最终地址为 http://host:port/<path>")
    parser.add_argument("--fps", type=float, default=None, help="覆盖视频原始帧率")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="输出最大宽度，0 表示不限制")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="输出最大高度，0 表示不限制")
    parser.add_argument("--no-resize", action="store_true", help="保持视频原始分辨率")
    parser.add_argument("--codec", choices={"h264", "vp8"}, default="h264", help="视频编码")
    parser.add_argument("--bitrate", type=int, default=2_000_000, help="目标码率，单位 bit/s")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="只播放一次")
    parser.set_defaults(loop=True)
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    # Windows 控制台/重定向日志默认使用 GBK，统一为 UTF-8 避免中文乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[模拟直播] %(levelname)s %(message)s",
    )
    # aiortc 的 ICE 候选对日志在 INFO 级别非常密集，默认只看告警。
    logging.getLogger("aiortc").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    video = Path(args.video).expanduser()
    if not video.is_file():
        print(f"视频文件不存在：{video}", file=sys.stderr)
        return 2

    try:
        probe = probe_video(video, args.fps)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    max_width = 0 if args.no_resize else max(0, args.width)
    max_height = 0 if args.no_resize else max(0, args.height)
    output_width, output_height = fit_size(probe.width, probe.height, max_width, max_height)

    configure_bitrate(max(500_000, int(args.bitrate)))

    server = MockLiveCameraServer(
        video,
        probe,
        loop=args.loop,
        max_width=max_width,
        max_height=max_height,
        codec=args.codec,
    )
    route_path = args.path.strip("/") or DEFAULT_PATH
    app = server.build_app(route_path)
    url = f"http://{args.host}:{args.port}/{route_path}"

    print("=" * 60)
    print("模拟直播摄像头已启动")
    print(f"  视频文件 : {video}")
    print(f"  原始画面 : {probe.width}x{probe.height} @ {probe.fps:g} fps")
    print(f"  输出画面 : {output_width}x{output_height} @ {probe.fps:g} fps（{'循环' if args.loop else '单次'}）")
    print(f"  编码     : {args.codec.upper()} / {args.bitrate} bit/s")
    print(f"  WHEP 地址: {url}")
    print("  在 WebUI 选择“直播流”，填入上面的 WHEP 地址，")
    print("  先点“测试 Python 后端拉流”，画面连续后再“开始检测”。")
    print("  按 Ctrl+C 停止。")
    print("=" * 60, flush=True)

    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
