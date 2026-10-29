"""Real local WHEP/H.264 handshake and decode; no external stream is required."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aiortc")
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, VideoStreamTrack
from av import VideoFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "safety_tools"))
from live_stream import LiveStreamReader


class TestVideoTrack(VideoStreamTrack):
    __test__ = False

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[:, :, 1] = 160
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


@pytest.fixture
def whep_endpoint():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    peers = []

    async def offer(sdp):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peers.append(peer)
        await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        peer.addTrack(TestVideoTrack())
        await peer.setLocalDescription(await peer.createAnswer())
        # ZLMediaKit wraps SDP in JSON, while standard WHEP returns raw SDP.
        return {"code": 0, "sdp": peer.localDescription.sdp, "type": "answer"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            sdp = self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8")
            answer = asyncio.run_coroutine_threadsafe(offer(sdp), loop).result(timeout=15)
            body = json.dumps(answer).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    def serve():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(5), "Local WHEP server did not start"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index/api/whep?app=live&stream=test"
    finally:
        async def stop():
            for peer in peers:
                await peer.close()
        asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
        server.shutdown()
        server.server_close()
        http_thread.join(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def test_real_whep_h264_first_frame_and_continuous_decode(whep_endpoint):
    reader = LiveStreamReader(whep_endpoint).start()
    try:
        first = reader.wait_for_frame(-1, timeout=20)
        assert first is not None, reader.error
        assert first.frame.shape == (180, 320, 3)
        second = reader.wait_for_frame(first.frame_id, timeout=5)
        assert second is not None and second.frame_id > first.frame_id, reader.error
        assert reader.connected and reader.opened_once
        assert reader.reconnect_count == 0
    finally:
        reader.stop()
        reader.join(timeout=10)


def test_real_whep_with_all_safety_algorithms_loads_one_model_after_first_frame(whep_endpoint, monkeypatch, tmp_path):
    import torch
    try:
        torch.zeros(1).numpy()
    except RuntimeError:
        pytest.skip("PyTorch/NumPy binary incompatibility")
    import safety_web_server as server
    import safety_video_monitor as monitor
    from live_stream import stop_all_shared_live_readers
    weights = Path(__file__).resolve().parents[1] / "all_best.pt"
    if not weights.exists():
        pytest.skip("all_best.pt is not available")
    job_id = uuid.uuid4().hex
    loads = []
    original_load = monitor.load_model

    def load(path):
        # The real managed-job path must complete decoding before model setup.
        assert server.JOBS[job_id]["raw_frame_id"] > 0
        assert server.JOBS[job_id]["stream_ready"] is True
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(monitor, "load_model", load)
    server.create_job(job_id, {"job_id": job_id, "status": "running", "_live_stream_url": whep_endpoint})
    thread = server.start_managed_job(
        job_id, server.run_unified_safety_job,
        (job_id, set(monitor.SAFETY_CHECK_ORDER), "live_stream", whep_endpoint, weights,
         tmp_path / "safety.jpg", tmp_path / "safety.json", 0, 0.55, 0.7, "cpu", {}, {}, []),
        uses_algorithm_slot=True,
    )
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            job = dict(server.JOBS[job_id])
            if job.get("frame_status") or job["status"] in {"error", "done", "canceled"}:
                break
            time.sleep(0.1)
        assert job["status"] == "running", job.get("error", job.get("progress_text"))
        assert set(job["frame_status"]["selected_checks"]) == set(monitor.SAFETY_CHECK_ORDER)
        assert len(loads) == 1
        assert job["raw_frame_id"] > 0 and job["frame_id"] > 0
        server.update_job(job_id, cancel_requested=True)
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert server.JOBS[job_id]["status"] == "canceled"
        assert (tmp_path / "safety.json").is_file()
    finally:
        server.update_job(job_id, cancel_requested=True)
        thread.join(timeout=15)
        stop_all_shared_live_readers()
        server.JOBS.pop(job_id, None)
