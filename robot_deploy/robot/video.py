from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class FramePacket:
    frame: Any
    timestamp: float


class GStreamerCameraStream:
    """RTSP camera reader using GStreamer and keeping the latest frame."""

    def __init__(
        self,
        rtsp_url: str,
        width: int = 1280,
        height: int = 720,
        reconnect_interval_sec: float = 0.8,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.reconnect_interval_sec = reconnect_interval_sec

        self._cv2 = None
        self._capture = None
        self._thread: threading.Thread | None = None
        self._running = False

        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._frames_ok = 0
        self._frames_fail = 0

    def start(self) -> None:
        if self._running:
            return
        self._cv2 = self._import_cv2()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name="gstreamer-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release_capture()

    def read_latest(self, timeout_sec: float = 2.0) -> FramePacket | None:
        deadline = time.time() + max(timeout_sec, 0.0)
        while time.time() < deadline:
            with self._lock:
                pkt = self._latest
            if pkt is not None:
                return pkt
            time.sleep(0.02)
        return None

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            ts = self._latest.timestamp if self._latest else 0.0
            age = max(time.time() - ts, 0.0) if ts > 0 else -1.0
            return {
                "frames_ok": self._frames_ok,
                "frames_fail": self._frames_fail,
                "latest_age_sec": age,
            }

    def _reader_loop(self) -> None:
        while self._running:
            if self._capture is None:
                self._capture = self._open_capture()
                if self._capture is None:
                    self._frames_fail += 1
                    time.sleep(self.reconnect_interval_sec)
                    continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._frames_fail += 1
                self._release_capture()
                time.sleep(self.reconnect_interval_sec)
                continue

            self._frames_ok += 1
            with self._lock:
                self._latest = FramePacket(frame=frame, timestamp=time.time())

    def _open_capture(self):
        cv2 = self._cv2
        pipeline = self._build_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        return None

    def _build_pipeline(self) -> str:
        scale = ""
        if self.width > 0 and self.height > 0:
            scale = f" ! videoscale ! video/x-raw,width={self.width},height={self.height}"

        return (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert"
            f"{scale} ! video/x-raw,format=BGR ! "
            "appsink sync=false drop=true max-buffers=1"
        )

    def _release_capture(self) -> None:
        cap = self._capture
        self._capture = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    @staticmethod
    def _import_cv2():
        try:
            import cv2

            return cv2
        except Exception as exc:
            raise RuntimeError("opencv-python is required for camera streaming") from exc
