from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_host_port(rtsp_url: str) -> Tuple[str, int]:
    # Minimal parser for rtsp://host:port/path
    if not rtsp_url.startswith("rtsp://"):
        raise ValueError(f"unsupported rtsp url: {rtsp_url}")

    rest = rtsp_url[len("rtsp://") :]
    host_port = rest.split("/", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        port = int(port_str)
    else:
        host = host_port
        port = 554
    return host, port


def tcp_check(host: str, port: int, timeout_sec: float) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def opencv_read_check(source, backend, timeout_sec: float) -> Tuple[bool, int, str]:
    import cv2

    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        return False, 0, "open_failed"

    deadline = time.monotonic() + max(timeout_sec, 0.1)
    frames_ok = 0
    last_err = ""
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                frames_ok += 1
                if frames_ok >= 1:
                    return True, frames_ok, "ok"
            else:
                last_err = "read_failed"
            time.sleep(0.05)
    finally:
        cap.release()

    return False, frames_ok, last_err or "timeout_no_frame"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose RTSP camera path step-by-step")
    parser.add_argument("--config", default="configs/default.json", help="Config json path")
    parser.add_argument("--url", default=None, help="Override rtsp url")
    parser.add_argument("--timeout", type=float, default=4.0, help="Timeout per check")
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp", help="FFmpeg RTSP transport")
    args = parser.parse_args()

    cfg = load_config(args.config)
    camera_cfg = cfg.get("camera", {})

    rtsp_url = args.url or camera_cfg.get("rtsp_url")
    if not rtsp_url:
        print("[ERR] rtsp_url is empty")
        return 2

    print(f"[INFO] rtsp: {rtsp_url}")
    print(f"[INFO] timeout: {args.timeout:.1f}s")
    print(f"[INFO] ffmpeg rtsp transport: {args.rtsp_transport}")

    try:
        host, port = parse_host_port(rtsp_url)
    except Exception as exc:
        print(f"[ERR] parse url failed: {exc}")
        return 2

    # 1) TCP port reachability
    ok_tcp = tcp_check(host, port, args.timeout)
    print(f"[CHECK-1] tcp {host}:{port}: {'OK' if ok_tcp else 'FAIL'}")
    if not ok_tcp:
        print("[HINT] Network route/VLAN is likely wrong, or camera service not listening")
        return 3

    # 2) OpenCV FFmpeg build + RTSP open check
    try:
        import cv2

        build_info = cv2.getBuildInformation()
        has_ffmpeg = any(
            ("FFMPEG" in ln and "YES" in ln)
            for ln in build_info.splitlines()
        )
        print(f"[CHECK-2] opencv ffmpeg support: {'YES' if has_ffmpeg else 'NO'}")
        if not has_ffmpeg:
            print("[HINT] Install OpenCV build with FFmpeg support in current environment")
            return 4

        ffmpeg_opts = (
            f"rtsp_transport;{args.rtsp_transport}|"
            "fflags;nobuffer|flags;low_delay|max_delay;500000|stimeout;5000000"
        )
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts
        ok_ffm, n_ffm, msg_ffm = opencv_read_check(rtsp_url, cv2.CAP_FFMPEG, args.timeout)
        print(f"[CHECK-3] opencv ffmpeg rtsp: {'OK' if ok_ffm else 'FAIL'} frames={n_ffm} reason={msg_ffm}")
    except Exception as exc:
        ok_ffm = False
        print(f"[CHECK-3] opencv ffmpeg rtsp: FAIL exception={exc}")
        return 4

    if not ok_ffm:
        print("[HINT] RTSP service is reachable but no decodable stream; verify URL path, codec, or robot camera service")
        return 5

    print("[OK] Camera stream looks healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
