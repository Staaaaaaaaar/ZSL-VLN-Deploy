from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure repo root is importable when executing: python scripts/xxx.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.robot import FFmpegCameraStream, GStreamerCameraStream


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open camera stream and visualize frames")
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to runtime config JSON",
    )
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for the first frame before failing fast",
    )
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "gstreamer"],
        default=None,
        help="Camera backend override",
    )
    parser.add_argument(
        "--rtsp-transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport mode",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    camera_cfg = cfg.get("camera", {})
    rtsp_url = camera_cfg.get("rtsp_url")
    if not rtsp_url:
        print("[ERR] rtsp url is empty")
        return 2

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    warmup_timeout = args.warmup_timeout
    if warmup_timeout is None:
        warmup_timeout = float(camera_cfg.get("warmup_timeout_sec", 4.0))
    backend = (args.backend or camera_cfg.get("backend", "ffmpeg")).strip().lower()

    if backend == "gstreamer":
        stream = GStreamerCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=args.rtsp_transport,
            latency=int(camera_cfg.get("gst_latency", 0)),
            drop=bool(camera_cfg.get("gst_drop", True)),
            max_buffers=int(camera_cfg.get("gst_max_buffers", 1)),
        )
    else:
        stream = FFmpegCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=args.rtsp_transport,
            low_latency=bool(camera_cfg.get("ffmpeg_low_latency", True)),
        )

    try:
        import cv2

        print(f"[INFO] camera rtsp: {rtsp_url}")
        print(f"[INFO] backend: {backend}")
        print(f"[INFO] first-frame timeout: {warmup_timeout:.1f}s")
        print(f"[INFO] rtsp transport: {args.rtsp_transport}")
        stream.start()
        waiting_since = None
        while True:
            pkt = stream.read_latest(timeout_sec=1.0)
            if pkt is None:
                if waiting_since is None:
                    waiting_since = time.monotonic()
                waited = time.monotonic() - waiting_since
                print("[WARN] waiting for frame...")
                if waited >= warmup_timeout:
                    stats = stream.stats()
                    print("[ERR] no frame received before timeout")
                    print(
                        "[HINT] robot network may be disconnected. "
                        "Switch to robot LAN (e.g., 192.168.234.x) and retry."
                    )
                    print(f"[HINT] stream stats: {stats}")
                    if stats.get("backend") == "open_failed":
                        if backend == "gstreamer":
                            print(
                                "[HINT] gstreamer open failed. Check OpenCV build info for GStreamer: YES."
                            )
                        else:
                            print(
                                "[HINT] both ffmpeg and generic OpenCV open failed. "
                                "Check URL/path/permissions on robot stream service."
                            )
                    return 3
                continue

            waiting_since = None

            cv2.imshow("zsi-rtsp", pkt.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        print("\n[WARN] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[ERR] camera test failed: {exc}")
        return 1
    finally:
        stream.stop()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
