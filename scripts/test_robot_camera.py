from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repo root is importable when executing: python scripts/xxx.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.robot import GStreamerCameraStream


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open camera stream and visualize frames")
    args = parser.parse_args()

    cfg = load_config("configs/default.json")
    camera_cfg = cfg.get("camera", {})
    rtsp_url = camera_cfg.get("rtsp_url")
    if not rtsp_url:
        print("[ERR] rtsp url is empty")
        return 2

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))

    stream = GStreamerCameraStream(
        rtsp_url=rtsp_url,
        width=width,
        height=height,
    )

    try:
        import cv2

        stream.start()
        while True:
            pkt = stream.read_latest(timeout_sec=1.0)
            if pkt is None:
                print("[WARN] waiting for frame...")
                continue

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
