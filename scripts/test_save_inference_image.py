from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# Ensure repo root is importable when executing: python scripts/xxx.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.robot import FFmpegCameraStream, GStreamerCameraStream
from robot_deploy.runtime import RuntimeController


@dataclass
class SaveResult:
    index: int
    timestamp_ms: int
    image_path: str | None
    ok: bool
    reason: str


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_stream(cfg: dict, backend_override: str | None, rtsp_transport: str):
    camera_cfg = cfg.get("camera", {})
    rtsp_url = camera_cfg.get("rtsp_url")
    if not rtsp_url:
        raise ValueError("camera.rtsp_url is empty")

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    backend = (backend_override or camera_cfg.get("backend", "gstreamer")).strip().lower()
    force_rgb_conversion = bool(camera_cfg.get("force_rgb_conversion", False))

    if backend == "gstreamer":
        stream = GStreamerCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=rtsp_transport,
            latency=int(camera_cfg.get("gst_latency", 0)),
            drop=bool(camera_cfg.get("gst_drop", True)),
            max_buffers=int(camera_cfg.get("gst_max_buffers", 1)),
            force_rgb_conversion=force_rgb_conversion,
        )
    elif backend == "ffmpeg":
        stream = FFmpegCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=rtsp_transport,
            low_latency=bool(camera_cfg.get("ffmpeg_low_latency", True)),
            force_rgb_conversion=force_rgb_conversion,
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")

    return stream, backend, rtsp_url


def _diagnose_frame_issue(frame: Any, min_std: float, min_range: int) -> str | None:
    try:
        import numpy as np

        arr = np.asarray(frame)
        if arr.size == 0:
            return "empty_array"

        if arr.dtype.kind == "f":
            arr_f = arr.astype(np.float32)
            if arr_f.size == 0:
                return "empty_array"
            vmin = float(arr_f.min())
            vmax = float(arr_f.max())
            std = float(arr_f.std())
            if (vmax - vmin) < float(min_range / 255.0) and std < float(min_std / 255.0):
                return "low_dynamic_range"
            return None

        arr_u8 = arr.astype(np.uint8, copy=False)
        vmin = int(arr_u8.min())
        vmax = int(arr_u8.max())
        std = float(arr_u8.std())
        if (vmax - vmin) < min_range and std < min_std:
            return "low_dynamic_range"
        return None
    except Exception:
        # If quality check fails unexpectedly, do not block saving path.
        return None


def _read_valid_frame(
    stream,
    *,
    timeout_sec: float,
    retries: int,
    retry_gap_sec: float,
    min_std: float,
    min_range: int,
):
    last_pkt = None
    last_issue = "no_frame"
    for _ in range(max(1, retries)):
        pkt = stream.read_latest(timeout_sec=timeout_sec)
        if pkt is None:
            last_issue = "no_frame"
            continue

        last_pkt = pkt
        issue = _diagnose_frame_issue(pkt.frame, min_std=min_std, min_range=min_range)
        if issue is None:
            return pkt, None

        last_issue = issue
        if retry_gap_sec > 0:
            time.sleep(retry_gap_sec)
    return last_pkt, last_issue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Periodically save current frame via RuntimeController._save_inference_image"
    )
    parser.add_argument("--config", default="configs/default.json", help="Path to config JSON")
    parser.add_argument(
        "--backend",
        choices=["gstreamer", "ffmpeg"],
        default=None,
        help="Override camera backend",
    )
    parser.add_argument(
        "--rtsp-transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport mode",
    )
    parser.add_argument(
        "--save-dir",
        default="runtime_logs/inference_image_test",
        help="Directory to save images",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=2.0,
        help="Save one frame every N seconds",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many frames to save",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=2.0,
        help="Timeout when waiting latest frame",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=2.0,
        help="Warmup time before first save to avoid unstable startup frame",
    )
    parser.add_argument(
        "--frame-retries",
        type=int,
        default=4,
        help="Retries for validating a frame before save",
    )
    parser.add_argument(
        "--retry-gap-sec",
        type=float,
        default=0.08,
        help="Sleep between frame retries",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=2.0,
        help="Minimum std-dev threshold for frame validity check",
    )
    parser.add_argument(
        "--min-range",
        type=int,
        default=6,
        help="Minimum max-min pixel range for frame validity check",
    )
    args = parser.parse_args()

    if args.interval_sec <= 0:
        print("[ERR] --interval-sec must be > 0")
        return 2
    if args.count <= 0:
        print("[ERR] --count must be > 0")
        return 2
    if args.warmup_sec < 0:
        print("[ERR] --warmup-sec must be >= 0")
        return 2
    if args.frame_retries <= 0:
        print("[ERR] --frame-retries must be > 0")
        return 2

    cfg = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        stream, backend, rtsp_url = choose_stream(cfg, args.backend, args.rtsp_transport)
    except Exception as exc:
        print(f"[ERR] build stream failed: {exc}")
        return 2

    # Reuse the exact method under test without constructing full runtime dependencies.
    controller = object.__new__(RuntimeController)
    results: list[SaveResult] = []

    print(f"[INFO] rtsp: {rtsp_url}")
    print(f"[INFO] backend: {backend}")
    print(f"[INFO] save_dir: {args.save_dir}")
    print(f"[INFO] interval_sec: {args.interval_sec}")
    print(f"[INFO] count: {args.count}")
    print(f"[INFO] warmup_sec: {args.warmup_sec}")
    print(f"[INFO] frame_retries: {args.frame_retries}")

    try:
        stream.start()
        if args.warmup_sec > 0:
            print("[INFO] warming up stream...")
            time.sleep(args.warmup_sec)

        next_ts = time.monotonic()

        for i in range(1, args.count + 1):
            sleep_sec = max(0.0, next_ts - time.monotonic())
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            next_ts += args.interval_sec

            pkt, issue = _read_valid_frame(
                stream,
                timeout_sec=args.frame_timeout,
                retries=args.frame_retries,
                retry_gap_sec=max(0.0, args.retry_gap_sec),
                min_std=float(args.min_std),
                min_range=int(args.min_range),
            )
            ts_ms = int(time.time() * 1000)
            record_id = f"save_test_{i:04d}_{ts_ms}"

            if pkt is None:
                reason = issue or "no_frame"
                print(f"[WARN] #{i}: skip, reason={reason}")
                results.append(
                    SaveResult(
                        index=i,
                        timestamp_ms=ts_ms,
                        image_path=None,
                        ok=False,
                        reason=reason,
                    )
                )
                continue

            image_path = controller._save_inference_image(
                save_dir=args.save_dir,
                record_id=record_id,
                image=pkt.frame,
            )

            if image_path is None:
                reason = "save_returned_none"
                ok = False
            elif not os.path.exists(image_path):
                reason = "path_not_exists"
                ok = False
            elif os.path.getsize(image_path) <= 0:
                reason = "empty_file"
                ok = False
            elif issue is not None:
                reason = f"saved_with_warning:{issue}"
                ok = True
            else:
                reason = "ok"
                ok = True

            status = "OK" if ok else "FAIL"
            print(f"[INFO] #{i}: {status} path={image_path} reason={reason}")

            results.append(
                SaveResult(
                    index=i,
                    timestamp_ms=ts_ms,
                    image_path=image_path,
                    ok=ok,
                    reason=reason,
                )
            )
    except KeyboardInterrupt:
        print("\n[WARN] interrupted by user")
    except Exception as exc:
        print(f"[ERR] run failed: {exc}")
        return 1
    finally:
        stream.stop()

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    fail_count = total - ok_count
    stats = stream.stats()

    summary = {
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "backend": backend,
        "save_dir": args.save_dir,
        "stream_stats": stats,
        "results": [
            {
                "index": r.index,
                "timestamp_ms": r.timestamp_ms,
                "image_path": r.image_path,
                "ok": r.ok,
                "reason": r.reason,
            }
            for r in results
        ],
    }

    summary_path = os.path.join(args.save_dir, f"summary_{int(time.time() * 1000)}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[SUMMARY] total={total} ok={ok_count} fail={fail_count}")
    print(f"[SUMMARY] stream_stats={stats}")
    print(f"[SUMMARY] summary_json={summary_path}")

    if total == 0:
        return 1
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
