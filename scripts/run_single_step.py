from __future__ import annotations

import json
import os
import sys
import time

# Ensure repo root is importable when executing: python scripts/xxx.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.core import NavigationRequest, RobotEndpoint, RuntimeSafetyLimits
from robot_deploy.interface import ActiveVLNActionInterface
from robot_deploy.model import ActiveVLNOpenAIModel
from robot_deploy.robot import ZSLHighLevelRobot
from robot_deploy.robot.video import FFmpegCameraStream, GStreamerCameraStream
from robot_deploy.runtime import RuntimeController, RuntimePolicy


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def capture_rtsp_image(camera_cfg: dict | None):
    if not camera_cfg:
        raise RuntimeError("camera config is required")

    rtsp_url = str(camera_cfg.get("rtsp_url", "")).strip()
    if not rtsp_url:
        raise RuntimeError("camera.rtsp_url is empty")

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    timeout_sec = float(camera_cfg.get("warmup_timeout_sec", 4.0))
    backend = str(camera_cfg.get("backend", "ffmpeg")).strip().lower()
    rtsp_transport = str(camera_cfg.get("rtsp_transport", "tcp")).strip().lower()

    if backend == "gstreamer":
        stream = GStreamerCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=rtsp_transport,
            latency=int(camera_cfg.get("gst_latency", 0)),
            drop=bool(camera_cfg.get("gst_drop", True)),
            max_buffers=int(camera_cfg.get("gst_max_buffers", 1)),
        )
    else:
        stream = FFmpegCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=rtsp_transport,
            low_latency=bool(camera_cfg.get("ffmpeg_low_latency", True)),
        )
    stream.start()

    try:
        deadline = time.time() + max(timeout_sec, 0.2)
        while time.time() < deadline:
            pkt = stream.read_latest(timeout_sec=0.5)
            if pkt is None:
                continue

            frame = pkt.frame
            try:
                # Convert OpenCV BGR frame to RGB for PIL/OpenAI image payload.
                rgb = frame[:, :, ::-1].copy()
            except Exception:
                return frame

            try:
                from PIL import Image

                return Image.fromarray(rgb)
            except Exception:
                return rgb
        raise RuntimeError(f"camera frame warmup timeout after {timeout_sec:.2f}s")
    finally:
        stream.stop()


def main() -> None:
    cfg = load_config(os.path.join("configs", "default.json"))

    robot_cfg = cfg["robot"]
    endpoint_cfg = robot_cfg["endpoint"]
    endpoint = RobotEndpoint(
        local_ip=endpoint_cfg["local_ip"],
        local_port=int(endpoint_cfg["local_port"]),
        dog_ip=endpoint_cfg["dog_ip"],
    )

    model_cfg = cfg["model"]
    interface_cfg = cfg["interface"]
    safety_cfg = cfg["runtime"]["safety"]
    policy_cfg = cfg["runtime"]["policy"]

    robot = ZSLHighLevelRobot(
        robot_model=robot_cfg["robot_model"],
        sdk_root=robot_cfg["sdk_root"],
    )
    model = ActiveVLNOpenAIModel(**model_cfg)
    action_interface = ActiveVLNActionInterface(**interface_cfg)
    safety = RuntimeSafetyLimits(**safety_cfg)
    policy = RuntimePolicy(**policy_cfg)

    controller = RuntimeController(
        robot=robot,
        model=model,
        action_interface=action_interface,
        safety=safety,
        policy=policy,
    )

    req_cfg = cfg["request"]
    camera_cfg = cfg.get("camera")
    request_image = capture_rtsp_image(camera_cfg)

    nav_request = NavigationRequest(
        instruction=req_cfg["instruction"],
        image=request_image,
    )

    controller.startup(endpoint)
    try:
        result = controller.run_once(nav_request)
        print("ok:", result.ok)
        print("model_text:", result.model_text)
        print("error:", result.error)
        print("executed_commands:", len(result.executed_commands))
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
