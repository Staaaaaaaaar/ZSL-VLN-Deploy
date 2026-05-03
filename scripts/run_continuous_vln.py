from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.core import NavigationRequest, RobotEndpoint, RuntimeEpisodeConfig, RuntimeSafetyLimits
from robot_deploy.interface import ActiveVLNActionInterface
from robot_deploy.model import ActiveVLNOpenAIModel
from robot_deploy.robot import ZSLHighLevelRobot
from robot_deploy.robot.video import FFmpegCameraStream, GStreamerCameraStream
from robot_deploy.runtime import RuntimeController, RuntimePolicy


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_camera_stream(camera_cfg: dict[str, Any]):
    backend = str(camera_cfg.get("backend", "ffmpeg")).strip().lower()
    rtsp_url = str(camera_cfg.get("rtsp_url", "")).strip()
    if not rtsp_url:
        raise RuntimeError("camera.rtsp_url is empty")

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    transport = str(camera_cfg.get("rtsp_transport", "tcp")).strip().lower()
    force_rgb_conversion = bool(camera_cfg.get("force_rgb_conversion", False))

    if backend == "gstreamer":
        return GStreamerCameraStream(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=transport,
            latency=int(camera_cfg.get("gst_latency", 0)),
            drop=bool(camera_cfg.get("gst_drop", True)),
            max_buffers=int(camera_cfg.get("gst_max_buffers", 1)),
            force_rgb_conversion=force_rgb_conversion,
        )

    return FFmpegCameraStream(
        rtsp_url=rtsp_url,
        width=width,
        height=height,
        rtsp_transport=transport,
        low_latency=bool(camera_cfg.get("ffmpeg_low_latency", True)),
        force_rgb_conversion=force_rgb_conversion,
    )


def _frame_to_image(frame: Any):
    try:
        rgb = frame[:, :, ::-1].copy()
    except Exception:
        return frame, "unknown"

    try:
        from PIL import Image

        return Image.fromarray(rgb), "rgb"
    except Exception:
        return rgb, "rgb"


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
    request_cfg = cfg["request"]
    camera_cfg = cfg.get("camera")
    if not camera_cfg:
        raise RuntimeError("camera config is required")

    episode_cfg_raw = cfg.get("runtime_episode", {})
    episode_cfg = RuntimeEpisodeConfig(
        max_turn_budget=int(episode_cfg_raw.get("max_turn_budget", 40)),
        max_step_budget=int(episode_cfg_raw.get("max_step_budget", 120)),
        max_actions_per_turn=int(episode_cfg_raw.get("max_actions_per_turn", 3)),
        early_stop_rotation=int(episode_cfg_raw.get("early_stop_rotation", 12)),
        stop_on_disconnect=bool(episode_cfg_raw.get("stop_on_disconnect", True)),
    )

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

    camera_stream = _build_camera_stream(camera_cfg)
    camera_stream.start()

    warmup_timeout_sec = float(camera_cfg.get("warmup_timeout_sec", 4.0))
    warmup_deadline = time.time() + max(warmup_timeout_sec, 0.2)
    while time.time() < warmup_deadline:
        pkt = camera_stream.read_latest(timeout_sec=0.5)
        if pkt is not None:
            break
    else:
        camera_stream.stop()
        raise RuntimeError(f"camera frame warmup timeout after {warmup_timeout_sec:.2f}s")

    instruction = str(request_cfg.get("instruction", "")).strip()
    if not instruction:
        camera_stream.stop()
        raise RuntimeError("request.instruction is empty")

    def request_provider() -> NavigationRequest:
        pkt = camera_stream.read_latest(timeout_sec=1.0)
        if pkt is None:
            raise RuntimeError("camera frame timeout during episode")

        image, image_color_space = _frame_to_image(pkt.frame)

        state = None
        try:
            state = robot.read_state()
        except Exception:
            state = None

        metadata = {
            "frame_timestamp": pkt.timestamp,
            "camera_stats": camera_stream.stats(),
            "robot_state": state,
            "image_color_space": image_color_space,
        }

        return NavigationRequest(
            instruction=instruction,
            image=image,
            metadata=metadata,
        )

    controller.startup(endpoint)
    try:
        result = controller.run_episode(request_provider=request_provider, episode_config=episode_cfg)
        print("=== Episode Result ===")
        print("ok:", result.ok)
        print("stop_reason:", result.stop_reason)
        print("turns:", result.turns)
        print("steps:", result.steps)
        print("model_turns:", len(result.model_outputs))
        print("executed_commands:", len(result.executed_commands))
        print("error:", result.error)
    finally:
        try:
            camera_stream.stop()
        finally:
            controller.shutdown()


if __name__ == "__main__":
    main()
