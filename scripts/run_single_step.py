from __future__ import annotations

import json
import os

from robot_deploy.core import NavigationRequest, RobotEndpoint, RuntimeSafetyLimits
from robot_deploy.interface import ActiveVLNActionInterface
from robot_deploy.model import ActiveVLNOpenAIModel
from robot_deploy.robot import ZSLHighLevelRobot
from robot_deploy.runtime import RuntimeController, RuntimePolicy


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_image(image_value):
    if image_value is None:
        return None

    if not isinstance(image_value, str):
        return None

    if not os.path.isfile(image_value):
        return None

    try:
        from PIL import Image

        return Image.open(image_value).convert("RGB")
    except Exception:
        return None


def main() -> None:
    cfg = load_config(os.path.join("robot_deploy", "configs", "default.json"))

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
    nav_request = NavigationRequest(
        instruction=req_cfg["instruction"],
        image=maybe_load_image(req_cfg.get("image")),
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
