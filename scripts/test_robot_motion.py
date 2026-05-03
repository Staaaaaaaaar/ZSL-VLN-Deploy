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

from robot_deploy.core import MotionCommand, RobotEndpoint
from robot_deploy.robot import ZSLHighLevelRobot


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _prompt_action() -> str:
    while True:
        try:
            raw = input("[HITL] 1前进 2左 3右 4停: ")
        except EOFError:
            return "stop"

        raw = str(raw).strip().lower()
        if raw in {"1", "f", "forward", "前进"}:
            return "forward"
        if raw in {"2", "l", "left", "左", "左转"}:
            return "left"
        if raw in {"3", "r", "right", "右", "右转"}:
            return "right"
        if raw in {"4", "s", "stop", "停", "停止"}:
            return "stop"
        print("[HITL] 输入无效，请输入 1/2/3/4。")


def _action_to_command(action: str, *, forward_vx: float, yaw_rate: float, duration: float) -> MotionCommand | None:
    action = str(action).strip().lower()
    if action == "forward":
        return MotionCommand(vx=forward_vx, vy=0.0, yaw_rate=0.0, duration_sec=duration)
    if action == "left":
        return MotionCommand(vx=0.0, vy=0.0, yaw_rate=yaw_rate, duration_sec=duration)
    if action == "right":
        return MotionCommand(vx=0.0, vy=0.0, yaw_rate=-yaw_rate, duration_sec=duration)
    if action == "stop":
        return None
    raise ValueError(f"unknown action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple motion test for ZSI robot high-level SDK")
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to runtime config json",
    )
    parser.add_argument("--forward-vx", type=float, default=0.15, help="Forward speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.35, help="Yaw rate in rad/s")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration per motion command")
    args = parser.parse_args()

    cfg = load_config(args.config)
    robot_cfg = cfg["robot"]
    endpoint_cfg = robot_cfg["endpoint"]

    endpoint = RobotEndpoint(
        local_ip=endpoint_cfg["local_ip"],
        local_port=int(endpoint_cfg["local_port"]),
        dog_ip=endpoint_cfg["dog_ip"],
    )

    robot = ZSLHighLevelRobot(
        robot_model=robot_cfg["robot_model"],
        sdk_root=robot_cfg["sdk_root"],
    )

    robot.connect(endpoint)

    time.sleep(3.0)

    if robot.check_connection():
        print("[OK] robot connected")
    else:
        print("[ERR] robot not connected")

    print("[STATE] initial robot state:")
    print(robot.read_state())

    print("[STEP] stand up")
    robot.stand_up()

    time.sleep(7.0)

    try:
        while True:
            executed_action = _prompt_action()
            print(f"[HITL] executed action: {executed_action}")

            cmd = _action_to_command(
                executed_action,
                forward_vx=args.forward_vx,
                yaw_rate=args.yaw_rate,
                duration=args.duration,
            )
            if cmd is None:
                robot.stop()
                break
            robot.send_motion(cmd)
    except KeyboardInterrupt:
        robot.stop()

    robot.stop()
    print("[STEP] lie down")
    robot.lie_down()
    time.sleep(3.0)
    print("[STEP] passive")
    robot.passive()
    time.sleep(2)
    robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
