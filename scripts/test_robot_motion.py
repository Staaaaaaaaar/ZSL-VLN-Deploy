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
    if robot.check_connection():
        print("[OK] robot connected")
    else:
        print("[ERR] robot not connected")

    print("[STEP] stand up")
    robot.stand_up()
    time.sleep(5)

    sequence = [
        ("forward", MotionCommand(vx=args.forward_vx, vy=0.0, yaw_rate=0.0, duration_sec=args.duration)),
        ("left turn", MotionCommand(vx=0.0, vy=0.0, yaw_rate=args.yaw_rate, duration_sec=args.duration)),
        ("right turn", MotionCommand(vx=0.0, vy=0.0, yaw_rate=-args.yaw_rate, duration_sec=args.duration)),
    ]

    for name, cmd in sequence:
        print(f"[STEP] {name} -> {cmd}")
        robot.send_motion(cmd)
        time.sleep(2)

    robot.stop()
    print("[OK] stop sent")
    time.sleep(2)

    print("[STEP] lie down")
    robot.lie_down()
    time.sleep(3)

    robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
