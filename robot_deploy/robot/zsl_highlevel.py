from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from typing import Any

from robot_deploy.core import MotionCommand, RobotAdapter, RobotEndpoint, RobotState


class ZSLHighLevelRobot(RobotAdapter):
    """High-level robot adapter based on https://github.com/zsibot/zsibot_sdk Python extension."""

    def __init__(self, robot_model: str = "zsl-1", sdk_root: str | None = None):
        if robot_model not in {"zsl-1", "zsl-1w"}:
            raise ValueError("robot_model must be one of {'zsl-1', 'zsl-1w'}")

        self.robot_model = robot_model
        self.sdk_root = sdk_root or os.path.join("third_party", "zsibot_sdk")
        self._sdk_module = self._load_sdk_module()
        self._robot: Any = self._sdk_module.HighLevel()
        self._connected = False

    def connect(self, endpoint: RobotEndpoint) -> None:
        self._robot.initRobot(endpoint.local_ip, endpoint.local_port, endpoint.dog_ip)
        self._connected = True

    def check_connection(self) -> bool:
        if not self._connected:
            return False
        
        return self._robot.checkConnect()

    def stand_up(self) -> None:
        self._robot.standUp()

    def lie_down(self) -> None:
        self._robot.lieDown()

    def send_motion(self, cmd: MotionCommand) -> None:
        self._robot.move(cmd.vx, cmd.vy, cmd.yaw_rate)
        if cmd.duration_sec > 0:
            time.sleep(cmd.duration_sec)
            self.stop()

    def stop(self) -> None:
        self._robot.move(0.0, 0.0, 0.0)

    def read_state(self) -> RobotState:
        return RobotState(
            connected=self.check_connection(),
            battery_power=self._safe_call("getBatteryPower"),
            rpy=self._safe_vector3("getRPY"),
            body_velocity=self._safe_vector3("getBodyVelocity"),
        )

    def close(self) -> None:
        try:
            self.lie_down()
        except Exception:
            pass
        self._connected = False

    def _load_sdk_module(self):
        arch = platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")
        lib_path = os.path.abspath(os.path.join(self.sdk_root, "lib", self.robot_model, arch))
        if not os.path.isdir(lib_path):
            raise FileNotFoundError(f"SDK library path not found: {lib_path}")

        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)

        module_name = "mc_sdk_zsl_1_py" if self.robot_model == "zsl-1" else "mc_sdk_zsl_1w_py"
        return importlib.import_module(module_name)

    def _safe_call(self, method_name: str):
        if not hasattr(self._robot, method_name):
            return None
        try:
            return getattr(self._robot, method_name)()
        except Exception:
            return None

    def _safe_vector3(self, method_name: str):
        value = self._safe_call(method_name)
        if value is None:
            return None
        try:
            if len(value) < 3:
                return None
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None
