from __future__ import annotations

import math
import re

from robot_deploy.core import (
    ActionInterface,
    ActionKind,
    MotionCommand,
    NavigationAction,
)


class ActiveVLNActionInterface(ActionInterface):
    """Translate model text actions into normalized robot motion commands."""

    def __init__(self, action_space: str = "r2r", forward_vx: float = 0.2, yaw_rate: float = 0.5):
        if action_space not in {"r2r", "rxr"}:
            raise ValueError("action_space must be either 'r2r' or 'rxr'")
        self.action_space = action_space
        self.forward_vx = forward_vx
        self.yaw_rate = yaw_rate

        self._turn_step_deg = 15 if action_space == "r2r" else 30
        self._allowed_forward = {25, 50, 75}
        self._allowed_turn = {self._turn_step_deg, self._turn_step_deg * 2, self._turn_step_deg * 3}

    def parse_model_response(self, response_text: str) -> list[NavigationAction]:
        if not response_text:
            return [NavigationAction(kind=ActionKind.STOP, raw_text="")]

        cleaned = response_text.lower().strip()
        cleaned = cleaned.replace("\n", ",")
        cleaned = cleaned.replace("，", ",")
        cleaned = cleaned.replace("；", ",")
        cleaned = cleaned.replace(";", ",")

        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if not parts:
            return [NavigationAction(kind=ActionKind.STOP, raw_text=cleaned)]

        actions: list[NavigationAction] = []
        for part in parts:
            action = self._parse_single_action(part)
            actions.append(action)
        return actions

    def to_motion_commands(self, actions: list[NavigationAction]) -> list[MotionCommand]:
        commands: list[MotionCommand] = []
        for action in actions:
            if action.kind == ActionKind.STOP:
                commands.append(MotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, duration_sec=0.0, source_action=action))
                continue

            if action.kind == ActionKind.MOVE_FORWARD:
                distance_cm = int(action.value or 25)
                distance_m = max(distance_cm, 1) / 100.0
                duration = distance_m / max(abs(self.forward_vx), 1e-6)
                commands.append(
                    MotionCommand(
                        vx=self.forward_vx,
                        vy=0.0,
                        yaw_rate=0.0,
                        duration_sec=duration,
                        source_action=action,
                    )
                )
                continue

            turn_deg = float(action.value or self._turn_step_deg)
            turn_rad = math.radians(abs(turn_deg))
            duration = turn_rad / max(abs(self.yaw_rate), 1e-6)
            yaw = self.yaw_rate if action.kind == ActionKind.TURN_LEFT else -self.yaw_rate
            commands.append(
                MotionCommand(
                    vx=0.0,
                    vy=0.0,
                    yaw_rate=yaw,
                    duration_sec=duration,
                    source_action=action,
                )
            )
        return commands

    def _parse_single_action(self, text: str) -> NavigationAction:
        if "stop" in text:
            return NavigationAction(kind=ActionKind.STOP, raw_text=text)

        if "forward" in text:
            value = self._extract_numeric(text, default=25)
            value = self._closest_allowed(value, self._allowed_forward)
            return NavigationAction(kind=ActionKind.MOVE_FORWARD, value=float(value), raw_text=text)

        if "left" in text:
            value = self._extract_numeric(text, default=self._turn_step_deg)
            value = self._closest_allowed(value, self._allowed_turn)
            return NavigationAction(kind=ActionKind.TURN_LEFT, value=float(value), raw_text=text)

        if "right" in text:
            value = self._extract_numeric(text, default=self._turn_step_deg)
            value = self._closest_allowed(value, self._allowed_turn)
            return NavigationAction(kind=ActionKind.TURN_RIGHT, value=float(value), raw_text=text)

        return NavigationAction(kind=ActionKind.STOP, raw_text=text)

    @staticmethod
    def _extract_numeric(text: str, default: int) -> int:
        m = re.search(r"-?\d+", text)
        if m is None:
            return default
        return abs(int(m.group(0)))

    @staticmethod
    def _closest_allowed(value: int, allowed: set[int]) -> int:
        return min(allowed, key=lambda x: abs(x - value))
