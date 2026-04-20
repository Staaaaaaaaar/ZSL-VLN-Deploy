from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class ActionKind(str, Enum):
    STOP = "stop"
    MOVE_FORWARD = "move forward"
    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"


@dataclass(frozen=True)
class RobotEndpoint:
    local_ip: str
    local_port: int
    dog_ip: str


@dataclass
class NavigationRequest:
    instruction: str
    image: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationAction:
    kind: ActionKind
    value: float | None = None
    raw_text: str = ""


@dataclass
class ModelResponse:
    text: str
    actions: list[NavigationAction] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    duration_sec: float = 0.0
    source_action: NavigationAction | None = None


@dataclass
class RobotState:
    connected: bool
    battery_power: int | None = None
    rpy: tuple[float, float, float] | None = None
    body_velocity: tuple[float, float, float] | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RuntimeSafetyLimits:
    # SDK valid ranges:
    # vx: (-3 ~ -0.05) U (0.05 ~ 3)
    # vy: (-1 ~ -0.1)  U (0.1 ~ 1)
    # yaw: (-3 ~ -0.02) U (0.02 ~ 3)
    max_vx: float = 3.0
    max_vy: float = 1.0
    max_yaw_rate: float = 3.0
    min_nonzero_vx: float = 0.05
    min_nonzero_vy: float = 0.1
    min_nonzero_yaw_rate: float = 0.02
    max_command_duration_sec: float = 2.0
    max_actions_per_cycle: int = 3


@dataclass
class RuntimeStepResult:
    ok: bool
    model_text: str = ""
    executed_commands: list[MotionCommand] = field(default_factory=list)
    error: str | None = None


@dataclass
class RuntimeEpisodeConfig:
    # Max number of model inference turns.
    max_turn_budget: int = 40
    # Max number of executed primitive actions.
    max_step_budget: int = 120
    # Max number of parsed actions accepted from one model response.
    max_actions_per_turn: int = 3
    # Early stop if the agent keeps rotating without moving forward.
    early_stop_rotation: int = 12
    # Whether to fail fast on robot disconnects during runtime.
    stop_on_disconnect: bool = True


@dataclass
class RuntimeEpisodeResult:
    ok: bool
    stop_reason: str
    turns: int
    steps: int
    model_outputs: list[str] = field(default_factory=list)
    executed_commands: list[MotionCommand] = field(default_factory=list)
    error: str | None = None
