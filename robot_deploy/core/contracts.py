from __future__ import annotations

from abc import ABC, abstractmethod

from .types import (
    ModelResponse,
    MotionCommand,
    NavigationAction,
    NavigationRequest,
    RobotEndpoint,
    RobotState,
)


class RobotAdapter(ABC):
    @abstractmethod
    def connect(self, endpoint: RobotEndpoint) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_connection(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stand_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def lie_down(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_motion(self, cmd: MotionCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ModelAdapter(ABC):
    @abstractmethod
    def infer(self, request: NavigationRequest) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ActionInterface(ABC):
    @abstractmethod
    def parse_model_response(self, response_text: str) -> list[NavigationAction]:
        raise NotImplementedError

    @abstractmethod
    def to_motion_commands(self, actions: list[NavigationAction]) -> list[MotionCommand]:
        raise NotImplementedError
