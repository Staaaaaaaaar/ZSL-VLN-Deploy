from __future__ import annotations

from dataclasses import dataclass

from robot_deploy.core import (
    ActionInterface,
    ModelAdapter,
    MotionCommand,
    NavigationRequest,
    RobotAdapter,
    RobotEndpoint,
    RuntimeSafetyLimits,
    RuntimeStepResult,
)


@dataclass
class RuntimePolicy:
    auto_stand_up: bool = True
    auto_lie_down_on_shutdown: bool = True
    emergency_stop_on_error: bool = True


class RuntimeController:
    """Single robot single-process runtime with built-in safety controls."""

    def __init__(
        self,
        robot: RobotAdapter,
        model: ModelAdapter,
        action_interface: ActionInterface,
        safety: RuntimeSafetyLimits | None = None,
        policy: RuntimePolicy | None = None,
    ):
        self.robot = robot
        self.model = model
        self.action_interface = action_interface
        self.safety = safety or RuntimeSafetyLimits()
        self.policy = policy or RuntimePolicy()

    def startup(self, endpoint: RobotEndpoint) -> None:
        self.robot.connect(endpoint)
        if not self.robot.check_connection():
            raise RuntimeError("Robot connection check failed after initRobot")
        if self.policy.auto_stand_up:
            self.robot.stand_up()

    def run_once(self, request_data: NavigationRequest) -> RuntimeStepResult:
        if not self.robot.check_connection():
            return RuntimeStepResult(ok=False, error="robot disconnected")

        try:
            model_rsp = self.model.infer(request_data)
            actions = self.action_interface.parse_model_response(model_rsp.text)
            actions = actions[: self.safety.max_actions_per_cycle]

            raw_commands = self.action_interface.to_motion_commands(actions)
            safe_commands = [self._apply_safety(cmd) for cmd in raw_commands]

            executed: list[MotionCommand] = []
            for cmd in safe_commands:
                self.robot.send_motion(cmd)
                executed.append(cmd)
                if cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.yaw_rate == 0.0:
                    break

            return RuntimeStepResult(ok=True, model_text=model_rsp.text, executed_commands=executed)
        except Exception as exc:
            if self.policy.emergency_stop_on_error:
                try:
                    self.robot.stop()
                except Exception:
                    pass
            return RuntimeStepResult(ok=False, error=str(exc))

    def shutdown(self) -> None:
        try:
            self.robot.stop()
            if self.policy.auto_lie_down_on_shutdown:
                self.robot.lie_down()
        finally:
            try:
                self.model.close()
            finally:
                self.robot.close()

    def _apply_safety(self, cmd: MotionCommand) -> MotionCommand:
        vx = self._clamp(cmd.vx, -self.safety.max_vx, self.safety.max_vx)
        vy = self._clamp(cmd.vy, -self.safety.max_vy, self.safety.max_vy)
        yaw = self._clamp(cmd.yaw_rate, -self.safety.max_yaw_rate, self.safety.max_yaw_rate)
        duration = self._clamp(cmd.duration_sec, 0.0, self.safety.max_command_duration_sec)

        return MotionCommand(
            vx=vx,
            vy=vy,
            yaw_rate=yaw,
            duration_sec=duration,
            source_action=cmd.source_action,
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
