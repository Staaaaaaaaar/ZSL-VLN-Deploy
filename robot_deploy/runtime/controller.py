from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from robot_deploy.core import (
    ActionKind,
    ActionInterface,
    ModelAdapter,
    MotionCommand,
    NavigationAction,
    NavigationRequest,
    RobotAdapter,
    RobotEndpoint,
    RuntimeEpisodeConfig,
    RuntimeEpisodeResult,
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
        # if not self.robot.check_connection():
        #     raise RuntimeError("Robot connection check failed after initRobot")
        if self.policy.auto_stand_up:
            self.robot.stand_up()

    def run_once(self, request_data: NavigationRequest) -> RuntimeStepResult:
        # if not self.robot.check_connection():
        #     return RuntimeStepResult(ok=False, error="robot disconnected")

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

    def run_episode(
        self,
        request_provider: Callable[[], NavigationRequest],
        episode_config: RuntimeEpisodeConfig | None = None,
        goal_checker: Callable[[NavigationRequest], bool] | None = None,
    ) -> RuntimeEpisodeResult:
        """Run a continuous closed-loop VLN episode on real hardware.

        The control flow follows ActiveVLN's rollout style:
        model-turn loop + action-step loop + budget/early-stop constraints.
        """
        cfg = episode_config or RuntimeEpisodeConfig()

        pending_actions: list[NavigationAction] = []
        model_outputs: list[str] = []
        executed_commands: list[MotionCommand] = []

        turns = 0
        steps = 0
        continuous_rotation_count = 0
        stop_reason = ""

        try:
            while True:
                if cfg.stop_on_disconnect and not self.robot.check_connection():
                    stop_reason = "robot_disconnected"
                    return RuntimeEpisodeResult(
                        ok=False,
                        stop_reason=stop_reason,
                        turns=turns,
                        steps=steps,
                        model_outputs=model_outputs,
                        executed_commands=executed_commands,
                        error="robot disconnected",
                    )

                if steps >= cfg.max_step_budget:
                    stop_reason = "episode_steps_exceeded"
                    break

                if not pending_actions:
                    if turns >= cfg.max_turn_budget:
                        stop_reason = "episode_turns_exceeded"
                        break

                    request_data = request_provider()
                    model_rsp = self.model.infer(request_data)
                    model_outputs.append(model_rsp.text)

                    actions = self.action_interface.parse_model_response(model_rsp.text)
                    actions = actions[: max(1, cfg.max_actions_per_turn)]
                    if not actions:
                        actions = [NavigationAction(kind=ActionKind.STOP, raw_text=model_rsp.text)]

                    pending_actions.extend(actions)
                    turns += 1

                action = pending_actions.pop(0)
                if action.kind in {ActionKind.TURN_LEFT, ActionKind.TURN_RIGHT}:
                    continuous_rotation_count += 1
                elif action.kind == ActionKind.MOVE_FORWARD:
                    continuous_rotation_count = 0

                if continuous_rotation_count > cfg.early_stop_rotation:
                    stop_reason = "early_stop_rotation"
                    break

                safe_commands = self._actions_to_safe_commands([action])
                for cmd in safe_commands:
                    self.robot.send_motion(cmd)
                    executed_commands.append(cmd)

                steps += 1

                if action.kind == ActionKind.STOP:
                    request_data = request_provider()
                    if goal_checker is not None and goal_checker(request_data):
                        stop_reason = "success"
                    else:
                        stop_reason = "stopped_goal_not_confirmed"
                    break

            try:
                self.robot.stop()
            except Exception:
                pass

            return RuntimeEpisodeResult(
                ok=stop_reason == "success",
                stop_reason=stop_reason,
                turns=turns,
                steps=steps,
                model_outputs=model_outputs,
                executed_commands=executed_commands,
            )
        except Exception as exc:
            if self.policy.emergency_stop_on_error:
                try:
                    self.robot.stop()
                except Exception:
                    pass
            return RuntimeEpisodeResult(
                ok=False,
                stop_reason="runtime_error",
                turns=turns,
                steps=steps,
                model_outputs=model_outputs,
                executed_commands=executed_commands,
                error=str(exc),
            )

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

    def _actions_to_safe_commands(self, actions: list[NavigationAction]) -> list[MotionCommand]:
        raw_commands = self.action_interface.to_motion_commands(actions)
        return [self._apply_safety(cmd) for cmd in raw_commands]

    def _apply_safety(self, cmd: MotionCommand) -> MotionCommand:
        vx = self._apply_axis_limit(cmd.vx, self.safety.min_nonzero_vx, self.safety.max_vx)
        vy = self._apply_axis_limit(cmd.vy, self.safety.min_nonzero_vy, self.safety.max_vy)
        yaw = self._apply_axis_limit(cmd.yaw_rate, self.safety.min_nonzero_yaw_rate, self.safety.max_yaw_rate)
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

    @classmethod
    def _apply_axis_limit(cls, value: float, min_nonzero: float, max_abs: float) -> float:
        clamped = cls._clamp(value, -max_abs, max_abs)
        if clamped == 0.0:
            return 0.0

        # SDK rejects tiny non-zero values; map dead-zone to full stop.
        if abs(clamped) < min_nonzero:
            return 0.0
        return math.copysign(abs(clamped), clamped)
