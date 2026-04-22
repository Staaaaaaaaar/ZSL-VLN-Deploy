from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Callable

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
    stand_up_settle_sec: float = 3.0
    inter_command_gap_sec: float = 0.1
    save_inference_info: bool = False
    inference_save_dir: str = "runtime_logs/inference"


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
            if self.policy.stand_up_settle_sec > 0:
                time.sleep(self.policy.stand_up_settle_sec)

    def run_episode(
        self,
        request_provider: Callable[[], NavigationRequest],
        episode_config: RuntimeEpisodeConfig | None = None,
    ) -> RuntimeEpisodeResult:
        """Run a continuous closed-loop VLN episode on real hardware.

        The control flow follows ActiveVLN's rollout style:
        model-turn loop + action-step loop + budget/early-stop constraints.
        """
        cfg = episode_config or RuntimeEpisodeConfig()

        pending_actions: list[NavigationAction] = []
        step_results: list[RuntimeStepResult] = []
        model_outputs: list[str] = []
        executed_commands: list[MotionCommand] = []
        active_step_result: RuntimeStepResult | None = None

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
                        step_results=step_results,
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

                    active_step_result = RuntimeStepResult(
                        ok=True,
                        instruction=request_data.instruction,
                        img=request_data.image,
                        model_text=model_rsp.text,
                    )
                    step_results.append(active_step_result)
                    self._save_inference(turn_index=turns + 1, step_result=active_step_result)

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
                    self._execute_motion_with_gap(cmd)
                    executed_commands.append(cmd)
                    if active_step_result is not None:
                        active_step_result.executed_commands.append(cmd)

                steps += 1

                if action.kind == ActionKind.STOP:
                    stop_reason = "stopped_by_model"
                    break

            try:
                self.robot.stop()
            except Exception:
                pass

            return RuntimeEpisodeResult(
                ok=True,
                stop_reason=stop_reason,
                turns=turns,
                steps=steps,
                step_results=step_results,
                model_outputs=model_outputs,
                executed_commands=executed_commands,
            )
        except Exception as exc:
            if active_step_result is not None and active_step_result.error is None:
                active_step_result.ok = False
                active_step_result.error = str(exc)
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
                step_results=step_results,
                model_outputs=model_outputs,
                executed_commands=executed_commands,
                error=str(exc),
            )

    def shutdown(self) -> None:
        try:
            try:
                self.robot.stop()
            except Exception:
                pass
            if self.policy.auto_lie_down_on_shutdown:
                self.robot.lie_down()
        finally:
            try:
                self.model.close()
            finally:
                self.robot.close()

    def _execute_motion_with_gap(self, cmd: MotionCommand) -> None:
        self.robot.send_motion(cmd)
        if self.policy.inter_command_gap_sec > 0:
            time.sleep(self.policy.inter_command_gap_sec)

    def _save_inference(self, turn_index: int, step_result: RuntimeStepResult) -> None:
        if not self.policy.save_inference_info:
            return

        save_dir = self.policy.inference_save_dir.strip()
        if not save_dir:
            return

        os.makedirs(save_dir, exist_ok=True)
        ts_ms = int(time.time() * 1000)
        record_id = f"turn_{turn_index:04d}_{ts_ms}"
        image_path = self._save_inference_image(save_dir=save_dir, record_id=record_id, image=step_result.img)

        payload = {
            "turn": turn_index,
            "timestamp_ms": ts_ms,
            "instruction": step_result.instruction,
            "model_text": step_result.model_text,
            "ok": step_result.ok,
            "error": step_result.error,
            "image_path": image_path,
        }
        json_path = os.path.join(save_dir, f"{record_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_inference_image(self, save_dir: str, record_id: str, image: Any | None) -> str | None:
        if image is None:
            return None

        image_path = os.path.join(save_dir, f"{record_id}.png")
        try:
            if hasattr(image, "save"):
                image.save(image_path)
                return image_path

            from PIL import Image

            rgb = image
            if hasattr(image, "ndim") and hasattr(image, "shape") and getattr(image, "ndim", 0) == 3:
                if image.shape[2] >= 3:
                    rgb = image[:, :, :3]
            Image.fromarray(rgb).save(image_path)
            return image_path
        except Exception:
            return None

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
