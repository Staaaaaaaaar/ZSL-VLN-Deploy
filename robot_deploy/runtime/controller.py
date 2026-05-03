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
        time.sleep(3.0)  # wait for connection to stabilize
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

                    # 打印本次模型输入与输出，便于在终端追踪运行状态
                    try:
                        print(f"[RuntimeController] Turn {turns+1}: instruction='{request_data.instruction}'")
                        print(f"[RuntimeController] Turn {turns+1}: model output: {model_rsp.text}")
                    except Exception:
                        pass

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

                    if active_step_result is not None:
                        active_step_result.planned_actions = list(actions)

                    pending_actions.extend(actions)
                    turns += 1

                action = pending_actions.pop(0)
                proposed_commands = self._actions_to_safe_commands([action])
                final_action, final_commands, was_intervened = self._interactive_review_action(
                    turn_index=turns,
                    step_index=steps + 1,
                    proposed_action=action,
                    proposed_commands=proposed_commands,
                )

                if final_action.kind in {ActionKind.TURN_LEFT, ActionKind.TURN_RIGHT}:
                    continuous_rotation_count += 1
                elif final_action.kind == ActionKind.MOVE_FORWARD:
                    continuous_rotation_count = 0

                if continuous_rotation_count > cfg.early_stop_rotation:
                    stop_reason = "early_stop_rotation"
                    break

                for cmd in final_commands:
                    self.robot.send_motion(cmd)
                    executed_commands.append(cmd)

                if active_step_result is not None:
                    active_step_result.executed_actions.append(final_action)
                    if was_intervened:
                        active_step_result.intervened = True

                steps += 1

                if final_action.kind == ActionKind.STOP:
                    stop_reason = "stopped_by_user" if was_intervened else "stopped_by_model"
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

    def _interactive_review_action(
        self,
        *,
        turn_index: int,
        step_index: int,
        proposed_action: NavigationAction,
        proposed_commands: list[MotionCommand],
    ) -> tuple[NavigationAction, list[MotionCommand], bool]:
        try:
            print(
                f"[RuntimeController][HITL] Turn {turn_index} Step {step_index}: planned action: {proposed_action}"
            )
        except Exception:
            pass

        while True:
            try:
                choice = input("[RuntimeController][HITL] y执行 / n手动 [y]: ")
            except EOFError:
                manual_action = NavigationAction(kind=ActionKind.STOP, raw_text="[manual] stop (EOF)")
                manual_commands = self._actions_to_safe_commands([manual_action])
                try:
                    print(
                        f"[RuntimeController][HITL] Turn {turn_index} Step {step_index}: executed action: {manual_action}"
                    )
                except Exception:
                    pass
                return manual_action, manual_commands, True

            choice = str(choice).strip().lower()
            if choice in {"", "y", "yes"}:
                try:
                    print(
                        f"[RuntimeController][HITL] Turn {turn_index} Step {step_index}: executed action: {proposed_action}"
                    )
                except Exception:
                    pass
                return proposed_action, proposed_commands, False
            if choice in {"n", "no"}:
                manual_action = self._interactive_prompt_manual_action(turn_index=turn_index, step_index=step_index)
                manual_commands = self._actions_to_safe_commands([manual_action])
                try:
                    print(
                        f"[RuntimeController][HITL] Turn {turn_index} Step {step_index}: executed action: {manual_action}"
                    )
                except Exception:
                    pass
                return manual_action, manual_commands, True
            print("[RuntimeController][HITL] 输入无效，请输入 y 或 n。")

    def _interactive_prompt_manual_action(self, *, turn_index: int, step_index: int) -> NavigationAction:
        prompt = f"[RuntimeController][HITL] 1前进 2左 3右 4停: "
        while True:
            try:
                raw = input(prompt)
            except EOFError:
                return NavigationAction(kind=ActionKind.STOP, raw_text="[manual] stop (EOF)")

            raw = str(raw).strip().lower()
            if raw in {"1", "f", "forward", "前进"}:
                return NavigationAction(kind=ActionKind.MOVE_FORWARD, raw_text="[manual] move forward")
            if raw in {"2", "l", "left", "左转"}:
                return NavigationAction(kind=ActionKind.TURN_LEFT, raw_text="[manual] turn left")
            if raw in {"3", "r", "right", "右转"}:
                return NavigationAction(kind=ActionKind.TURN_RIGHT, raw_text="[manual] turn right")
            if raw in {"4", "s", "stop", "停止"}:
                return NavigationAction(kind=ActionKind.STOP, raw_text="[manual] stop")

            print("[RuntimeController][HITL] 输入无效，请输入 1/2/3/4。")

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
            # 1) PIL-like object
            if hasattr(image, "save"):
                try:
                    image.save(image_path)
                    return image_path
                except Exception:
                    pass

            # 2) raw bytes (e.g. JPEG/PNG buffer)
            if isinstance(image, (bytes, bytearray)):
                try:
                    from PIL import Image
                    import io

                    Image.open(io.BytesIO(image)).convert("RGB").save(image_path)
                    return image_path
                except Exception:
                    pass

            # 3) numpy array-like
            try:
                import numpy as np

                arr = None
                if isinstance(image, np.ndarray):
                    arr = image
                elif hasattr(image, "ndim") and hasattr(image, "shape"):
                    try:
                        arr = np.array(image)
                    except Exception:
                        arr = None

                if arr is not None:
                    # normalize floats -> uint8
                    if arr.dtype.kind == "f":
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    elif arr.dtype != np.uint8:
                        arr = arr.astype(np.uint8)

                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    if arr.ndim == 3 and arr.shape[2] == 4:
                        arr = arr[:, :, :3]

                    from PIL import Image

                    Image.fromarray(arr).save(image_path)
                    return image_path
            except Exception:
                pass

            # 4) try buffer/memoryview fallback (for some GStreamer buffer-like objects)
            try:
                from PIL import Image
                import io

                mv = memoryview(image)
                Image.open(io.BytesIO(mv.tobytes())).convert("RGB").save(image_path)
                return image_path
            except Exception:
                pass
        except Exception:
            pass

        try:
            print(f"[RuntimeController] Could not save inference image: type={type(image)} record_id={record_id}")
        except Exception:
            pass
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
