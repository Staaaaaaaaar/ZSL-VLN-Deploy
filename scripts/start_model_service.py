from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_limit_mm(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""

        # Already a JSON object string.
        if raw.startswith("{"):
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("limit_mm_per_prompt JSON must be an object")
            return json.dumps(obj, ensure_ascii=True)

        # Backward-compatible format: "image=200,video=0".
        mm: dict[str, int] = {}
        for item in raw.split(","):
            part = item.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid limit_mm_per_prompt entry: {part}")
            key, val = part.split("=", 1)
            mm[key.strip()] = int(val.strip())

        if not mm:
            return ""
        return json.dumps(mm, ensure_ascii=True)

    raise ValueError("limit_mm_per_prompt must be dict or string")


def build_command(cfg: dict) -> tuple[list[str], list[str]]:
    model_path = str(cfg["model_path"])
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8003))
    warnings: list[str] = []

    cmd = [
        "vllm",
        "serve",
        model_path,
    ]

    # `--task` is unsupported on current vLLM CLI.
    if cfg.get("task") is not None:
        warnings.append("ignore config key 'task' because current vLLM CLI does not support it")

    cmd.extend(["--host", host])
    cmd.extend(["--port", str(port)])

    if bool(cfg.get("trust_remote_code", True)):
        cmd.append("--trust-remote-code")

    limit_mm = cfg.get("limit_mm_per_prompt")
    if limit_mm is not None:
        limit_mm_arg = _normalize_limit_mm(limit_mm)
        if limit_mm_arg:
            cmd.extend(["--limit-mm-per-prompt", limit_mm_arg])

    mm_kwargs = cfg.get("mm_processor_kwargs")
    if mm_kwargs:
        # Follow ActiveVLN README format.
        cmd.extend(["--mm_processor_kwargs", json.dumps(mm_kwargs, ensure_ascii=True)])

    max_model_len = cfg.get("max_model_len")
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    if bool(cfg.get("enable_prefix_caching", True)):
        cmd.append("--enable-prefix-caching")

    # `--disable-log-requests` is unsupported on current vLLM CLI.
    if cfg.get("disable_log_requests") is not None:
        warnings.append("ignore config key 'disable_log_requests' because current vLLM CLI does not support it")

    for item in cfg.get("extra_args", []):
        cmd.append(str(item))

    return cmd, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Start local vLLM model service from config")
    parser.add_argument(
        "--config",
        default=str(Path("robot_deploy") / "configs" / "model_service.json"),
        help="Path to model service config json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print command")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] config not found: {cfg_path}")
        return 2

    cfg = load_config(cfg_path)
    cmd, warnings = build_command(cfg)
    if warnings:
        for line in warnings:
            print(f"[WARN] {line}")
    print("[CMD]", " ".join(cmd))

    if args.dry_run:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
