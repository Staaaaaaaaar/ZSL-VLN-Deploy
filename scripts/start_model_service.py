from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_command(cfg: dict) -> list[str]:
    model_path = cfg["model_path"]
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8003))
    task = cfg.get("task", "generate")

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--task",
        task,
        "--host",
        host,
        "--port",
        str(port),
    ]

    if bool(cfg.get("trust_remote_code", True)):
        cmd.append("--trust-remote-code")

    limit_mm = cfg.get("limit_mm_per_prompt")
    if limit_mm:
        cmd.extend(["--limit-mm-per-prompt", str(limit_mm)])

    mm_kwargs = cfg.get("mm_processor_kwargs")
    if mm_kwargs:
        cmd.extend(["--mm_processor_kwargs", json.dumps(mm_kwargs, ensure_ascii=True)])

    max_model_len = cfg.get("max_model_len")
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    if bool(cfg.get("enable_prefix_caching", True)):
        cmd.append("--enable-prefix-caching")

    if bool(cfg.get("disable_log_requests", True)):
        cmd.append("--disable-log-requests")

    for item in cfg.get("extra_args", []):
        cmd.append(str(item))

    return cmd


def health_url(cfg: dict) -> str:
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8003))
    prefix = str(cfg.get("api_prefix", "/v1")).rstrip("/")
    return f"http://{host}:{port}{prefix}/models"


def wait_service_ready(url: str, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            req = request.Request(url, method="GET")
            with request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if isinstance(payload, dict) and payload.get("data"):
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def save_pid_file(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid), encoding="utf-8")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start local vLLM model inference service.")
    parser.add_argument(
        "--config",
        default=str(Path("robot_deploy") / "configs" / "model_service.json"),
        help="Path to model service config json.",
    )
    parser.add_argument("--foreground", action="store_true", help="Run service in foreground.")
    parser.add_argument("--wait-ready", action="store_true", help="Wait until /models becomes ready.")
    parser.add_argument("--ready-timeout", type=float, default=180.0, help="Ready wait timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only print command without executing.")
    parser.add_argument(
        "--pid-file",
        default=str(Path("run") / "model_service.pid"),
        help="Pid file path used in background mode.",
    )
    parser.add_argument(
        "--log-file",
        default=str(Path("run") / "model_service.log"),
        help="Log file path used in background mode.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config file not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_config(cfg_path)
    cmd = build_command(cfg)
    ready_endpoint = health_url(cfg)

    print("vllm command:")
    print(" ".join(cmd))
    print(f"ready endpoint: {ready_endpoint}")

    if args.dry_run:
        return 0

    if args.foreground:
        if args.wait_ready:
            # Foreground mode cannot block on health before process starts;
            # print endpoint and let caller verify in another shell.
            print("foreground mode: use scripts/check_model_service.py to verify readiness.")
        return subprocess.call(cmd)

    pid_file = Path(args.pid_file)
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            old_pid = -1
        if process_alive(old_pid):
            print(f"service seems already running, pid={old_pid}", file=sys.stderr)
            return 3

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    save_pid_file(pid_file, proc.pid)
    print(f"started model service in background, pid={proc.pid}")
    print(f"log file: {log_file}")
    print(f"pid file: {pid_file}")

    if args.wait_ready:
        ok = wait_service_ready(ready_endpoint, args.ready_timeout)
        if ok:
            print("model service is ready")
            return 0

        print("model service did not become ready within timeout", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
