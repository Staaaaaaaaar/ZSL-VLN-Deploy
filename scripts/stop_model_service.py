from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_pid(pid: int, timeout_sec: float = 10.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return False

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not process_alive(pid):
            return True
        time.sleep(0.5)

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return False
    return not process_alive(pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop background local vLLM model service.")
    parser.add_argument(
        "--pid-file",
        default=str(Path("run") / "model_service.pid"),
        help="Pid file path from start_model_service.py",
    )
    args = parser.parse_args()

    pid_file = Path(args.pid_file)
    if not pid_file.exists():
        print(f"pid file not found: {pid_file}")
        return 2

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        print("invalid pid file content", file=sys.stderr)
        return 3

    if not process_alive(pid):
        print(f"process already stopped, pid={pid}")
        pid_file.unlink(missing_ok=True)
        return 0

    if terminate_pid(pid):
        print(f"service stopped, pid={pid}")
        pid_file.unlink(missing_ok=True)
        return 0

    print(f"failed to stop service, pid={pid}", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
