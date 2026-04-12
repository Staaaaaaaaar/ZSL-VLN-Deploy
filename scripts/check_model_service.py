from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib import error, request


def load_service_url(config_path: Path | None, base_url_arg: str | None) -> str:
    if base_url_arg:
        return base_url_arg.rstrip("/")

    if config_path is None:
        return "http://127.0.0.1:8003/v1"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8003))
    prefix = str(cfg.get("api_prefix", "/v1")).rstrip("/")
    return f"http://{host}:{port}{prefix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local model service health and model listing.")
    parser.add_argument(
        "--config",
        default=str(Path("robot_deploy") / "configs" / "model_service.json"),
        help="Path to model service config json.",
    )
    parser.add_argument("--base-url", default=None, help="Override base url, e.g. http://127.0.0.1:8003/v1")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    base_url = load_service_url(config_path if config_path and config_path.exists() else None, args.base_url)
    url = f"{base_url}/models"

    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=10.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        print(f"service unreachable: {url}")
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"request failed: {url}")
        print(str(exc), file=sys.stderr)
        return 3

    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not data:
        print("service responded but model list is empty")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 4

    print(f"service ready: {url}")
    print("models:")
    for item in data:
        print(f"- {item.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
