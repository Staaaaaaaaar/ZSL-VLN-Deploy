from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_deploy.core import NavigationRequest
from robot_deploy.interface import ActiveVLNActionInterface
from robot_deploy.model import ActiveVLNOpenAIModel


def load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_base_url(cfg: dict) -> str:
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8003))
    prefix = str(cfg.get("api_prefix", "/v1")).rstrip("/")
    return f"http://{host}:{port}{prefix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer one step from local image via model service")
    parser.add_argument("--image", required=True, help="Path to local image file")
    parser.add_argument("--instruction", required=True, help="Navigation instruction text")
    parser.add_argument(
        "--service-config",
        default="configs/model_service.json",
        help="Path to model service config",
    )
    parser.add_argument(
        "--runtime-config",
        default="configs/default.json",
        help="Path to runtime config",
    )
    parser.add_argument("--model-name", default=None, help="Optional model name override")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERR] image not found: {args.image}")
        return 2

    service_cfg = load_json(args.service_config)
    runtime_cfg = load_json(args.runtime_config)

    base_url = build_base_url(service_cfg)
    action_space = str(runtime_cfg.get("interface", {}).get("action_space", "r2r"))

    model = ActiveVLNOpenAIModel(
        base_url=base_url,
        api_key="EMPTY",
        model_name=args.model_name,
        action_space=action_space,
        timeout_sec=30.0,
        max_tokens=256,
        temperature=0.2,
        top_p=0.8,
    )
    parser_if = ActiveVLNActionInterface(action_space=action_space)

    req = NavigationRequest(instruction=args.instruction, image=args.image)

    try:
        rsp = model.infer(req)
    except Exception as exc:
        print(f"[ERR] inference failed: {exc}")
        return 1
    finally:
        model.close()

    print("[RAW_TEXT]")
    print(rsp.text)

    actions = parser_if.parse_model_response(rsp.text)
    print("[PARSED_ACTIONS]")
    for idx, a in enumerate(actions, start=1):
        print(f"{idx}. kind={a.kind.value}, value={a.value}, raw={a.raw_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
