from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from urllib import request

from robot_deploy.core import ModelAdapter, ModelResponse, NavigationRequest


SYSTEM_PROMPTS = {
    "r2r": (
        "You are a helpful assistant. "
        "Your goal is to follow the given instruction to reach a specified destination. "
        "Your action set is: move forward 25cm, move forward 50cm, move forward 75cm, "
        "turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
        "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop."
    ),
    "rxr": (
        "You are a helpful assistant. "
        "Your goal is to follow the given instruction to reach a specified destination. "
        "Your action set is: move forward 25cm, move forward 50cm, move forward 75cm, "
        "turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, "
        "turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop."
    ),
}


class ActiveVLNOpenAIModel(ModelAdapter):
    """OpenAI-compatible client for ActiveVLN style action generation."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model_name: str | None = None,
        action_space: str = "r2r",
        timeout_sec: float = 30.0,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.8,
    ):
        if action_space not in SYSTEM_PROMPTS:
            raise ValueError("action_space must be one of {'r2r', 'rxr'}")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.action_space = action_space
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._resolved_model_name: str | None = model_name

    def infer(self, request_data: NavigationRequest) -> ModelResponse:
        model_name = self._resolved_model_name or self._resolve_model_name()
        payload = self._build_payload(model_name, request_data)

        endpoint = f"{self.base_url}/chat/completions"
        raw = self._post_json(endpoint, payload)

        text = self._extract_text(raw)
        return ModelResponse(text=text, raw_payload=raw)

    def close(self) -> None:
        return None

    def _resolve_model_name(self) -> str:
        endpoint = f"{self.base_url}/models"
        raw = self._get_json(endpoint)
        data = raw.get("data", [])
        if not data:
            raise RuntimeError("No model found from OpenAI-compatible endpoint /models")
        first = data[0]
        model_id = first.get("id")
        if not model_id:
            raise RuntimeError("Invalid /models payload: missing model id")
        self._resolved_model_name = model_id
        return model_id

    def _build_payload(self, model_name: str, req: NavigationRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": f"Instruction: {req.instruction}"})
        content.append(
            {
                "type": "text",
                "text": "Decide your next action. You can take up to 3 actions at a time, separated by ','.",
            }
        )

        image_b64 = self._try_encode_image(req.image)
        if image_b64 is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[self.action_space]}]},
            {"role": "user", "content": content},
        ]

        return {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _extract_text(self, raw: dict[str, Any]) -> str:
        choices = raw.get("choices", [])
        if not choices:
            raise RuntimeError("Invalid completion response: missing choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return " ".join(t for t in texts if t).strip()

        return str(content).strip()

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, endpoint: str) -> dict[str, Any]:
        req = request.Request(
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="GET",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _try_encode_image(self, image: Any) -> str | None:
        if image is None:
            return None

        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, str):
                pil_image = Image.open(image)
            elif hasattr(image, "shape") and hasattr(image, "astype"):
                # Keep ndarray compatibility without hard importing numpy.
                pil_image = Image.fromarray(image.astype("uint8"))
            else:
                return None

            buf = BytesIO()
            pil_image.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None
