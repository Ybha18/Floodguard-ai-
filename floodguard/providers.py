from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import SETTINGS


@dataclass
class ModelStatus:
    name: str
    provider: str
    configured: bool
    online: bool
    message: str
    latency_ms: float | None = None


# Nemotron/Muse endpoints can expose internal reasoning inline in content or in a
# dedicated reasoning field depending on the serving configuration. The operator
# must never see raw reasoning traces; only the model's visible answer is used.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_UNCLOSED_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def _strip_reasoning(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_BLOCK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()


def _post_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout: float = 60.0,
    enable_thinking: bool | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    top_p: float = 0.95,
    thinking_token_budget: int | None = None,
) -> tuple[str, float]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if thinking_token_budget is not None:
        payload["thinking_token_budget"] = thinking_token_budget

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]

    # IMPORTANT: Never fall back to reasoning/reasoning_content. If visible content
    # is empty, the provider did not produce a usable operator-facing answer.
    content = str(message.get("content") or "").strip()
    content = _strip_reasoning(content)
    return content, (time.perf_counter() - started) * 1000


def call_nemotron(prompt: str, *, role: str = "primary") -> tuple[str, float]:
    if not SETTINGS.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    if role == "synthesis":
        model = SETTINGS.nvidia_synthesis_model
        # Super remains a reasoning model, but low-effort thinking reduces the
        # chance of spending the entire generation budget before producing the
        # required visible JSON synthesis.
        result = _post_openai_compatible(
            "https://integrate.api.nvidia.com/v1",
            SETTINGS.nvidia_api_key,
            model,
            prompt,
            timeout=90.0,
            enable_thinking=True,
            max_tokens=16384,
            temperature=1.0,
            top_p=0.95,
        )
        return result

    model = SETTINGS.nvidia_primary_model
    return _post_openai_compatible(
        "https://integrate.api.nvidia.com/v1",
        SETTINGS.nvidia_api_key,
        model,
        prompt,
        timeout=75.0,
        enable_thinking=True,
        max_tokens=16384,
        temperature=1.0,
        top_p=0.95,
        thinking_token_budget=8192,
    )


def call_muse(prompt: str) -> tuple[str, float]:
    if not SETTINGS.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")
    return _post_openai_compatible(
        "https://integrate.api.nvidia.com/v1",
        SETTINGS.nvidia_api_key,
        SETTINGS.nvidia_vision_model,
        prompt,
        timeout=75.0,
        enable_thinking=True,
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
    )


def call_gemini(prompt: str) -> tuple[str, float]:
    if not SETTINGS.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    started = time.perf_counter()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{SETTINGS.gemini_model}:generateContent"
    response = requests.post(
        url,
        params={"key": SETTINGS.gemini_api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "agreement": {
                            "type": "STRING",
                            "enum": ["CONSENSUS", "PARTIAL_AGREEMENT", "DISAGREEMENT"],
                        },
                        "analysis": {"type": "STRING"},
                    },
                    "required": ["agreement", "analysis"],
                },
            },
        },
        timeout=45,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    return _strip_reasoning(text), (time.perf_counter() - started) * 1000


def configured_status() -> list[ModelStatus]:
    return [
        ModelStatus(
            name=SETTINGS.nvidia_primary_model,
            provider="NVIDIA / Nemotron — primary analysis",
            configured=bool(SETTINGS.nvidia_api_key),
            online=False,
            message="Primary evidence analysis" if SETTINGS.nvidia_api_key else "API key missing",
        ),
        ModelStatus(
            name=SETTINGS.gemini_model,
            provider="Google / Gemini Flash — independent second opinion",
            configured=bool(SETTINGS.gemini_api_key and SETTINGS.enable_gemini),
            online=False,
            message="Independent cross-check" if SETTINGS.gemini_api_key and SETTINGS.enable_gemini else "Disabled / not configured",
        ),
        ModelStatus(
            name=SETTINGS.nvidia_vision_model,
            provider="NVIDIA / Muse Glimmer — optional model review",
            configured=bool(SETTINGS.nvidia_api_key and SETTINGS.enable_muse),
            online=False,
            message="Additional independent review" if SETTINGS.nvidia_api_key and SETTINGS.enable_muse else "Optional / disabled",
        ),
        ModelStatus(
            name=SETTINGS.nvidia_synthesis_model,
            provider="NVIDIA / Nemotron Super — cross-check + final synthesis",
            configured=bool(SETTINGS.nvidia_api_key),
            online=False,
            message="Final grounded synthesis" if SETTINGS.nvidia_api_key else "API key missing",
        ),
    ]
