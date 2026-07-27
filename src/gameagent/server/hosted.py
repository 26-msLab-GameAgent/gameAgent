"""HTTP clients for hosted multimodal model providers."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def generate_hosted(
    provider: str,
    model: str,
    prompt: str,
    image_path: Path,
    include_image: bool,
    max_tokens: int,
    timeout_s: float,
) -> str:
    """Generate text with one of the supported hosted providers."""
    image_b64 = _image_base64(image_path) if include_image else None
    media_type = _media_type(image_path)
    if provider == "openai":
        return _generate_openai(model, prompt, image_b64, media_type, max_tokens, timeout_s)
    if provider == "openrouter":
        return _generate_openrouter(
            model,
            prompt,
            image_b64,
            media_type,
            max_tokens,
            timeout_s,
        )
    if provider == "gemini":
        return _generate_gemini(model, prompt, image_b64, media_type, max_tokens, timeout_s)
    if provider == "anthropic":
        return _generate_anthropic(model, prompt, image_b64, media_type, max_tokens, timeout_s)
    if provider == "ollama":
        return _generate_ollama(model, prompt, image_b64, max_tokens, timeout_s)
    raise ValueError(f"Unsupported hosted provider: {provider}")


def _generate_ollama(
    model: str,
    prompt: str,
    image_b64: str | None,
    max_tokens: int,
    timeout_s: float,
) -> str:
    """Generate JSON text through a local Ollama server."""
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_b64:
        message["images"] = [image_b64]

    data = _post_json(
        os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        + "/api/chat",
        {
            "model": model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "options": {
                # LLaVA tends to restate visual details before completing the
                # requested object. Avoid truncating a valid JSON response at
                # the server-wide 128-token default.
                "num_predict": max(max_tokens, 512),
                "temperature": 0,
            },
        },
        {"Content-Type": "application/json"},
        timeout_s,
    )
    return _extract_ollama_output_text(data)


def _generate_openrouter(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _required_env("OPENROUTER_API_KEY", "openrouter")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            }
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url := os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = site_url
    if app_name := os.environ.get("OPENROUTER_APP_NAME"):
        headers["X-OpenRouter-Title"] = app_name

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    _apply_openrouter_model_options(payload, model)

    data = _post_json(
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        + "/chat/completions",
        payload,
        headers,
        timeout_s,
    )
    return _extract_openrouter_output_text(data)


def _is_openrouter_claude(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("~anthropic/claude") or normalized.startswith(
        "anthropic/claude"
    )


def _is_openrouter_gemini(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("~google/gemini") or normalized.startswith(
        "google/gemini"
    )


def _apply_openrouter_model_options(payload: dict[str, Any], model: str) -> None:
    if _is_openrouter_claude(model):
        # Claude reasoning uses at least 1,024 tokens. Keep enough additional
        # output budget for the compact JSON answer required by the pipeline.
        payload["max_tokens"] = max(int(payload.get("max_tokens", 0)), 4096)
        payload["reasoning"] = {"max_tokens": 1024, "exclude": True}
        payload["response_format"] = {"type": "json_object"}
    elif _is_openrouter_gemini(model):
        # Current Gemini Flash aliases enable thinking by default. Keep it at
        # the lowest supported level so the compact JSON answer retains budget.
        payload["reasoning"] = {"effort": "minimal", "exclude": True}
        payload["response_format"] = {"type": "json_object"}


def _generate_openai(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _required_env("OPENAI_API_KEY", "openai")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if image_b64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_b64}",
            }
        )
    data = _post_json(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/responses",
        {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
        },
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout_s,
    )
    text = data.get("output_text")
    return text.strip() if isinstance(text, str) else _extract_openai_output_text(data)


def _generate_gemini(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _env_first("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY")
    input_items: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64:
        input_items.append({"type": "image", "data": image_b64, "mime_type": media_type})
    data = _post_json(
        os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        + "/interactions",
        {
            "model": model,
            "input": input_items,
            "generation_config": {"max_output_tokens": max_tokens},
        },
        {"x-goog-api-key": api_key, "Content-Type": "application/json"},
        timeout_s,
    )
    text = data.get("output_text")
    return text.strip() if isinstance(text, str) else _extract_gemini_output_text(data)


def _generate_anthropic(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _env_first("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("anthropic provider requires ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    content: list[dict[str, Any]] = []
    if image_b64:
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            }
        )
    content.append({"type": "text", "text": prompt})
    data = _post_json(
        os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        + "/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    return _extract_anthropic_output_text(data)


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body}") from exc


def _image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _media_type(image_path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(image_path))
    return guessed or "image/png"


def _required_env(name: str, provider: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{provider} provider requires {name}")
    return value


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _extract_openai_output_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise ValueError(f"OpenAI response did not contain output text: {str(data)[:500]}")
    return "\n".join(parts).strip()


def _extract_ollama_output_text(data: dict[str, Any]) -> str:
    message = data.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Ollama response did not contain a message: {str(data)[:500]}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Ollama response did not contain output text: {str(data)[:500]}")
    return content.strip()


def _extract_openrouter_output_text(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        raise ValueError(f"OpenRouter response did not contain choices: {str(data)[:500]}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"OpenRouter response did not contain a message: {str(data)[:500]}")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    raise ValueError(f"OpenRouter response did not contain output text: {str(data)[:500]}")


def _extract_gemini_output_text(data: dict[str, Any]) -> str:
    for key in ("text", "response_text"):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip()
    output = data.get("output")
    if isinstance(output, list):
        parts = [
            item.get("text")
            for item in output
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    raise ValueError(f"Gemini response did not contain output text: {str(data)[:500]}")


def _extract_anthropic_output_text(data: dict[str, Any]) -> str:
    parts = [
        item.get("text")
        for item in data.get("content", []) or []
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    if not parts:
        raise ValueError(f"Anthropic response did not contain output text: {str(data)[:500]}")
    return "\n".join(parts).strip()
