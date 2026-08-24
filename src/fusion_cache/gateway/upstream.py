"""Upstream adapters for the fusion-cache gateway.

Each adapter knows how to call one provider family and normalize the response
into the OpenAI chat-completions shape the cache pipeline expects.  The
gateway picks an adapter from ``FUSION_UPSTREAM_PROVIDER`` (default
``openai``), falling back to ``openai`` for anything unrecognized.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, Optional

from httpx import AsyncClient, Response

DEFAULT_TIMEOUT = 120.0


def _headers_for(provider: str) -> Dict[str, str]:
    """Build auth headers for the given provider family."""
    if provider == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("FUSION_UPSTREAM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
        }
        return headers
    # openai-compatible (openai / deepseek / others)
    api_key = os.environ.get("FUSION_UPSTREAM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_body(provider: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an OpenAI-style request into the provider's body shape."""
    if provider == "anthropic":
        # OpenAI messages: [{"role": "user", "content": "..."}]
        # Anthropic messages: same roles, system extracted into top-level field.
        messages = list(kwargs.get("messages", []))
        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        messages = [m for m in messages if m.get("role") != "system"]
        body: Dict[str, Any] = {
            "model": kwargs.get("model", "claude-3-5-sonnet-latest"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        if system:
            body["system"] = system
        if kwargs.get("temperature") is not None:
            body["temperature"] = kwargs["temperature"]
        return body
    return kwargs


def _response_to_openai(provider: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a provider response into the OpenAI chat-completion shape."""
    if provider == "anthropic":
        # Anthropic: {"content": [{"type":"text","text":"..."}], "usage": {...}, "stop_reason": "end_turn"}
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
        usage = body.get("usage", {}) or {}
        return {
            "id": body.get("id", "msg-anthropic"),
            "object": "chat.completion",
            "created": int(body.get("created", 0) or 0) or 1_700_000_000,
            "model": body.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _anthropic_stop(body.get("stop_reason", "")),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                # Anthropic does not expose prompt_cache_hit/miss the same way;
                # leave them 0 (honest — no prefix accounting available).
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": usage.get("input_tokens", 0),
            },
        }
    return body


def _anthropic_stop(stop_reason: str) -> str:
    return {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}.get(stop_reason, "stop")


def _chat_url(provider: str, base_url: str) -> str:
    if provider == "anthropic":
        return f"{base_url}/v1/messages"
    return f"{base_url}/chat/completions"


async def call_upstream_nonstream(
    provider: str, base_url: str, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """One non-streaming upstream call, normalized to OpenAI shape."""
    headers = _headers_for(provider)
    body = _request_body(provider, kwargs)
    async with AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp: Response = await client.post(_chat_url(provider, base_url), json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"upstream error {resp.status_code}: {resp.text[:300]}")
        return _response_to_openai(provider, resp.json())


async def call_upstream_stream(
    provider: str, base_url: str, kwargs: Dict[str, Any]
) -> AsyncIterator[Dict[str, Any]]:
    """Streaming upstream call yielding parsed chunks (OpenAI SSE shape)."""
    headers = _headers_for(provider)
    body = _request_body(provider, kwargs)

    async def gen() -> AsyncIterator[Dict[str, Any]]:
        async with AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            async with client.stream(
                "POST", _chat_url(provider, base_url), json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"upstream error {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue

    return gen()
