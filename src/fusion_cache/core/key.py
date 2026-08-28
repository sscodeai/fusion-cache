"""L1 canonicalization: turn chat/completions-style requests into cache keys.

A canonical key normalizes every field that does *not* change the upstream
response, so that cosmetic differences (whitespace, ordering, defaults)
still hit the exact cache.  Anything that *does* change the upstream response
(model, temperature, top_p, presence/frequency penalty, stream) stays in the
key verbatim.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence, Tuple

# Fields of a chat.completions request that change the upstream response and
# therefore MUST be part of the cache key.
_RESPONSE_AFFECTING = (
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "modalities",
    "audio",
    "reasoning_effort",
    "verbosity",
    "metadata",
)

# Booleans that change the *transport* but not the logical content.  We keep
# them in the key so a stream and a non-stream request never collide, but we
# keep them stable so that only the literal value matters.
_BOOLEAN_TRANSPORT_FIELDS = ("stream",)

# Fields that are pure transport/default noise and are dropped entirely.
_TRANSPORT_ONLY = (
    "stream_options",
    "stream",
)


def normalize_text(value: str) -> str:
    """Collapse runs of whitespace and strip surrounding whitespace."""
    return re.sub(r"\s+", " ", value).strip()


def canonicalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a chat message list.

    - Message keys are sorted (``role``, ``content``, ``name``, ``tool_calls``...).
    - Text content is whitespace-collapsed.
    - Content arrays (multi-part content) are collapsed key-wise.
    - Tool calls / tool call ids are normalized.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {}
        for key in sorted(message.keys()):
            value = message[key]
            if key == "content" and isinstance(value, str):
                item[key] = normalize_text(value)
            elif key == "content" and isinstance(value, list):
                parts: list[Any] = []
                for part in value:
                    if isinstance(part, dict):
                        parts.append(
                            {k: (normalize_text(v) if isinstance(v, str) else v) for k, v in sorted(part.items())}
                        )
                    elif isinstance(part, str):
                        parts.append(normalize_text(part))
                    else:
                        parts.append(part)
                item[key] = parts
            elif isinstance(value, str):
                item[key] = normalize_text(value)
            else:
                item[key] = value
        out.append(item)
    return out


def _sortable(value: Any) -> Any:
    """Return a JSON-sortable / stable-hashable copy of an arbitrary value."""
    if isinstance(value, dict):
        return {str(k): _sortable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_sortable(v) for v in value]
    return value


def canonicalize_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical dict for a chat/completions request.

    ``request`` accepts either the raw ``openai`` kwarg dict (with
    ``messages``, ``model``, ...) or an already-``dict()``-ed request body.
    """
    req = dict(request)
    canonical: dict[str, Any] = {}

    for field in _RESPONSE_AFFECTING:
        if field in req and req[field] is not None:
            canonical[field] = _sortable(req[field])

    stream = bool(req.get("stream", False))
    canonical["stream"] = stream

    messages = req.get("messages")
    if messages is not None:
        canonical["messages"] = canonicalize_messages(messages)
    # prompt-style completions (non-chat) — keep verbatim
    if "prompt" in req:
        canonical["prompt"] = normalize_text(str(req["prompt"]))

    return canonical


def request_hash(request: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonicalized request — the L1 cache key."""
    canonical = canonicalize_request(request)
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_fingerprint(request: Mapping[str, Any]) -> Tuple[str, dict[str, Any]]:
    """Return (sha256, canonical) — convenient for debugging/benchmarks."""
    canonical = canonicalize_request(request)
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), canonical
