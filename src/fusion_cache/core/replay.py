"""Cached-response replay, including buffered streaming.

MVP decision (per open-source-decision.md §6.6): **buffer-then-replay**.
We wait for the full upstream stream, then replay it chunk-by-chunk to the
caller.  This trades a higher TTFB for full savings and a byte-exact replay.

Helpers here are used both by the wrapper (which replays real chunk objects)
and by the pipeline when serving cached entries.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .key import normalize_text

# Chunk keys that identify per-stream-tick payloads vs meta/terminal chunks.
_META_KEYS = {"id", "object", "created", "model", "system_fingerprint", "usage", "service_tier"}


def astream(obj: Any) -> AsyncIterator[Any]:
    """Yield items from either an async or a sync iterator, asynchronously."""
    if hasattr(obj, "__aiter__"):
        return _aiter(obj)
    return _sync_to_async(obj)


async def _aiter(obj: Any) -> AsyncIterator[Any]:
    async for item in obj:
        yield item


async def _sync_to_async(obj: Any) -> AsyncIterator[Any]:
    for item in obj:
        yield item


async def buffer_stream(chunks: AsyncIterator[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Drain a stream of chunks (async or sync iterator).

    Returns ``(chunks, meta)`` where ``meta`` is the union of all top-level
    metadata fields seen across chunks (id, created, model, usage, ...).
    """
    buffered: list[Any] = []
    meta: dict[str, Any] = {}
    async for chunk in astream(chunks):
        buffered.append(chunk)
        if hasattr(chunk, "model_dump"):
            data = chunk.model_dump(exclude_none=False)
        elif isinstance(chunk, dict):
            data = chunk
        else:
            data = {}
        for key in _META_KEYS:
            if key in data and key not in meta and data[key] is not None:
                meta[key] = data[key]
    if not buffered:
        raise RuntimeError("upstream returned an empty stream")
    return buffered, meta


async def replay_chunks(chunks: List[Any]) -> AsyncIterator[Any]:
    """Yield buffered chunks one-by-one (async generator)."""
    for chunk in chunks:
        yield chunk


def replay_chunks_sync(chunks: List[Any]) -> Iterator[Any]:
    """Yield buffered chunks one-by-one (sync generator)."""
    for chunk in chunks:
        yield chunk


async def replay_stream_ttfb_ms(chunks: List[Any]) -> float:
    """Time to first byte when replaying a buffered stream (for metrics)."""
    started = time.perf_counter()
    async for _ in replay_chunks(chunks):
        return (time.perf_counter() - started) * 1000.0
    return (time.perf_counter() - started) * 1000.0


def stream_text(chunks: List[Any]) -> str:
    """Concatenate the text deltas of a buffered chunk list."""
    parts: list[str] = []
    for chunk in chunks:
        if hasattr(chunk, "choices"):
            choices = chunk.choices
        elif isinstance(chunk, dict):
            choices = chunk.get("choices") or []
        else:
            choices = []
        for choice in choices:
            if hasattr(choice, "delta"):
                delta = choice.delta
            elif isinstance(choice, dict):
                delta = choice.get("delta") or {}
            else:
                delta = {}
            if hasattr(delta, "content"):
                content = delta.content
            elif isinstance(delta, dict):
                content = delta.get("content")
            else:
                content = None
            if content:
                parts.append(content)
    return "".join(parts)


def stream_text_normalized(chunks: List[Any]) -> str:
    """Whitespace-normalized stream text (for L2 content checks)."""
    return normalize_text(stream_text(chunks))
