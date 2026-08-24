"""Drop-in OpenAI / AsyncOpenAI-compatible wrapper.

``CachedOpenAI`` wraps either the sync or the async OpenAI client and
transparently routes ``chat.completions.create`` through the fusion pipeline.
Every other method falls through to the wrapped client, so this is a true
drop-in::

    import openai
    from fusion_cache import FusionCache
    from fusion_cache.wrapper.openai import CachedOpenAI

    cache = FusionCache()
    client = CachedOpenAI(openai.OpenAI(api_key=...), cache=cache)
    resp = client.chat.completions.create(model="deepseek-chat", messages=[...])

Both sync and async clients are supported.

Streaming: when ``stream=True`` the wrapper buffers the upstream chunks
inside the pipeline, stores them, and then replays them to the caller as an
iterator (sync client) or async iterator (async client).  The terminal
``usage`` chunk is preserved on the last replayed chunk.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional

from ..core.pipeline import BufferedStream, FusionCache, PipelineResult
from ..core.replay import replay_chunks, replay_chunks_sync

__all__ = ["CachedOpenAI"]


def _stream_flag_from_kwargs(kwargs: Mapping[str, Any]) -> bool:
    return bool(kwargs.get("stream", False))


class CachedOpenAI:
    """Wrap an OpenAI client (sync or async) with the fusion cache."""

    def __init__(self, client: Any, cache: Optional[FusionCache] = None) -> None:
        self._client = client
        self.cache = cache or FusionCache()
        self._is_async = _looks_async(client)
        # Mirror OpenAI's layout so attribute access on the wrapper behaves
        # like the real client: wrapper.chat.completions.create(...)
        self.chat = _ChatNamespace(self)

    # ------------------------------------------------------------------ core
    def _nonstream_upstream(self, **kwargs: Any) -> Any:
        completions = _completions(self._client)
        return completions.create(**dict(kwargs))

    def _stream_upstream(self, **kwargs: Any) -> Any:
        completions = _completions(self._client)
        return completions.create(**dict(kwargs))

    def _chat_create(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("CachedOpenAI.chat.completions.create accepts keyword arguments only")
        stream = _stream_flag_from_kwargs(kwargs)
        upstream = self._stream_upstream if stream else self._nonstream_upstream

        if self._is_async:
            return self._async_chat_create(upstream, kwargs, stream)
        return self._sync_chat_create(upstream, kwargs, stream)

    # ------------------------------------------------------------ async path
    async def _async_chat_create(self, upstream: Any, kwargs: Mapping[str, Any], stream: bool) -> Any:
        request = dict(kwargs)
        result = await self.cache.chat_completion(request=request, upstream=upstream, stream=stream)
        return _deliver_result(result, stream, async_client=True)

    # ------------------------------------------------------------ sync path
    def _sync_chat_create(self, upstream: Any, kwargs: Mapping[str, Any], stream: bool) -> Any:
        request = dict(kwargs)
        result = _run_coro(self.cache.chat_completion(request=request, upstream=upstream, stream=stream))
        return _deliver_result(result, stream, async_client=False)

    # -------------------------------------------------------------- lifecycle
    @property
    def stats(self) -> Dict[str, Any]:
        return self.cache.stats_dict()

    def close(self) -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        if loop.is_running():
            return
        loop.run_until_complete(self.cache.aclose())

    def __getattr__(self, name: str) -> Any:
        # fall through to the wrapped client for anything else
        return getattr(self._client, name)


def _completions(client: Any) -> Any:
    chat = getattr(client, "chat", None)
    if chat is None:
        raise AttributeError("wrapped client has no .chat namespace")
    return chat.completions


def _looks_async(client: Any) -> bool:
    return "async" in type(client).__name__.lower()


def _run_coro(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise RuntimeError(
            "Cannot call a sync CachedOpenAI method from inside a running event loop; "
            "use the async client variant instead."
        )
    return asyncio.run(coro)


def _deliver_result(result: Any, stream: bool, async_client: bool) -> Any:
    """Convert a PipelineResult into the caller-facing shape."""
    if isinstance(result, PipelineResult):
        if stream:
            return _stream_iter(result, async_client)
        return result.response
    return result


def _stream_iter(result: PipelineResult, async_client: bool) -> Any:
    """Return a chunk iterator from a buffered pipeline result.

    The wrapper is sync or async; the caller of ``create(stream=True)`` gets
    an iterator that matches the wrapped client's style.  We make a best
    effort to preserve ``usage`` on the final replayed chunk.
    """
    buffered = result.response
    if not isinstance(buffered, BufferedStream):
        # passthrough already-iterable responses
        return buffered
    chunks = buffered.chunks
    usage = buffered.usage or result.meta

    if async_client:
        return _replay_async(chunks, usage)
    return replay_chunks_sync(_attach_usage_last(chunks, usage))


async def _replay_async(chunks: list[Any], usage: Dict[str, Any]) -> AsyncIterator[Any]:
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        if i == n - 1 and usage:
            chunk = _attach_usage(chunk, usage)
        yield chunk


def _attach_usage_last(chunks: list[Any], usage: Dict[str, Any]) -> list[Any]:
    """Return a copy of ``chunks`` with ``usage`` attached to the final chunk."""
    if not chunks or not usage:
        return chunks
    out = list(chunks)
    out[-1] = _attach_usage(out[-1], usage)
    return out


def _attach_usage(chunk: Any, usage: Dict[str, Any]) -> Any:
    """Best-effort attach of the usage dict to the terminal chunk."""
    try:
        if isinstance(chunk, dict):
            chunk = {**chunk, "usage": usage}
        else:
            try:
                chunk.usage = usage
            except Exception:
                pass
    except Exception:
        pass
    return chunk


class _ChatNamespace:
    """Mirror of ``client.chat.completions`` for attribute-style access."""

    def __init__(self, wrapper: CachedOpenAI) -> None:
        self._wrapper = wrapper
        self.completions = _CompletionsNamespace(wrapper)


class _CompletionsNamespace:
    def __init__(self, wrapper: CachedOpenAI) -> None:
        self._wrapper = wrapper

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper._chat_create(*args, **kwargs)
