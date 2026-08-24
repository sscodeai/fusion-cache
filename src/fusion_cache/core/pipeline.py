"""Core fusion pipeline: L1 exact → L2 semantic → L3 prefix accounting.

Flow for every request::

    request
      │
      ├─ L1 exact lookup ── hit? ──▶ replay cached response
      │
      ├─ L2 semantic lookup (embed → cosine vs stored rows) ── hit? ──▶ replay + extend TTL
      │
      └─ L3 miss → call upstream (chat.completions.create)
            └─ capture prompt_cache_hit_tokens/miss_tokens → $ saved
            └─ store in L1 (+ L2 row if semantic enabled)

The pipeline is async end-to-end; the embedder call is the only external
dependency on the miss path.

**Streaming:** for ``stream=True`` requests the pipeline buffers the upstream
chunks, stores them in L1, and returns a ``PipelineResult`` whose
``response`` is the chunk list plus the terminal ``usage`` snapshot.  The
wrapper converts that into a chunk iterator for the caller.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from ..config import FusionCacheConfig
from ..metrics.registry import MetricsRegistry
from ..prefix.accounting import UsageBreakdown, account_prefix_costs
from ..semantic.embedder import Embedder
from ..semantic.matcher import SemanticMatcher
from ..stores.base import Store
from ..stores.memory import MemoryStore
from .breaker import CircuitBreaker
from .key import request_hash
from .replay import astream

AsyncCompletionsFn = Callable[..., Awaitable[Any]]


@dataclass
class BufferedStream:
    """What the pipeline stores for a streamed (buffered) response."""

    chunks: list[Any]
    usage: Dict[str, Any]


@dataclass
class PipelineResult:
    """What the pipeline produced for one request."""

    response: Any = None
    layer: str = "miss"  # 'exact' | 'semantic' | 'miss' (L3 accounting is separate)
    hit: bool = False
    cached: bool = False
    hit_key: Optional[str] = None
    prefix_hit: bool = False  # upstream returned prompt_cache_hit_tokens > 0
    cost_saved_usd: float = 0.0
    upstream_hit_tokens: int = 0
    upstream_miss_tokens: int = 0
    upstream_total_tokens: int = 0
    latency_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stream(self) -> bool:
        return isinstance(self.response, BufferedStream)

    @property
    def stream_chunks(self) -> list[Any]:
        if isinstance(self.response, BufferedStream):
            return self.response.chunks
        return []

    @property
    def usage(self) -> Dict[str, Any]:
        if isinstance(self.response, BufferedStream):
            return self.response.usage
        return self.meta


@dataclass
class FusionCacheStats:
    """Aggregate stats exposed on the cache object (thread-safe read)."""

    requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    shared_hits: int = 0
    prefix_hits: int = 0
    misses: int = 0
    cost_saved_usd: float = 0.0

    def hit_rate(self) -> float:
        """Fraction of requests served without calling upstream (L1+L2+shared).

        L3 prefix hits are upstream-side discounts, reported separately via
        ``prefix_hits`` and ``cost_saved_usd``.
        """
        if self.requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits + self.shared_hits) / self.requests

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "shared_hits": self.shared_hits,
            "prefix_hits": self.prefix_hits,
            "misses": self.misses,
            "cost_saved_usd": self.cost_saved_usd,
            "hit_rate": self.hit_rate(),
        }


class FusionCache:
    """The three-layer fusion cache. Thread-safe and async-safe.

    Usage::

        cache = FusionCache(config=FusionCacheConfig())
        result = await cache.chat_completion(
            request={"model": "deepseek-chat", "messages": [...], "temperature": 0.2},
            upstream=async_fn,   # async_fn(**request) -> OpenAI ChatCompletion
            stream=False,
        )
    """

    def __init__(
        self,
        config: Optional[FusionCacheConfig] = None,
        store: Optional[Store] = None,
        semantic_store: Optional[Store] = None,
        embedder: Optional[Embedder] = None,
        metrics: Optional[MetricsRegistry] = None,
        max_concurrent_embeds: int = 8,
    ) -> None:
        self.config = config or FusionCacheConfig()
        self.store = store or MemoryStore(max_entries=self.config.max_entries)
        self.semantic_store = semantic_store or MemoryStore(max_entries=self.config.semantic_max_entries)
        self.embedder = embedder or Embedder(self.config.embedder)
        self.metrics = metrics or MetricsRegistry()
        self.matcher = SemanticMatcher(self.config)
        self.breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            cooldown_s=self.config.circuit_breaker_cooldown_s,
            enabled=self.config.circuit_breaker_enabled,
        )
        self._embed_sem = asyncio.Semaphore(max_concurrent_embeds)
        self._stats = FusionCacheStats()
        self._stats_lock = asyncio.Lock()
        self._inflight: Dict[str, "asyncio.Future[Any]"] = {}
        self._inflight_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> FusionCacheStats:
        return self._stats

    def stats_dict(self) -> Dict[str, Any]:
        return {**self._stats.as_dict(), "metrics": self.metrics.snapshot()}

    async def _record(self, result: PipelineResult) -> None:
        async with self._stats_lock:
            self._stats.requests += 1
            if result.layer == "exact":
                self._stats.exact_hits += 1
            elif result.layer == "semantic":
                self._stats.semantic_hits += 1
            elif result.layer == "shared":
                self._stats.shared_hits += 1
            else:
                self._stats.misses += 1
                if result.prefix_hit:
                    self._stats.prefix_hits += 1
            self._stats.cost_saved_usd += result.cost_saved_usd

    # ------------------------------------------------------------- L1 helpers
    def _exact_key(self, request: Mapping[str, Any]) -> str:
        return request_hash(request)

    async def _exact_get(self, key: str) -> Optional[Dict[str, Any]]:
        return await self.store.aget(key)

    # ------------------------------------------------------------- L2 helpers
    async def _embed(self, text: str) -> list[float]:
        async with self._embed_sem:
            return await self.embedder.embed(text)

    async def _semantic_search(self, query_embedding: list[float], request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        entries = []
        for entry_key in await self.semantic_store.akeys():
            entry = await self.semantic_store.aget(entry_key)
            if entry is not None and "embedding" in entry:
                entries.append(entry)
        return self.matcher.best_match(query_embedding, entries, request=request)

    # ------------------------------------------------------------ core path
    async def chat_completion(
        self,
        request: Mapping[str, Any],
        upstream: AsyncCompletionsFn,
        stream: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline for one chat completion request.

        ``upstream`` must be an async callable accepting ``**request`` and
        returning an OpenAI-style object.  If ``stream`` is True, upstream is
        expected to return an async iterator of chunks (the pipeline buffers
        and stores them; the wrapper replays them to the caller).
        """
        started = time.perf_counter()
        req = dict(request)
        req["stream"] = bool(stream)
        result = PipelineResult()

        key = self._exact_key(req)

        # ---- L1 exact -----------------------------------------------------
        if self.config.enable_exact:
            cached = await self._exact_get(key)
            if cached is not None:
                result.layer = "exact"
                result.hit = True
                result.cached = True
                result.hit_key = key
                result.response = cached["response"]
                result.meta = cached.get("meta") or {}
                result.latency_ms = (time.perf_counter() - started) * 1000.0
                self.metrics.record("exact", hit=True, latency_ms=result.latency_ms)
                await self._record(result)
                return result

        # ---- L2 semantic ----------------------------------------------------
        if self.config.enable_semantic and self.embedder.enabled:
            try:
                query_embedding = await self._embed(self._embed_text(req))
                entry = await self._semantic_search(query_embedding, req)
                if entry is not None:
                    result.layer = "semantic"
                    result.hit = True
                    result.cached = True
                    result.hit_key = entry.get("key")
                    result.response = entry["response"]
                    result.meta = entry.get("meta") or {}
                    result.meta["semantic_score"] = entry.get("_score", 0.0)
                    result.latency_ms = (time.perf_counter() - started) * 1000.0
                    self.metrics.record(
                        "semantic", hit=True, latency_ms=result.latency_ms, tokens=0
                    )
                    await self._record(result)
                    return result
            except Exception:
                self.metrics.record("semantic", hit=False, reason="embed_error")

        # ---- L3 miss → upstream ----------------------------------------------
        # Circuit breaker: if upstream is failing, serve cache hits normally
        # but fail fast on requests that would need a fresh upstream call.
        if self.config.circuit_breaker_enabled and not self.breaker.allow_request():
            result.layer = "miss"
            result.hit = False
            result.cached = False
            result.latency_ms = (time.perf_counter() - started) * 1000
            self.metrics.record("miss", hit=False, reason="circuit_open")
            await self._record(result)
            raise RuntimeError("upstream circuit breaker open; cache miss cannot be served")

        # Single-flight: concurrent requests for the SAME key share one
        # upstream call (prevents cache stampede on cold keys).  The future
        # carries the upstream response object; waiters continue processing
        # with that response (cache fill is idempotent).
        # Check + claim must be atomic under the lock, otherwise concurrent
        # requests all see "no in-flight" and stampede upstream.
        fut: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
        async with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is not None:
                pass  # handled below, outside the lock (await can't hold the lock)
            else:
                self._inflight[key] = fut
        if existing is not None:
            try:
                upstream_obj = await existing
            except Exception:
                # The fetching request failed and will clean up the map in its
                # finally block; we simply retry the whole flow by falling
                # through to claim a fresh future below.
                upstream_obj = None
            if upstream_obj is not None:
                fut_result = upstream_obj
                # Shared result from another in-flight request: this
                # request did NOT call upstream.  Mark it as a hit so
                # stats reflect reality (no upstream spend).
                return await self._process_shared_response(key, req, fut_result, result, stream, started)
            # Fetching request failed; claim our own future.
            fut = asyncio.get_running_loop().create_future()
            async with self._inflight_lock:
                # Only claim if no one else already did while we awaited.
                self._inflight.setdefault(key, fut)
        try:
            try:
                upstream_result = upstream(**req)
                if inspect.isawaitable(upstream_result):
                    upstream_obj = await upstream_result
                else:
                    upstream_obj = upstream_result
            except Exception as exc:
                self.metrics.record("miss", hit=False, reason="upstream_error")
                if self.config.circuit_breaker_enabled:
                    self.breaker.record_failure()
                fut.set_exception(exc)
                raise
            if self.config.circuit_breaker_enabled:
                self.breaker.record_success()
            fut.set_result(upstream_obj)
        finally:
            async with self._inflight_lock:
                self._inflight.pop(key, None)

        return await self._process_response(key, req, upstream_obj, result, stream, started)

    # ------------------------------------------------------------- helpers
    async def _process_response(
        self,
        key: str,
        req: Mapping[str, Any],
        upstream_obj: Any,
        result: PipelineResult,
        stream: bool,
        started: float,
    ) -> PipelineResult:
        """Process an upstream response: buffer/parse, account L3, fill caches, record metrics."""
        # "miss" from our perspective: we called upstream.  L3 prefix
        # accounting is expressed via cost_saved_usd and the "prefix" metric
        # layer (upstream prefix-cache hits), not as a cache hit.
        result.layer = "miss"
        result.prefix_hit = False

        if stream:
            # buffer the stream, snapshot usage from the terminal chunk
            chunks: list[Any] = []
            usage: Dict[str, Any] = {}
            async for chunk in astream(upstream_obj):
                chunks.append(chunk)
                u = _extract_usage(chunk)
                if u:
                    usage = u
            if not chunks:
                raise RuntimeError("upstream returned an empty stream")
            result.response = BufferedStream(chunks=chunks, usage=usage)
            result.meta = dict(usage)
        else:
            result.response = upstream_obj
            u = _extract_usage(upstream_obj)
            result.meta = dict(u) if u else {}

        breakdown = UsageBreakdown.from_dict(result.meta)
        result.upstream_hit_tokens = breakdown.prompt_cache_hit_tokens
        result.upstream_miss_tokens = breakdown.prompt_cache_miss_tokens
        result.upstream_total_tokens = breakdown.total_tokens
        result.cost_saved_usd = account_prefix_costs(breakdown, self.config.price_model).saved_usd
        if result.cost_saved_usd > 0 or result.upstream_hit_tokens > 0:
            result.prefix_hit = True
        result.cached = False
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        # ---- populate caches ------------------------------------------------
        if self.config.enable_exact:
            await self.store.aset(
                key,
                {"response": result.response, "meta": result.meta, "stream": bool(stream)},
                ttl=self.config.exact_ttl,
            )
        if self.config.enable_semantic and self.embedder.enabled and not stream:
            try:
                emb = await self._embed(self._embed_text(req))
                await self.semantic_store.aset(
                    f"sem:{key}",
                    {
                        "key": key,
                        "response": result.response,
                        "embedding": emb,
                        "meta": {**result.meta, "_request": dict(req)},
                        "stream": bool(stream),
                    },
                    ttl=self.config.semantic_ttl,
                )
            except Exception:
                self.metrics.record("semantic", hit=False, reason="embed_error")

        self.metrics.record(
            "miss",
            hit=False,
            latency_ms=result.latency_ms,
            cost_saved_usd=result.cost_saved_usd,
            tokens=result.upstream_hit_tokens,
        )
        if result.upstream_hit_tokens:
            self.metrics.record("prefix", hit=True, tokens=result.upstream_hit_tokens)
        await self._record(result)
        return result

    async def _process_shared_response(
        self,
        key: str,
        req: Mapping[str, Any],
        upstream_obj: Any,
        result: PipelineResult,
        stream: bool,
        started: float,
    ) -> PipelineResult:
        """Process a response shared from another in-flight request (single-flight).

        This request did NOT call upstream — another request did and we
        piggybacked on its result.  Marked as a *shared hit* (layer="shared",
        hit=True) so stats reflect that no upstream spend happened here.
        """
        result.layer = "shared"
        result.hit = True
        result.cached = True
        result.hit_key = key

        if stream:
            chunks: list[Any] = []
            usage: Dict[str, Any] = {}
            async for chunk in astream(upstream_obj):
                chunks.append(chunk)
                u = _extract_usage(chunk)
                if u:
                    usage = u
            if not chunks:
                raise RuntimeError("upstream returned an empty stream")
            result.response = BufferedStream(chunks=chunks, usage=usage)
            result.meta = dict(usage)
        else:
            result.response = upstream_obj
            u = _extract_usage(upstream_obj)
            result.meta = dict(u) if u else {}

        breakdown = UsageBreakdown.from_dict(result.meta)
        result.upstream_hit_tokens = breakdown.prompt_cache_hit_tokens
        result.upstream_miss_tokens = breakdown.prompt_cache_miss_tokens
        result.upstream_total_tokens = breakdown.total_tokens
        result.cost_saved_usd = account_prefix_costs(breakdown, self.config.price_model).saved_usd
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        # Fill caches idempotently (the first request may have already done so).
        if self.config.enable_exact:
            await self.store.aset(
                key,
                {"response": result.response, "meta": result.meta, "stream": bool(stream)},
                ttl=self.config.exact_ttl,
            )
        self.metrics.record(
            "shared", hit=True, latency_ms=result.latency_ms,
            cost_saved_usd=result.cost_saved_usd, tokens=result.upstream_hit_tokens,
        )
        await self._record(result)
        return result

    # ------------------------------------------------------------- helpers
    def _embed_text(self, request: Mapping[str, Any]) -> str:
        messages = request.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                    )
                parts.append(f"{role}: {content}")
            return "\n".join(parts)
        return str(request.get("prompt", ""))

    # ------------------------------------------------------------ lifecycle
    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.embedder.aclose()
        await self.store.aclose()
        await self.semantic_store.aclose()


def _extract_usage(obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of the usage dict from an OpenAI-style object."""
    if isinstance(obj, dict):
        return obj.get("usage") or {}
    usage = getattr(obj, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return usage.model_dump() if hasattr(usage, "model_dump") else {}
