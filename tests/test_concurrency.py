"""Concurrency / load tests: verify no races or memory blowups under load."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache


async def _ok_upstream(**kwargs: Any) -> Dict[str, Any]:
    await asyncio.sleep(0.001)  # simulate a tiny upstream latency
    return {
        "id": "ok",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def _hammer(cache: FusionCache, n: int, concurrency: int) -> Dict[str, int]:
    """Fire n requests with `concurrency` workers; return per-layer counts."""
    sem = asyncio.Semaphore(concurrency)
    counts = {"exact": 0, "semantic": 0, "miss": 0, "errors": 0}

    async def one(i: int) -> None:
        # 10 distinct prompts, each repeated n/10 times → heavy L1 reuse.
        prompt = f"question number {i % 10} about caching"
        request = {"model": "m", "messages": [{"role": "user", "content": prompt}]}
        async with sem:
            try:
                r = await cache.chat_completion(request=request, upstream=_ok_upstream, stream=False)
                counts[r.layer] = counts.get(r.layer, 0) + 1
            except Exception:
                counts["errors"] += 1

    await asyncio.gather(*(one(i) for i in range(n)))
    return counts


async def test_concurrent_200_requests_no_errors():
    """200 requests at concurrency 50: no errors, no races, stats consistent."""
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    counts = await _hammer(cache, n=200, concurrency=50)
    await cache.aclose()

    assert counts["errors"] == 0
    assert counts["miss"] == 10  # 10 distinct prompts → 10 cold misses
    assert counts["exact"] + counts.get("shared", 0) == 190  # rest are hits (exact or single-flight shared)
    assert cache.stats.requests == 200
    assert cache.stats.exact_hits + cache.stats.shared_hits == 190
    assert cache.stats.misses == 10
    # Hit rate: 95%
    assert abs(cache.stats.hit_rate() - 0.95) < 0.001


async def test_concurrent_same_key_single_upstream_call():
    """100 concurrent identical requests → exactly 1 upstream call (no stampede)."""
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    upstream_calls = {"n": 0}

    async def counting_upstream(**kwargs: Any) -> Dict[str, Any]:
        upstream_calls["n"] += 1
        await asyncio.sleep(0.002)
        return {
            "id": "ok",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    request = {"model": "m", "messages": [{"role": "user", "content": "identical"}]}
    results = await asyncio.gather(
        *(cache.chat_completion(request=request, upstream=counting_upstream, stream=False) for _ in range(100))
    )
    await cache.aclose()

    assert upstream_calls["n"] == 1  # exactly one upstream call, no stampede
    # First request is the miss that populated; rest are hits (shared or exact).
    assert results[0].layer == "miss"
    assert all(r.layer in ("shared", "exact") for r in results[1:])


async def test_concurrent_streaming_no_corruption():
    """Concurrent streaming requests: buffered chunks stay intact per request."""
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)

    async def stream_upstream(**kwargs: Any) -> Any:
        async def gen():
            for i in range(5):
                yield {"choices": [{"delta": {"content": f"c{i}"}}]}
            yield {"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}

        return gen()

    async def one(i: int):
        request = {"model": "m", "messages": [{"role": "user", "content": f"stream-{i}"}]}
        r = await cache.chat_completion(request=request, upstream=stream_upstream, stream=True)
        chunks = r.stream_chunks
        texts = [
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c.get("choices") and c["choices"][0].get("delta", {}).get("content")
        ]
        assert texts == [f"c{j}" for j in range(5)], f"corrupted stream for request {i}"

    await asyncio.gather(*(one(i) for i in range(30)))
    await cache.aclose()


async def test_concurrent_same_key_streaming_single_upstream_call():
    """Concurrent identical streams share buffered output, not the raw iterator."""
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    upstream_calls = {"n": 0}

    async def stream_upstream(**kwargs: Any) -> Any:
        upstream_calls["n"] += 1

        async def gen():
            await asyncio.sleep(0.001)
            for i in range(5):
                yield {"choices": [{"delta": {"content": f"c{i}"}}]}
            yield {"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}

        return gen()

    request = {"model": "m", "messages": [{"role": "user", "content": "same stream"}]}
    results = await asyncio.gather(
        *(cache.chat_completion(request=request, upstream=stream_upstream, stream=True) for _ in range(50))
    )
    await cache.aclose()

    assert upstream_calls["n"] == 1
    assert results[0].layer == "miss"
    assert all(r.layer in ("shared", "exact") for r in results[1:])
    for r in results:
        texts = [
            c["choices"][0]["delta"].get("content", "")
            for c in r.stream_chunks
            if c.get("choices") and c["choices"][0].get("delta", {}).get("content")
        ]
        assert texts == [f"c{j}" for j in range(5)]


async def test_metrics_thread_safe_under_load():
    """Metrics registry stays consistent under concurrent recording."""
    from fusion_cache.metrics.registry import MetricsRegistry

    reg = MetricsRegistry()
    await asyncio.gather(
        *(asyncio.to_thread(reg.record, "exact", hit=True, latency_ms=1.0) for _ in range(200))
    )
    snap = reg.snapshot()
    assert snap["layers"]["exact"]["hits"] == 200
    assert snap["total_requests"] == 200
    assert snap["hit_rate"] == 1.0
