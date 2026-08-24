"""Circuit breaker + graceful degradation tests."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.breaker import CircuitBreaker
from fusion_cache.core.pipeline import FusionCache


def test_breaker_opens_after_threshold():
    b = CircuitBreaker(failure_threshold=3, cooldown_s=30, enabled=True)
    assert b.allow_request() is True
    b.record_failure()
    b.record_failure()
    assert b.allow_request() is True  # not yet at threshold
    b.record_failure()
    assert b.state == "open"
    assert b.allow_request() is False  # open → blocked


def test_breaker_half_open_after_cooldown():
    b = CircuitBreaker(failure_threshold=1, cooldown_s=0.05, enabled=True)
    b.record_failure()
    assert b.state == "open"
    assert b.allow_request() is False

    import time

    time.sleep(0.06)
    # half-open: one probe allowed
    assert b.state == "half-open"
    assert b.allow_request() is True


def test_breaker_resets_on_success():
    b = CircuitBreaker(failure_threshold=2, cooldown_s=30, enabled=True)
    b.record_failure()
    b.record_failure()
    assert b.state == "open"
    b.record_success()  # a success (e.g. from a probe) resets
    assert b.state == "closed"
    assert b.allow_request() is True


def test_breaker_disabled():
    b = CircuitBreaker(failure_threshold=1, cooldown_s=30, enabled=False)
    b.record_failure()
    b.record_failure()
    assert b.allow_request() is True  # never blocks when disabled


async def _failing_upstream(**kwargs: Any) -> Any:
    raise RuntimeError("upstream down")


async def _ok_upstream(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": "ok",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def test_pipeline_circuit_breaker_opens():
    """After N upstream failures the breaker opens and further misses fail fast."""
    cfg = FusionCacheConfig(
        enable_semantic=False,
        circuit_breaker_enabled=True,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_cooldown_s=60,
    )
    cache = FusionCache(config=cfg)
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    # First 3 failures trip the breaker.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cache.chat_completion(request=request, upstream=_failing_upstream, stream=False)
    assert cache.breaker.state == "open"

    # Now a cache-miss fails fast with the circuit-open message.
    with pytest.raises(RuntimeError, match="circuit breaker"):
        await cache.chat_completion(request=request, upstream=_failing_upstream, stream=False)


async def test_pipeline_cache_hits_served_during_circuit_open():
    """When the breaker is open, cached hits are STILL served — graceful degradation."""
    cfg = FusionCacheConfig(
        enable_semantic=False,
        circuit_breaker_enabled=True,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_cooldown_s=60,
    )
    cache = FusionCache(config=cfg)
    request = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}

    # Warm the cache with a successful call.
    first = await cache.chat_completion(request=request, upstream=_ok_upstream, stream=False)
    assert first.hit is False  # miss, populated cache

    # Trip the breaker with failures on a DIFFERENT request (so cache stays warm).
    other = {"model": "m", "messages": [{"role": "user", "content": "other"}]}
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cache.chat_completion(request=other, upstream=_failing_upstream, stream=False)
    assert cache.breaker.state == "open"

    # The cached request is still served as an exact hit — graceful degradation.
    hit = await cache.chat_completion(request=request, upstream=_failing_upstream, stream=False)
    assert hit.hit is True
    assert hit.layer == "exact"


async def test_pipeline_breaker_recovers():
    """After cooldown, a successful call closes the breaker."""
    cfg = FusionCacheConfig(
        enable_semantic=False,
        circuit_breaker_enabled=True,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_cooldown_s=0.05,
    )
    cache = FusionCache(config=cfg)
    request = {"model": "m", "messages": [{"role": "user", "content": "recover"}]}

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cache.chat_completion(request=request, upstream=_failing_upstream, stream=False)
    assert cache.breaker.state == "open"

    import asyncio

    await asyncio.sleep(0.06)  # cooldown passes → half-open probe allowed
    # A successful call closes it.
    ok = await cache.chat_completion(request=request, upstream=_ok_upstream, stream=False)
    assert ok.hit is False
    assert cache.breaker.state == "closed"
