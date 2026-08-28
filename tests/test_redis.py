"""Redis shared-cache tests: two cache instances sharing one Redis see each other's entries.

Requires a running Redis (default localhost:6380). Skipped when unreachable.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import BufferedStream, FusionCache
from fusion_cache.stores.redis import RedisStore, _decode_value, _encode_value

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/0")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_REDIS", "1") == "1",
    reason="Redis tests disabled",
)


def test_buffered_stream_serialization_roundtrip():
    value = {
        "response": BufferedStream(
            chunks=[
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}},
            ],
            usage={"total_tokens": 3},
        ),
        "meta": {"total_tokens": 3},
        "stream": True,
    }

    decoded = _decode_value(_encode_value(value))

    assert isinstance(decoded["response"], BufferedStream)
    assert decoded["response"].chunks == value["response"].chunks
    assert decoded["response"].usage == {"total_tokens": 3}


def _redis_available() -> bool:
    try:
        import redis.asyncio as aioredis

        async def _ping() -> bool:
            client = aioredis.from_url(REDIS_URL, socket_connect_timeout=2)
            try:
                return bool(await client.ping())
            except Exception:
                return False
            finally:
                await client.aclose()

        import asyncio

        return asyncio.run(_ping())
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
async def test_two_instances_share_exact_cache():
    """Instance A fills the cache; instance B (same Redis) hits it."""
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    store_a = RedisStore(url=REDIS_URL, prefix="fc-test:")
    store_b = RedisStore(url=REDIS_URL, prefix="fc-test:")
    await store_a.aclear()  # clean slate between runs
    cache_a = FusionCache(config=cfg, store=store_a)
    cache_b = FusionCache(config=cfg, store=store_b)

    request = {"model": "m", "messages": [{"role": "user", "content": "shared across instances"}]}

    # A misses and fills Redis.
    r_a = await cache_a.chat_completion(request=request, upstream=_ok_upstream, stream=False)
    assert r_a.layer == "miss"

    # B hits the same key via shared Redis.
    r_b = await cache_b.chat_completion(request=request, upstream=_ok_upstream, stream=False)
    assert r_b.layer == "exact"
    assert r_b.hit is True

    await cache_a.aclose()
    await cache_b.aclose()


@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
async def test_redis_store_ttl_and_clear():
    """RedisStore honors TTL and clears."""
    store = RedisStore(url=REDIS_URL, prefix="fc-ttl-test:")
    await store.aclear()

    await store.aset("k1", {"response": {"choices": []}}, ttl=1)
    assert await store.aget("k1") is not None

    import asyncio

    await asyncio.sleep(1.5)
    assert await store.aget("k1") is None  # expired

    await store.aclear()
    assert await store.asize() == 0


async def _ok_upstream(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": "ok",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
