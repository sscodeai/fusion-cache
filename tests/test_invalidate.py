"""Cache invalidation tests: pipeline-level and gateway endpoint."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from httpx import ASGITransport, AsyncClient, Response

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.gateway.app import create_app

from conftest import _default_chat_response, make_cache_no_embed

FAKE = "https://inv-upstream.test"


async def _ok_upstream(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": "ok",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def test_invalidate_single_request():
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    req = {"model": "m", "messages": [{"role": "user", "content": "invalidate me"}]}

    r1 = await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)
    assert r1.layer == "miss"

    r2 = await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)
    assert r2.layer == "exact"  # cached

    removed = await cache.invalidate(req)
    assert removed is True

    r3 = await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)
    assert r3.layer == "miss"  # evicted → miss again


async def test_invalidate_all():
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    for i in range(5):
        req = {"model": "m", "messages": [{"role": "user", "content": f"task {i}"}]}
        await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)

    n = await cache.invalidate_all()
    assert n >= 5

    # All evicted → next call is a miss again.
    req = {"model": "m", "messages": [{"role": "user", "content": "task 0"}]}
    r = await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)
    assert r.layer == "miss"


async def test_invalidate_unknown_request_returns_false():
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    cache = FusionCache(config=cfg)
    removed = await cache.invalidate({"model": "m", "messages": [{"role": "user", "content": "never cached"}]})
    assert removed is False


@pytest.fixture
def inv_router():
    import respx

    r = respx.mock(base_url=FAKE, assert_all_called=False)
    r.post(f"{FAKE}/chat/completions").mock(return_value=Response(200, json=_default_chat_response()))
    r.start()
    yield r
    r.stop()


async def test_gateway_invalidate_endpoint(inv_router):
    cache = make_cache_no_embed()
    app = create_app(cache=cache, upstream_base_url=FAKE, gateway_api_key="sekrit")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        h = {"Authorization": "Bearer sekrit"}

        # Fill cache.
        await client.post("/v1/chat/completions", json=payload, headers=h)

        # Invalidate all.
        r = await client.post("/v1/cache/invalidate", json={"all": True}, headers=h)
        assert r.status_code == 200
        assert r.json()["all"] is True

        # Auth required.
        r = await client.post("/v1/cache/invalidate", json={"all": True})
        assert r.status_code == 401

        # Bad body.
        r = await client.post("/v1/cache/invalidate", json={"foo": 1}, headers=h)
        assert r.status_code == 400
