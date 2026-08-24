"""Rate limiter + gateway rate-limit tests."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient, Response

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.gateway.app import create_app
from fusion_cache.gateway.ratelimit import SlidingWindowRateLimiter

from conftest import _default_chat_response

FAKE = "https://rl-upstream.test"


def test_sliding_window_allows_until_limit():
    rl = SlidingWindowRateLimiter(limit_per_minute=3)
    assert rl.allow("k1") is True
    assert rl.allow("k1") is True
    assert rl.allow("k1") is True
    assert rl.allow("k1") is False  # 4th exceeds


def test_sliding_window_per_key():
    rl = SlidingWindowRateLimiter(limit_per_minute=2)
    assert rl.allow("a") is True
    assert rl.allow("a") is True
    assert rl.allow("a") is False
    assert rl.allow("b") is True  # different key unaffected


def test_sliding_window_rolls_over():
    rl = SlidingWindowRateLimiter(limit_per_minute=1, window_s=0.05)
    assert rl.allow("k") is True
    assert rl.allow("k") is False

    import time

    time.sleep(0.06)
    assert rl.allow("k") is True  # window rolled over


def test_disabled_limiter_allows_all():
    rl = SlidingWindowRateLimiter(limit_per_minute=0)
    for _ in range(100):
        assert rl.allow("k") is True


def _make_app_with_ratelimit(cache, limit: int):
    old = os.environ.get("FUSION_RATE_LIMIT")
    os.environ["FUSION_RATE_LIMIT"] = str(limit)
    try:
        app = create_app(cache=cache, upstream_base_url=FAKE)
    finally:
        if old is None:
            os.environ.pop("FUSION_RATE_LIMIT", None)
        else:
            os.environ["FUSION_RATE_LIMIT"] = old
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def cache():
    from conftest import make_cache_no_embed

    return make_cache_no_embed()


@pytest.fixture
def rl_router():
    import respx

    r = respx.mock(base_url=FAKE, assert_all_called=False)
    r.post(f"{FAKE}/chat/completions").mock(return_value=Response(200, json=_default_chat_response()))
    r.start()
    yield r
    r.stop()


async def test_gateway_ratelimit_429(cache, rl_router):
    """After N requests, the gateway returns 429."""
    client = _make_app_with_ratelimit(cache, limit=2)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    r1 = await client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    r2 = await client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    r3 = await client.post("/v1/chat/completions", json=payload)
    assert r3.status_code == 429
    assert "rate limit" in r3.json()["error"]["message"]


async def test_gateway_ratelimit_by_api_key(cache, rl_router):
    """Different API keys get independent rate-limit buckets."""
    client = _make_app_with_ratelimit(cache, limit=2)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    h1 = {"Authorization": "Bearer key-aaa"}
    h2 = {"Authorization": "Bearer key-bbb"}

    assert (await client.post("/v1/chat/completions", json=payload, headers=h1)).status_code == 200
    assert (await client.post("/v1/chat/completions", json=payload, headers=h1)).status_code == 200
    assert (await client.post("/v1/chat/completions", json=payload, headers=h1)).status_code == 429
    # key-bbb is a fresh bucket
    assert (await client.post("/v1/chat/completions", json=payload, headers=h2)).status_code == 200
