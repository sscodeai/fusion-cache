"""MCP server tests: the four tools work against a real FusionCache."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.mcp_server import build_server


async def _ok_upstream(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": "ok",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
async def cache():
    cfg = FusionCacheConfig(enable_semantic=False, circuit_breaker_enabled=False)
    c = FusionCache(config=cfg)
    req = {"model": "m", "messages": [{"role": "user", "content": "mcp test"}]}
    await c.chat_completion(request=req, upstream=_ok_upstream, stream=False)  # populate
    await c.chat_completion(request=req, upstream=_ok_upstream, stream=False)  # hit
    yield c
    await c.aclose()


async def test_cache_stats_tool(cache):
    server = build_server(cache=cache)
    result = await server.call_tool("cache_stats", {})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "hit_rate" in text
    assert "requests" in text


async def test_cache_status_tool(cache):
    server = build_server(cache=cache)
    result = await server.call_tool("cache_status", {})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "circuit_breaker" in text
    assert "store_type" in text


async def test_cache_config_tool(cache):
    server = build_server(cache=cache)
    result = await server.call_tool("cache_config", {})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "similarity_threshold" in text
    assert "price_model" in text


async def test_cache_invalidate_all(cache):
    server = build_server(cache=cache)
    result = await server.call_tool("cache_invalidate", {"all": True})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "evicted" in text

    # After invalidate-all, a previously cached request is a miss again.
    req = {"model": "m", "messages": [{"role": "user", "content": "mcp test"}]}
    r = await cache.chat_completion(request=req, upstream=_ok_upstream, stream=False)
    assert r.layer == "miss"


async def test_cache_invalidate_single(cache):
    server = build_server(cache=cache)
    req = {"model": "m", "messages": [{"role": "user", "content": "mcp test"}]}
    result = await server.call_tool("cache_invalidate", {"request": req})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "evicted" in text


async def test_cache_invalidate_no_args_returns_error(cache):
    server = build_server(cache=cache)
    result = await server.call_tool("cache_invalidate", {})
    text = result[0].text if isinstance(result, list) else str(result)
    assert "error" in text
