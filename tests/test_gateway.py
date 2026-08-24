"""Gateway tests: FastAPI app endpoints against a fake upstream (respx)."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
import respx
from httpx import AsyncClient, Response

from fusion_cache.config import FusionCacheConfig, EmbedderConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.gateway.app import create_app
from fusion_cache.semantic.embedder import Embedder

from conftest import _default_chat_response, make_embedder

FAKE_UPSTREAM = "https://fake-upstream.test"


@pytest.fixture
def router():
    r = respx.mock(base_url=FAKE_UPSTREAM, assert_all_called=False)

    def chat_handler(request):
        import json

        body = json.loads(request.content.decode())
        if body.get("stream"):
            # SSE-style streamed response
            sse = (
                'data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1700000000,'
                '"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                'data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1700000000,'
                '"model":"deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,'
                '"prompt_cache_hit_tokens":8,"prompt_cache_miss_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            )
            return Response(
                200,
                content=sse,
                headers={"Content-Type": "text/event-stream"},
            )
        return Response(200, json=_default_chat_response())

    r.post(f"{FAKE_UPSTREAM}/chat/completions").mock(side_effect=chat_handler)
    r.get(f"{FAKE_UPSTREAM}/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "deepseek-chat"}]})
    )
    r.start()
    yield r
    r.stop()


@pytest.fixture
def cache():
    return FusionCache(
        config=FusionCacheConfig(enable_semantic=False),  # keep L2 out of gateway tests
        embedder=make_embedder(),
    )


@pytest.fixture
def client(cache, router):
    from httpx import ASGITransport

    app = create_app(cache=cache, upstream_base_url=FAKE_UPSTREAM)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_models_passthrough(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "deepseek-chat"


async def test_chat_completions_miss_then_hit(client):
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Explain caching"}],
        "temperature": 0.2,
    }
    # First call → upstream (miss)
    r1 = await client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["choices"][0]["message"]["content"] == "Hello from fake upstream"
    assert body1["_fusion_cache"]["layer"] == "miss"
    assert body1["_fusion_cache"]["hit"] is False

    # Identical call → L1 exact hit
    r2 = await client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["_fusion_cache"]["layer"] == "exact"
    assert body2["_fusion_cache"]["hit"] is True


async def test_chat_completions_bad_json(client):
    r = await client.post("/v1/chat/completions", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "error" in r.json()


async def test_chat_completions_streaming(client):
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "stream please"}],
        "stream": True,
    }
    r = await client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "data: {" in text
    assert "data: [DONE]" in text


async def test_metrics_prometheus(client):
    # Prime the cache so there is data.
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "metrics"}],
    }
    await client.post("/v1/chat/completions", json=payload)
    await client.post("/v1/chat/completions", json=payload)

    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "fusion_cache_requests_total" in text
    assert "fusion_cache_exact_hits_total" in text
    assert "fusion_cache_hit_rate" in text


async def test_metrics_json(client):
    await client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "json"}]},
    )
    r = await client.get("/metrics", headers={"Accept": "application/json"})
    assert r.status_code == 200
    data = r.json()
    assert "requests" in data
    assert "hit_rate" in data


async def test_dashboard_html(client):
    r = await client.get("/dashboard")
    assert r.status_code == 200
    assert "fusion-cache dashboard" in r.text
    assert "<html" in r.text.lower()
    assert "hit rate" in r.text.lower()


# ---------------------------------------------------------------- auth


def _make_auth_app(cache, router):
    from httpx import ASGITransport

    app = create_app(cache=cache, upstream_base_url=FAKE_UPSTREAM, gateway_api_key="sekrit")
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_auth_required(cache, router):
    client = _make_auth_app(cache, router)
    # No key → 401
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401

    # Wrong key → 401
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401

    # Correct key → 200
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 200


async def test_auth_models(cache, router):
    client = _make_auth_app(cache, router)
    r = await client.get("/v1/models")
    assert r.status_code == 401
    r = await client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200


async def test_health_open_without_auth(cache, router):
    """health/metrics/dashboard stay open even when auth is configured."""
    client = _make_auth_app(cache, router)
    r = await client.get("/health")
    assert r.status_code == 200


# ---------------------------------------------------------------- CORS


async def test_cors_headers(cache, router):
    from httpx import ASGITransport

    app = create_app(
        cache=cache, upstream_base_url=FAKE_UPSTREAM, cors_origins=["https://app.example.com"]
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health", headers={"Origin": "https://app.example.com"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


# ---------------------------------------------------------------- Anthropic


async def test_anthropic_provider(cache):
    """Anthropic adapter normalizes /v1/messages responses into OpenAI shape."""
    from httpx import ASGITransport

    anth_url = "https://api.anthropic.test"
    r = respx.mock(base_url=anth_url, assert_all_called=False)

    def handler(request):
        # Verify the anthropic request shape: x-api-key header + /v1/messages path.
        assert request.headers.get("x-api-key") == "sk-anth"
        body = json.loads(request.content.decode())
        assert "messages" in body
        return Response(
            200,
            json={
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-latest",
                "content": [{"type": "text", "text": "Hello from Claude"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 5},
            },
        )

    r.post(f"{anth_url}/v1/messages").mock(side_effect=handler)
    r.start()

    import os

    os.environ["FUSION_UPSTREAM_API_KEY"] = "sk-anth"
    try:
        app = create_app(cache=cache, upstream_base_url=anth_url, provider="anthropic")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "claude-3-5-sonnet-latest", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["choices"][0]["message"]["content"] == "Hello from Claude"
            assert body["usage"]["prompt_tokens"] == 12
    finally:
        os.environ.pop("FUSION_UPSTREAM_API_KEY", None)
        r.stop()


# ---------------------------------------------------------------- 429 retry


async def test_429_retry_succeeds(cache):
    """Upstream 429 then success → the gateway retries and succeeds."""
    from httpx import ASGITransport

    retry_url = "https://flaky.test"
    calls = {"n": 0}
    r = respx.mock(base_url=retry_url, assert_all_called=False)

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return Response(429, json={"error": {"message": "rate limited"}})
        return Response(200, json=_default_chat_response())

    r.post(f"{retry_url}/chat/completions").mock(side_effect=handler)
    r.start()

    import os

    old = os.environ.get("FUSION_UPSTREAM_RETRIES")
    os.environ["FUSION_UPSTREAM_RETRIES"] = "2"
    try:
        app = create_app(cache=cache, upstream_base_url=retry_url)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200
            assert calls["n"] == 2
    finally:
        if old is None:
            os.environ.pop("FUSION_UPSTREAM_RETRIES", None)
        else:
            os.environ["FUSION_UPSTREAM_RETRIES"] = old
        r.stop()
