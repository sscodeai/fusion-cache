"""FastAPI gateway for fusion-cache: an OpenAI-compatible reverse proxy.

The gateway exposes the same surface as the OpenAI API (``/v1/chat/completions``,
``/v1/models``) but routes every request through the three-layer fusion cache
before (optionally) hitting the real upstream.  It also exposes:

- ``GET /metrics``        — Prometheus-style text metrics (or JSON when
                            ``Accept: application/json``)
- ``GET /dashboard``      — self-contained HTML dashboard (inline CSS/JS, no CDN)
- ``GET /health``         — liveness probe

Upstream is configured with ``FUSION_UPSTREAM_BASE_URL`` (default
``https://api.deepseek.com``).  The cache uses a ``RedisStore`` when
``REDIS_URL`` is set, otherwise an in-memory ``MemoryStore``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.stores.redis import RedisStore

DEFAULT_UPSTREAM = os.environ.get("FUSION_UPSTREAM_BASE_URL", "https://api.deepseek.com")


def _build_cache() -> FusionCache:
    """Construct the shared cache: RedisStore when REDIS_URL is set, else memory."""
    cfg = FusionCacheConfig.from_env()
    store = None
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        store = RedisStore(url=redis_url)
    return FusionCache(config=cfg, store=store)


def create_app(cache: Optional[FusionCache] = None, upstream_base_url: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application.

    ``cache`` and ``upstream_base_url`` are injectable for tests; defaults
    build from the environment.
    """
    app = FastAPI(title="fusion-cache gateway", version="0.1.0")
    app.state.cache = cache or _build_cache()
    app.state.upstream_base_url = (upstream_base_url or DEFAULT_UPSTREAM).rstrip("/")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """Passthrough to the upstream models endpoint (best effort)."""
        upstream = app.state.upstream_base_url
        from httpx import AsyncClient

        async with AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(f"{upstream}/models")
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        # Fallback: advertise a reasonable default set without hitting the wire.
        return {"object": "list", "data": [{"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        cache: FusionCache = app.state.cache
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})

        stream = bool(body.get("stream", False))

        async def upstream(**kwargs: Any) -> Any:
            """Call the real upstream via httpx and return a JSON dict or an async chunk iterator."""
            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get("FUSION_UPSTREAM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            if stream:
                # The httpx client must outlive the generator: the pipeline
                # consumes the stream AFTER this function returns, so the
                # client is created inside the generator and closed when the
                # generator finishes.
                async def gen() -> AsyncIterator[Dict[str, Any]]:
                    from httpx import AsyncClient

                    async with AsyncClient(timeout=120) as client:
                        async with client.stream(
                            "POST",
                            f"{app.state.upstream_base_url}/chat/completions",
                            json=kwargs,
                            headers=headers,
                        ) as resp:
                            if resp.status_code != 200:
                                raise RuntimeError(f"upstream error {resp.status_code}")
                            async for line in resp.aiter_lines():
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                payload = line[len("data:"):].strip()
                                if payload == "[DONE]":
                                    return
                                try:
                                    yield json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                return gen()
            from httpx import AsyncClient

            async with AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{app.state.upstream_base_url}/chat/completions",
                    json=kwargs,
                    headers=headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"upstream error {resp.status_code}: {resp.text[:300]}")
                return resp.json()

        try:
            result = await cache.chat_completion(request=body, upstream=upstream, stream=stream)
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": {"message": f"upstream failure: {exc}"}})

        if stream:
            async def sse() -> AsyncIterator[str]:
                chunks = result.stream_chunks
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming: return the upstream-shaped JSON, augmented with cache metadata.
        payload = result.response if isinstance(result.response, dict) else _response_to_dict(result.response)
        payload = dict(payload)
        payload["_fusion_cache"] = {
            "layer": result.layer,
            "hit": result.hit,
            "cached": result.cached,
            "prefix_hit": result.prefix_hit,
            "cost_saved_usd": round(result.cost_saved_usd, 8),
            "latency_ms": round(result.latency_ms, 2),
        }
        return JSONResponse(content=payload)

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        cache: FusionCache = app.state.cache
        stats = cache.stats_dict()
        accept = request.headers.get("accept", "")
        if "json" in accept:
            return JSONResponse(content=stats)

        m = [
            "# HELP fusion_cache_requests_total Total requests seen by the cache",
            "# TYPE fusion_cache_requests_total counter",
            f"fusion_cache_requests_total {stats['requests']}",
            "# HELP fusion_cache_exact_hits_total Exact (L1) cache hits",
            "# TYPE fusion_cache_exact_hits_total counter",
            f"fusion_cache_exact_hits_total {stats['exact_hits']}",
            "# HELP fusion_cache_semantic_hits_total Semantic (L2) cache hits",
            "# TYPE fusion_cache_semantic_hits_total counter",
            f"fusion_cache_semantic_hits_total {stats['semantic_hits']}",
            "# HELP fusion_cache_prefix_hits_total Upstream prefix-cache (L3) hits",
            "# TYPE fusion_cache_prefix_hits_total counter",
            f"fusion_cache_prefix_hits_total {stats['prefix_hits']}",
            "# HELP fusion_cache_misses_total Upstream calls made (L3 miss)",
            "# TYPE fusion_cache_misses_total counter",
            f"fusion_cache_misses_total {stats['misses']}",
            "# HELP fusion_cache_hit_rate Overall hit rate (L1+L2)/requests",
            "# TYPE fusion_cache_hit_rate gauge",
            f"fusion_cache_hit_rate {stats['hit_rate']}",
            "# HELP fusion_cache_cost_saved_usd_total USD saved via prefix cache",
            "# TYPE fusion_cache_cost_saved_usd_total counter",
            f"fusion_cache_cost_saved_usd_total {stats['cost_saved_usd']}",
        ]
        metrics_data = stats.get("metrics", {})
        for layer, d in metrics_data.items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        m.append(f"fusion_cache_layer_{layer}_{k} {v}")
        return Response(content="\n".join(m) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        cache: FusionCache = app.state.cache
        stats = cache.stats_dict()
        return _render_dashboard(stats)

    return app


def _response_to_dict(resp: Any) -> Dict[str, Any]:
    """Best-effort conversion of an OpenAI-style response object to a dict."""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if isinstance(resp, dict):
        return resp
    # Fall back to a minimal envelope.
    return {"object": "chat.completion", "choices": [], "usage": {}}


def _render_dashboard(stats: Dict[str, Any]) -> str:
    """Self-contained HTML dashboard (inline CSS, no external resources)."""
    metrics = stats.get("metrics", {})
    layers_html = ""
    for layer, d in metrics.items():
        if not isinstance(d, dict):
            continue
        parts = " · ".join(f"{k}={v}" for k, v in d.items() if isinstance(v, (int, float)))
        layers_html += f"<tr><td><code>{layer}</code></td><td>{parts}</td></tr>"
    if not layers_html:
        layers_html = "<tr><td colspan=2>no metrics yet</td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fusion-cache dashboard</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:40px 20px; }}
  .wrap {{ max-width:760px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; border-bottom:1px solid #2a2d3a; padding-bottom:12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:24px 0; }}
  .card {{ background:#171a23; border:1px solid #2a2d3a; border-radius:10px; padding:16px; }}
  .card .v {{ font-size:1.5rem; font-weight:700; color:#4fd1c5; }}
  .card .l {{ font-size:.75rem; color:#8b90a0; text-transform:uppercase; letter-spacing:.05em; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; background:#171a23; border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:10px 14px; border-bottom:1px solid #2a2d3a; font-size:.85rem; }}
  th {{ background:#1c2030; color:#8b90a0; text-transform:uppercase; font-size:.7rem; letter-spacing:.05em; }}
  .muted {{ color:#8b90a0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>⚡ fusion-cache dashboard</h1>
  <div class="cards">
    <div class="card"><div class="v">{stats['requests']}</div><div class="l">requests</div></div>
    <div class="card"><div class="v">{stats['exact_hits']}</div><div class="l">L1 exact hits</div></div>
    <div class="card"><div class="v">{stats['semantic_hits']}</div><div class="l">L2 semantic hits</div></div>
    <div class="card"><div class="v">{stats['prefix_hits']}</div><div class="l">L3 prefix hits</div></div>
    <div class="card"><div class="v">{stats['hit_rate']:.2%}</div><div class="l">hit rate</div></div>
    <div class="card"><div class="v">${stats['cost_saved_usd']:.4f}</div><div class="l">cost saved</div></div>
  </div>
  <h2 class="muted" style="font-size:1rem">Per-layer metrics</h2>
  <table>
    <tr><th>layer</th><th>values</th></tr>
    {layers_html}
  </table>
</div>
</body>
</html>"""


app = create_app()
