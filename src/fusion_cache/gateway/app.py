"""FastAPI gateway for fusion-cache: an OpenAI-compatible reverse proxy.

The gateway exposes the same surface as the OpenAI API (``/v1/chat/completions``,
``/v1/models``) but routes every request through the three-layer fusion cache
before (optionally) hitting the real upstream.  It also exposes:

- ``GET /metrics``        — Prometheus-style text metrics (or JSON when
                            ``Accept: application/json``)
- ``GET /dashboard``      — self-contained HTML dashboard (inline CSS/JS, no CDN)
- ``GET /health``         — liveness probe

Configuration (env vars):

- ``FUSION_UPSTREAM_BASE_URL``   — upstream base URL (default https://api.deepseek.com)
- ``FUSION_UPSTREAM_PROVIDER``   — ``openai`` (default, any OpenAI-compatible
  endpoint) or ``anthropic``
- ``FUSION_UPSTREAM_API_KEY`` / ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY`` /
  ``ANTHROPIC_API_KEY``          — upstream credentials (provider-dependent)
- ``FUSION_GATEWAY_API_KEY``     — if set, require ``Authorization: Bearer <key>``
  on all /v1/* requests (health/metrics/dashboard stay open)
- ``FUSION_CORS_ORIGINS``        — comma-separated allowed origins (default: none)
- ``REDIS_URL``                  — if set, use RedisStore (else in-memory)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.gateway.upstream import call_upstream_nonstream, call_upstream_stream
from fusion_cache.stores.redis import RedisStore

logger = logging.getLogger("fusion_cache.gateway")

DEFAULT_UPSTREAM = os.environ.get("FUSION_UPSTREAM_BASE_URL", "https://api.deepseek.com")
DEFAULT_PROVIDER = os.environ.get("FUSION_UPSTREAM_PROVIDER", "openai")


def _build_cache() -> FusionCache:
    """Construct the shared cache: RedisStore when REDIS_URL is set, else memory."""
    cfg = FusionCacheConfig.from_env()
    store = None
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        store = RedisStore(url=redis_url)
    return FusionCache(config=cfg, store=store)


def _gateway_api_key() -> Optional[str]:
    key = os.environ.get("FUSION_GATEWAY_API_KEY", "")
    return key or None


def _cors_origins() -> list[str]:
    raw = os.environ.get("FUSION_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(
    cache: Optional[FusionCache] = None,
    upstream_base_url: Optional[str] = None,
    provider: Optional[str] = None,
    gateway_api_key: Optional[str] = None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create the FastAPI application.

    All parameters are injectable for tests; defaults build from the
    environment.
    """
    app = FastAPI(title="fusion-cache gateway", version="0.1.0")
    app.state.cache = cache or _build_cache()
    app.state.upstream_base_url = (upstream_base_url or DEFAULT_UPSTREAM).rstrip("/")
    app.state.provider = provider or DEFAULT_PROVIDER
    app.state.gateway_api_key = gateway_api_key if gateway_api_key is not None else _gateway_api_key()
    origins = cors_origins if cors_origins is not None else _cors_origins()

    # ---- middleware: CORS -------------------------------------------------
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ---- auth dependency ---------------------------------------------------
    def _check_auth(request: Request) -> Optional[JSONResponse]:
        key = app.state.gateway_api_key
        if not key:
            return None
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {key}"
        if auth == expected:
            return None
        return JSONResponse(status_code=401, content={"error": {"message": "invalid or missing API key"}})

    # ---- routes ------------------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> Any:
        auth_err = _check_auth(request)
        if auth_err:
            return auth_err
        # Passthrough to the upstream models endpoint (best effort).
        from httpx import AsyncClient

        async with AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(f"{app.state.upstream_base_url}/models")
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return {"object": "list", "data": [{"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        cache: FusionCache = app.state.cache
        start = time.perf_counter()

        auth_err = _check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})

        stream = bool(body.get("stream", False))

        async def upstream(**kwargs: Any) -> Any:
            """Call the real upstream via the provider adapter."""
            if stream:
                return await call_upstream_stream(app.state.provider, app.state.upstream_base_url, dict(kwargs))
            # Retry on 429 (rate limit) with a small backoff.
            max_retries = int(os.environ.get("FUSION_UPSTREAM_RETRIES", "2"))
            delay = 0.5
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await call_upstream_nonstream(
                        app.state.provider, app.state.upstream_base_url, dict(kwargs)
                    )
                except RuntimeError as exc:
                    last_exc = exc
                    if "429" in str(exc) and attempt < max_retries:
                        logger.warning("upstream 429, retrying in %.1fs (attempt %d)", delay, attempt + 1)
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise
            if last_exc is None:  # pragma: no cover
                raise RuntimeError("upstream call failed")
            raise last_exc

        try:
            result = await cache.chat_completion(request=body, upstream=upstream, stream=stream)
        except Exception as exc:
            logger.error("chat_completions failed: %s", exc)
            return JSONResponse(status_code=502, content={"error": {"message": f"upstream failure: {exc}"}})

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "chat_completions layer=%s hit=%s latency=%.1fms cost_saved=%.6f",
            result.layer, result.hit, elapsed_ms, result.cost_saved_usd,
        )

        if stream:
            async def sse() -> AsyncIterator[str]:
                for chunk in result.stream_chunks:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        payload = result.response if isinstance(result.response, dict) else _response_to_dict(result.response)
        payload = dict(payload)
        payload["_fusion_cache"] = {
            "layer": result.layer,
            "hit": result.hit,
            "cached": result.cached,
            "prefix_hit": result.prefix_hit,
            "cost_saved_usd": round(result.cost_saved_usd, 8),
            "latency_ms": round(elapsed_ms, 2),
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
        series = cache.metrics.latency_series(120)
        return _render_dashboard(stats, latency_series=series)

    return app


def _response_to_dict(resp: Any) -> Dict[str, Any]:
    """Best-effort conversion of an OpenAI-style response object to a dict."""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if isinstance(resp, dict):
        return resp
    return {"object": "chat.completion", "choices": [], "usage": {}}


def _render_dashboard(stats: Dict[str, Any], latency_series: Optional[List[float]] = None) -> str:
    """Self-contained HTML dashboard (inline CSS/SVG, no external resources)."""
    metrics = stats.get("metrics", {})
    layers = metrics.get("layers", metrics) if isinstance(metrics, dict) else {}
    total_req = stats.get("requests", 0) or sum(
        v.get("hits", 0) + v.get("misses", 0) for v in layers.values() if isinstance(v, dict)
    )
    total_err = sum(v.get("errors", 0) for v in layers.values() if isinstance(v, dict))
    err_rate = (total_err / total_req) if total_req else 0.0

    lat = metrics.get("latency", {}) if isinstance(metrics, dict) else {}
    p50 = lat.get("p50_ms", 0.0)
    p95 = lat.get("p95_ms", 0.0)
    mean = lat.get("mean_ms", 0.0)
    uptime = metrics.get("uptime_s", 0) if isinstance(metrics, dict) else 0

    layers_html = ""
    for layer, d in layers.items():
        if not isinstance(d, dict):
            continue
        hits = d.get("hits", 0)
        misses = d.get("misses", 0)
        errors = d.get("errors", 0)
        saved = d.get("cost_saved_usd", 0.0)
        tokens = d.get("tokens", 0)
        total = hits + misses
        rate = (hits / total) if total else 0.0
        layers_html += (
            f"<tr><td><code>{layer}</code></td>"
            f"<td>{hits}</td><td>{misses}</td><td>{errors}</td>"
            f"<td>{rate:.0%}</td><td>${saved:.6f}</td><td>{tokens}</td></tr>"
        )
    if not layers_html:
        layers_html = "<tr><td colspan=7>no metrics yet</td></tr>"

    spark = _sparkline_svg(latency_series or [])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fusion-cache dashboard</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:40px 20px; }}
  .wrap {{ max-width:860px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; border-bottom:1px solid #2a2d3a; padding-bottom:12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:24px 0; }}
  .card {{ background:#171a23; border:1px solid #2a2d3a; border-radius:10px; padding:16px; }}
  .card .v {{ font-size:1.5rem; font-weight:700; color:#4fd1c5; }}
  .card .l {{ font-size:.75rem; color:#8b90a0; text-transform:uppercase; letter-spacing:.05em; margin-top:4px; }}
  .card .v.red {{ color:#f87171; }}
  .card .v.yellow {{ color:#fbbf24; }}
  table {{ width:100%; border-collapse:collapse; background:#171a23; border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:10px 14px; border-bottom:1px solid #2a2d3a; font-size:.85rem; }}
  th {{ background:#1c2030; color:#8b90a0; text-transform:uppercase; font-size:.7rem; letter-spacing:.05em; }}
  .muted {{ color:#8b90a0; }}
  .spark {{ background:#171a23; border:1px solid #2a2d3a; border-radius:10px; padding:12px; margin:18px 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>⚡ fusion-cache dashboard</h1>
  <div class="cards">
    <div class="card"><div class="v">{total_req}</div><div class="l">requests</div></div>
    <div class="card"><div class="v">{stats.get('exact_hits', 0)}</div><div class="l">L1 exact hits</div></div>
    <div class="card"><div class="v">{stats.get('semantic_hits', 0)}</div><div class="l">L2 semantic hits</div></div>
    <div class="card"><div class="v">{stats.get('prefix_hits', 0)}</div><div class="l">L3 prefix hits</div></div>
    <div class="card"><div class="v">{stats.get('hit_rate', 0):.2%}</div><div class="l">hit rate</div></div>
    <div class="card"><div class="v">${stats.get('cost_saved_usd', 0):.4f}</div><div class="l">cost saved</div></div>
    <div class="card"><div class="v {'red' if err_rate > 0.05 else 'yellow' if err_rate > 0 else ''}">{err_rate:.2%}</div><div class="l">error rate</div></div>
    <div class="card"><div class="v">{p50:.0f}ms</div><div class="l">P50 latency</div></div>
    <div class="card"><div class="v">{p95:.0f}ms</div><div class="l">P95 latency</div></div>
    <div class="card"><div class="v">{uptime:.0f}s</div><div class="l">uptime</div></div>
  </div>
  <div class="spark">
    <div class="muted" style="font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Latency trend (recent requests, ms)</div>
    {spark}
  </div>
  <h2 class="muted" style="font-size:1rem">Per-layer metrics</h2>
  <table>
    <tr><th>layer</th><th>hits</th><th>misses</th><th>errors</th><th>hit rate</th><th>$ saved</th><th>tokens</th></tr>
    {layers_html}
  </table>
</div>
</body>
</html>"""


def _sparkline_svg(series: List[float], width: int = 820, height: int = 60) -> str:
    """Inline SVG sparkline of the latency series (no external JS)."""
    if len(series) < 2:
        return '<span class="muted">collecting data…</span>'
    vmax = max(series) or 1.0
    vmin = min(series)
    rng = (vmax - vmin) or 1.0
    n = len(series)
    points = []
    for i, v in enumerate(series):
        x = (i / (n - 1)) * width
        y = height - ((v - vmin) / rng) * (height - 8) - 4
        points.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(points)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">'
        f'<polyline points="{poly}" fill="none" stroke="#4fd1c5" stroke-width="2" stroke-linejoin="round"/>'
        f'<circle cx="{width:.1f}" cy="{points[-1].split(",")[1]}" r="3" fill="#4fd1c5"/>'
        f'</svg>'
    )


app = create_app()
