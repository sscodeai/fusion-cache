"""MCP server for fusion-cache.

Lets an MCP-capable agent (Claude Code, opencode, Cursor, ...) observe and
manage a fusion-cache instance:

- ``cache_stats``      — hit rate, per-layer hits, cost saved, latency.
- ``cache_invalidate`` — evict one request or the whole cache.
- ``cache_status``     — circuit-breaker state, store type, uptime.
- ``cache_config``     — current config (thresholds, TTLs, price model).

Note on architecture: the MCP server does NOT proxy LLM requests.  Caching
happens in the fusion-cache gateway (point the agent's ``base_url`` at it);
the MCP server is the observability/management channel so the agent can
*see* and *control* the cache. See docs/agent-integration.md.

Run with::

    fusion-cache-mcp            # stdio transport (default, for agents)

Or programmatically::

    from fusion_cache.mcp_server import build_server
    server = build_server(cache=my_cache)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from fusion_cache.core.pipeline import FusionCache

MCP_SERVER_NAME = "fusion-cache"


def _build_cache() -> FusionCache:
    """Build a FusionCache from the environment (same logic as the gateway)."""
    from fusion_cache.config import FusionCacheConfig
    from fusion_cache.stores.redis import RedisStore

    cfg = FusionCacheConfig.from_env()
    store = None
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        store = RedisStore(url=redis_url)
    return FusionCache(config=cfg, store=store)


def build_server(cache: Optional[FusionCache] = None) -> FastMCP:
    """Create the MCP server. ``cache`` injectable for tests."""
    cache = cache or _build_cache()
    mcp = FastMCP(MCP_SERVER_NAME)

    @mcp.tool()
    def cache_stats() -> Dict[str, Any]:
        """Return cache statistics: hit rate, per-layer hits, cost saved, latency."""
        return cache.stats_dict()

    @mcp.tool()
    def cache_status() -> Dict[str, Any]:
        """Return runtime status: circuit breaker state, store backend, uptime."""
        store_type = type(cache.store).__name__
        breaker = {
            "state": cache.breaker.state,
            "failures": cache.breaker._failures,
            "failure_threshold": cache.breaker.failure_threshold,
            "cooldown_s": cache.breaker.cooldown_s,
            "enabled": cache.breaker.enabled,
        }
        return {
            "store_type": store_type,
            "circuit_breaker": breaker,
            "requests": cache.stats.requests,
            "uptime_s": cache.metrics.snapshot().get("uptime_s", 0),
        }

    @mcp.tool()
    def cache_config() -> Dict[str, Any]:
        """Return the current cache configuration (thresholds, TTLs, price model, toggles)."""
        cfg = cache.config
        return {
            "enable_exact": cfg.enable_exact,
            "enable_semantic": cfg.enable_semantic,
            "enable_prefix_accounting": cfg.enable_prefix_accounting,
            "exact_ttl": cfg.exact_ttl,
            "semantic_ttl": cfg.semantic_ttl,
            "similarity_threshold": cfg.similarity_threshold,
            "max_entries": cfg.max_entries,
            "semantic_max_entries": cfg.semantic_max_entries,
            "price_model": cfg.price_model.model_dump(),
            "circuit_breaker": {
                "enabled": cfg.circuit_breaker_enabled,
                "failure_threshold": cfg.circuit_breaker_failure_threshold,
                "cooldown_s": cfg.circuit_breaker_cooldown_s,
            },
        }

    @mcp.tool()
    async def cache_invalidate(all: bool = False, request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evict cached responses.

        Args:
            all: if True, evict the entire cache.
            request: a dict shaped like an OpenAI chat request; evicts just
                that request's entries.
        Returns:
            {"evicted": n} where n is the number of entries removed.
        """
        if all:
            n = await cache.invalidate_all()
            return {"evicted": n, "all": True}
        if request is None:
            return {"error": "pass all=true or a request dict"}
        removed = await cache.invalidate(request)
        return {"evicted": 1 if removed else 0, "all": False}

    return mcp


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
