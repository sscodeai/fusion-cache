# Deployment Guide

How to run fusion-cache in production, from a single instance to a
horizontally-scaled gateway fleet.

## 1. Single instance (simplest)

```bash
pip install fusion-cache[gateway]
export FUSION_UPSTREAM_BASE_URL=https://api.deepseek.com
export DEEPSEEK_API_KEY=sk-...
export FUSION_GATEWAY_API_KEY=change-me     # protect /v1/*
fusion-cache serve --port 8000
```

- Cache is in-memory (`MemoryStore`). Restart = cache cleared (cold start).
- Fine for low traffic / single process. No Redis needed.

## 2. Multi-instance (horizontal scale)

```
        LB (Traefik/nginx + TLS)
        ├── gateway-1 ─┐
        ├── gateway-2 ─┼── Redis (shared cache)
        └── gateway-3 ─┘
```

```bash
# Each gateway:
export REDIS_URL=redis://redis-host:6379/0
fusion-cache serve --port 8000
```

- **Redis is required for multi-instance** — otherwise each instance has its
  own cache and hit rate divides by instance count.
- Enable Redis persistence (AOF) for cache durability across restarts:
  `appendonly yes` in redis.conf.

## 3. Docker Compose (gateway + Redis)

```bash
docker compose up --build
# gateway on :8000, redis on :6379
```

Set env in `docker-compose.yml` (see `.env.example`).

## 4. Observability

The gateway exposes Prometheus metrics at `/metrics`.

```yaml
# prometheus.yml scrape config
scrape_configs:
  - job_name: fusion-cache
    metrics_path: /metrics
    static_configs:
      - targets: ["gateway:8000"]
```

- **Alerting rules**: `observability/prometheus-alerts.yml` (hit-rate drop,
  all-misses, no-savings).
- **Grafana dashboard**: import `observability/grafana-dashboard.json`
  (requests, hit rate, cost saved, upstream calls, per-layer hits).

## 5. Security checklist

- [ ] `FUSION_GATEWAY_API_KEY` set — protects `/v1/*` (health/metrics/dashboard stay open)
- [ ] Rate limiting on: `FUSION_RATE_LIMIT=60` (per minute per key)
- [ ] TLS at the LB (never expose plain HTTP to the public internet)
- [ ] Upstream keys in the environment / secret manager, NOT in the repo
- [ ] `.env` gitignored (already done); use `.env.example` as the template
- [ ] `FUSION_CORS_ORIGINS` set if the dashboard is accessed from a browser

## 6. Operations

| Action | How |
|---|---|
| Flush all cached responses (after model change) | `POST /v1/cache/invalidate {"all": true}` (auth required) |
| Evict one request | `POST /v1/cache/invalidate {"request": {...}}` |
| Check health | `GET /health` |
| View stats | `GET /metrics` or `/dashboard` |
| CLI stats | `fusion-cache stats --url http://gateway:8000` |
| `fusion-cache check` | environment health check (deps, upstream, key) |

## 7. Sizing

| Component | Notes |
|---|---|
| Gateway | ~50-80 MB RSS per instance (Python runtime); 1000 cached responses ≈ 1 MB |
| Redis | Shared cache; size = entries × entry size, bounded by TTL |
| Scaling | Gateway scales horizontally; Redis is the shared layer; upstream rate limits are the real ceiling |

## 8. Graceful degradation

If the upstream LLM goes down:

- The **circuit breaker** opens after N consecutive failures
  (`FUSION_CIRCUIT_BREAKER_FAILURE_THRESHOLD`, default 5).
- While open, **cached hits are still served** — only requests that need a
  fresh upstream call fail fast.
- After cooldown (`FUSION_CIRCUIT_BREAKER_COOLDOWN_S`, default 30s), one
  probe call is allowed; success re-closes the breaker.
