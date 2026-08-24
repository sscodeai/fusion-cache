# fusion-cache

Fusion cache for LLM APIs: **exact → semantic → prefix-cache accounting** in one
framework-agnostic Python middleware layer. Drop-in wrapper for the OpenAI SDK,
DeepSeek-first pricing model, honest money-saved metrics.

> **Status:** v0.1.0 (MVP) + **v1.1 (gateway form)**: FastAPI OpenAI-compatible
> reverse proxy, Redis store, CLI, Docker, and self-contained HTML dashboard.
> Wrapper form, in-memory store, all three layers, metrics — all shipped.

---

## Why "fusion"?

Most "semantic caches" are single-layer: they embed every query and hope a
vector search is right. Most "exact caches" miss every paraphrase. fusion-cache
runs three layers in order, each with a distinct job:

| Layer | Job | Mechanism | Cost of a hit |
|---|---|---|---|
| **L1 exact** | byte-identical requests | canonicalized SHA-256 key (normalizes whitespace/param noise) | ~0 ms, in-process |
| **L2 semantic** | paraphrased requests | embedding + numpy cosine vs stored rows, configurable threshold + structural guardrail | 1 embed call |
| **L3 prefix** | *never a cache miss* | pass through to DeepSeek's automatic prefix cache, capture `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, account the discount | 0 extra calls |

L3 is the differentiator most caches ignore: DeepSeek already gives you a
~31× discount on cache-hit input tokens (off-peak, deepseek-v4-flash:
`$0.007` vs `$0.22` per 1M). Nobody reports it back to you as money saved —
fusion-cache does, with a pinned, configurable price model.

```
 request ──▶ L1 exact ── hit? ──▶ replay (0 ms)
               │
               ▼ miss
            L2 semantic (embed + cosine) ── hit? ──▶ replay (guardrail-checked)
               │
               ▼ miss
            L3 pass-through ──▶ upstream (chat.completions.create)
                 └─ capture prompt_cache_hit/miss_tokens ──▶ $ saved
                 └─ populate L1 (+ L2 row)
```

---

## Install

```bash
pip install -e .            # core (pydantic, httpx, numpy)
pip install -e ".[test]"    # + pytest, pytest-asyncio, respx
pip install -e ".[redis]"   # + optional redis.asyncio store
```

Requires Python ≥ 3.10.

---

## Quickstart: DeepSeek

```python
import openai
from fusion_cache import FusionCache
from fusion_cache.wrapper.openai import CachedOpenAI

cache = FusionCache()  # defaults: exact+semantic+prefix accounting on
client = CachedOpenAI(openai.OpenAI(
    api_key="sk-...",           # or DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com",
), cache=cache)

# First call → upstream (recorded as a miss).
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Explain prefix caching"}],
)

# Identical call → replayed from L1 (exact hit).
# Paraphrased call → L2 semantic hit (needs an embedder key, see below).
# Either way the upstream sees stable prefixes → DeepSeek bills fewer
# input tokens, and L3 accounting reports the discount.

print(cache.stats_dict())
# {
#   'requests': 2, 'exact_hits': 1, 'semantic_hits': 0, 'prefix_hits': 1,
#   'misses': 1, 'cost_saved_usd': 0.00001704, 'hit_rate': 1.0,
#   'metrics': {...}   # per-layer hit/miss, P50/P95 latency
# }
```

Async is identical — wrap `AsyncOpenAI` and `await` the call.

### Embedder for the L2 semantic layer

L2 needs an OpenAI-compatible `/embeddings` endpoint. DeepSeek serves one at
`https://api.deepseek.com` (model `deepseek-embedding`); any compatible
endpoint works via `base_url`:

```python
from fusion_cache import FusionCacheConfig

cache = FusionCache(FusionCacheConfig(
    similarity_threshold=0.93,                      # raise → fewer (safer) hits
    embedder={"base_url": "https://api.deepseek.com",
              "api_key": "sk-...",
              "model": "deepseek-embedding"},
))
```

No embedder key → the L2 layer is skipped automatically (L1 + L3 still work).

---

## Configuration

All knobs live in [`FusionCacheConfig`](src/fusion_cache/config.py) (pydantic v2):

```python
from fusion_cache import FusionCacheConfig

FusionCacheConfig(
    enable_exact=True, enable_semantic=True, enable_prefix_accounting=True,
    exact_ttl=3600.0, semantic_ttl=7200.0, semantic_grace_ttl=3600.0,
    similarity_threshold=0.93,
    max_entries=10_000, semantic_max_entries=10_000,
    price_model={   # DeepSeek off-peak snapshot (USD per 1M tokens, 2026-08)
        "input_miss_per_1m": 0.22,
        "input_hit_per_1m": 0.007,
        "output_per_1m": 1.1,
    },
)
```

### Streaming (buffer-then-replay)

`stream=True` requests are fully buffered by the pipeline, stored, then
replayed chunk-by-chunk to the caller with the terminal `usage` chunk
preserved. MVP trades a higher TTFB for full savings and byte-exact replay
(decision documented in `open-source-decision.md` §6.6). Async and sync
clients both return a chunk iterator matching the wrapped client's style.

---

## Architecture

```
src/fusion_cache/
├── config.py              # pydantic v2 config: layer toggles, TTLs, threshold,
│                          #   DeepSeek price model, embedder settings
├── core/
│   ├── key.py             # L1 canonicalization (normalize model/temp/stream)
│   ├── pipeline.py        # exact → semantic → prefix, async, per-layer TTL
│   └── replay.py          # buffered streaming replay + stream text helpers
├── stores/
│   ├── base.py            # Store protocol (get/set/delete, TTL, keys)
│   ├── memory.py          # in-memory LRU/TTL store (default)
│   └── redis.py           # optional redis.asyncio adapter (JSON + native EX)
├── semantic/
│   ├── embedder.py        # OpenAI-compatible embed client (base_url configurable)
│   └── matcher.py         # numpy cosine + threshold + false-positive guardrail
├── prefix/
│   └── accounting.py      # prompt_cache_hit/miss_tokens → $ saved
├── metrics/
│   └── registry.py        # per-layer hit/miss, hit rate, $ saved, P50/P95
├── gateway/
│   └── app.py             # FastAPI OpenAI-compatible reverse proxy (v1.1)
├── cli.py                 # fusion-cache serve / stats / check (v1.1)
└── wrapper/
    └── openai.py          # CachedOpenAI: drop-in sync/async OpenAI wrapper
```

### Stores

- **MemoryStore** (default): dict + LRU-by-use eviction, lazy TTL expiry,
  thread-safe.
- **RedisStore** (optional, `[redis]` extra): `redis.asyncio` adapter, JSON
  entries under a prefix, TTL via native `SET ... EX`. Async-first
  (`aget`/`aset`/…); the wrapper drives it through the event loop.

### Metrics

`cache.metrics.snapshot()` returns per-layer hit/miss counts and hit rate,
cost saved (USD), prefix-cache tokens, and P50/P95/mean latency from a rolling
buffer. `cache.stats_dict()` combines it with aggregate counters. This is the
input for the v1.1 dashboard.

---

## Gateway (v1.1)

The FastAPI gateway is an OpenAI-compatible reverse proxy: your app talks to
the gateway exactly like it talks to DeepSeek/OpenAI, and the gateway routes
every request through the fusion cache before hitting the real upstream.

### Run it

```bash
pip install -e ".[gateway,redis]"

# In-memory store:
fusion-cache serve --port 8000

# With Redis (shared cache across instances):
fusion-cache serve --port 8000 --redis redis://localhost:6379/0
```

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `FUSION_UPSTREAM_BASE_URL` | `https://api.deepseek.com` | upstream base URL |
| `FUSION_UPSTREAM_PROVIDER` | `openai` | `openai` (any OpenAI-compatible endpoint) or `anthropic` |
| `FUSION_UPSTREAM_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | upstream credentials (provider-dependent) |
| `FUSION_GATEWAY_API_KEY` | — | if set, require `Authorization: Bearer <key>` on `/v1/*` (health/metrics/dashboard stay open) |
| `FUSION_CORS_ORIGINS` | — | comma-separated allowed origins (e.g. `https://app.example.com,https://admin.example.com`) |
| `FUSION_UPSTREAM_RETRIES` | `2` | max retries on upstream 429 (exponential backoff) |
| `REDIS_URL` | — | enable RedisStore (else in-memory) |
| `FUSION_HOST` / `FUSION_PORT` | `0.0.0.0` / `8000` | bind address |

### Provider support

- **`openai`** (default): any OpenAI-compatible endpoint — DeepSeek, OpenAI,
  OpenRouter, local vLLM/Ollama, etc. Just point `FUSION_UPSTREAM_BASE_URL` at it.
- **`anthropic`**: set `FUSION_UPSTREAM_PROVIDER=anthropic`. The gateway
  translates the request to `/v1/messages` with `x-api-key` auth and normalizes
  the response back to the OpenAI shape. Note: Anthropic does not expose
  DeepSeek-style `prompt_cache_hit_tokens`, so L3 prefix accounting reports 0
  for this provider (honest — no discount to count).

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions through the cache (streaming via SSE supported) |
| `GET /v1/models` | model list (passthrough, with fallback) |
| `GET /metrics` | Prometheus metrics (or JSON with `Accept: application/json`) |
| `GET /dashboard` | self-contained HTML dashboard (no CDN) |
| `GET /health` | liveness probe |

Non-streaming responses carry a `_fusion_cache` metadata block:

```json
{
  "_fusion_cache": {
    "layer": "exact",
    "hit": true,
    "cached": true,
    "prefix_hit": true,
    "cost_saved_usd": 0.000017,
    "latency_ms": 0.42
  }
}
```

### Docker

```bash
docker compose up --build
# gateway on :8000, redis on :6379
```

### CLI

```bash
fusion-cache serve --port 8000            # start gateway
fusion-cache stats --url http://localhost:8000   # print cache stats
fusion-cache check                        # environment health check
```

---

## Comparison (honest)

| | fusion-cache | GPTCache | LiteLLM | usewhale/Whale |
|---|---|---|---|---|
| Exact → semantic → prefix fusion | ✅ built-in | ❌ semantic only | ❌ prefix = passthrough | n/a (coding agent) |
| DeepSeek prefix `$` accounting | ✅ first-class | ❌ | ❌ | partial (agent-internal) |
| Money-saved / hit-rate metrics | ✅ dashboard-ready JSON | ❌ | partial | agent-specific |
| Form | pip package, wrapper-first | library + vector DB | gateway service | terminal agent |
| Deployment weight | lightweight | needs Milvus + scalar store | full service | n/a |

Research notes: GPTCache is dormant (last release 2024-08), LiteLLM is a
general gateway where caching is 1 of hundreds of features, and
`usewhale/Whale` is a DeepSeek **coding agent** (its ~98% prompt-cache-hit
pitch is our differentiation target, not a competitor). No maintained project
ships fusion + DeepSeek accounting as a general middleware. See
[`open-source-decision.md`](open-source-decision.md) for the full matrix.

---

## Benchmark methodology

The benchmark targets in the decision doc (`.00 → .18` USD/task, P95
4.2s → 1.7s) are **targets the bench script must reproduce**, not yet-measured
numbers. Reproducing them requires a real DeepSeek key; the MVP test suite
runs against a fake OpenAI server instead.

Methodology (when a key is available):

1. **Workload:** N distinct tasks with shared system prompt + stable
   prompt-first layout, issued 3× each interleaved (warm/cold cache mix).
2. **Upstream:** DeepSeek `deepseek-chat`, pinned model + `temperature=0.2`,
   `stream=False`; no other middleware in the path.
3. **Warm vs cold:** cold = first run on an empty cache; warm = subsequent runs.
4. **Metrics:** exact/semantic/prefix hit counts, hit rate, per-request
   latency (P50/P95) and cost; cost = `sum(prompt_cache_miss_tokens × miss
   price + prompt_cache_hit_tokens × hit price + completion_tokens × output
   price)` from real usage fields.
5. **Report:** per-layer hit table, P50/P95 before/after, `$`/task
   before/after, and the percentage of cost reduction attributable to each
   layer.

Honest claims only: L3 savings are real upstream discounts (usage fields are
provider-returned); L1/L2 savings are upstream calls avoided.

---

## License

Apache-2.0. Built from scratch; reference material only was reused from
GPTCache/LiteLLM/semcache APIs (MIT/Apache-2.0) — no code copied.
