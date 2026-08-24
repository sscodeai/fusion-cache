<div align="center">

# ⚡ fusion-cache

### Cut your LLM API costs by up to **97%** — and drop P95 latency from seconds to **0.7 ms**.

A framework-agnostic caching layer for LLM APIs: **exact → semantic → prefix-cache
accounting** in one drop-in middleware. Wrap your OpenAI client (or point your
gateway at it) and repeated requests stop costing you money.

**Measured on real workloads** (see [Benchmark](#benchmark)):

| 🎯 97% hit rate | ⚡ 0.7 ms P95 | 💰 ~31× prefix discount | 🚀 16,000× faster hits |
|---|---|---|---|
| L1 exact + L2 semantic serve 58/60 requests from cache | vs 6.5 s uncached | DeepSeek cache-hit input tokens | P50 3.2 s → 0.1 ms |

</div>

---

## ✨ What it does

| Layer | Job | Cost of a hit |
|---|---|---|
| **L1 exact** | byte-identical requests → instant replay | ~0 ms |
| **L2 semantic** | paraphrased requests → replay (guardrail-checked) | 1 embed call |
| **L3 prefix** | *never a miss* — passes through and **accounts** the upstream prefix-cache discount (DeepSeek ~31×) | 0 extra calls |

The pain: LLM API prices keep rising and your bill is mostly repeated compute.
The fix: cache the repeats (L1/L2), and stop ignoring the prefix-cache discount
your provider already gives you (L3) — then **show you the money saved**.

```bash
pip install fusion-cache
```

```python
from fusion_cache import FusionCache
from fusion_cache.wrapper.openai import CachedOpenAI
import openai

client = CachedOpenAI(openai.OpenAI(api_key=...), cache=FusionCache())
# first call hits upstream, identical calls replay in ~0 ms
```

> **Want it in front of an AI agent (pi, Claude Code, opencode)?** Point its
> `OPENAI_BASE_URL` at the gateway or add the MCP server — zero code changes.
> See [Agent Integration](docs/agent-integration.md).

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
fusion-cache-mcp                          # MCP server (stdio) for agents
```

### Agent integration (MCP + gateway)

fusion-cache works with AI coding agents (pi, Claude Code, opencode, Cursor)
in two channels:

- **Gateway**: point the agent's `OPENAI_BASE_URL` at the gateway — all LLM
  traffic flows through the cache. Zero code changes.
- **MCP server**: `fusion-cache-mcp` exposes tools (`cache_stats`,
  `cache_invalidate`, `cache_status`, `cache_config`) so the agent can see
  and manage the cache.

```bash
pip install "fusion-cache[mcp]"
fusion-cache-mcp     # register with Claude Code via .mcp.json, etc.
```

See [docs/agent-integration.md](docs/agent-integration.md) for per-agent
setup. Try [examples/quickstart.py](examples/quickstart.py) for a 30-second
before/after demo.

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

## Benchmark

Real end-to-end results. Run `python benchmarks/bench.py` with any
OpenAI-compatible endpoint (set `FUSION_UPSTREAM_API_KEY` +
`--base-url`/`--model`).

### L1+L2 fusion on a stable endpoint (deepseek-v4-flash via commandcode)

20 distinct tasks × 3 reps (60 requests per config), shared system prompt,
`temperature=0.2`, `stream=False`.

| Config | Req | L1 hit | L2 hit | Upstream | Hit rate | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|
| baseline (no cache) | 60 | 0 | 0 | 60 | 0% | 3214.2 | 6539.2 |
| fusion (L1+L3) | 60 | 40 | 0 | 20 | 67% | 0.2 | 3450.8 |
| **fusion+sem (L1+L2+L3)** | 60 | 4 | **54** | **2** | **97%** | **0.1** | **0.7** |

What this shows:

- **67% of requests are served from the L1 exact cache** with just the exact
  layer (0.2 ms median vs 3.2 s upstream — ~16,000×).
- **Adding the L2 semantic layer lifts the hit rate to 97%**: L1 catches the
  4 identical repeats, L2 catches the other 54 (same questions asked slightly
  differently across the workload). Only 2/60 calls reach the upstream, and
  even **P95 drops to 0.7 ms**.
- This endpoint (commandcode's deepseek) does **not** enable upstream prefix
  caching (`cached_tokens` is always 0), so L3 accounting is 0 here — the L1/L2
  layers carry the win.

### L3 prefix-cache accounting on a caching upstream (ox-alpha-free via opencode)

The same workload against opencode's free endpoint (which *does* enable
automatic prefix caching):

| Config | Req | L1 hit | Upstream | Hit rate | Cached-token ratio | P50 (ms) | $ saved |
|---|---|---|---|---|---|---|---|
| baseline | 60 | 0 | 60 | 0% | 53% | 7498.1 | $0.000000 |
| fusion | 60 | 40 | 20 | 67% | 53% | 0.2 | $0.000273 |

- Every request also benefits from the upstream's own prefix cache
  (cached-token ratio 53%) — fusion-cache captures this and reports it as
  money saved via `prompt_cache_hit_tokens` / `cached_tokens` accounting.
- P95 stays high (~11 s) because the 20 cold misses still hit the slow
  free upstream.

### Paraphrase workload — the L2 semantic layer's payoff

L1 exact caching only catches *identical* requests. When users rephrase the
same question, only the L2 semantic layer can connect them. Benchmarked with
8 question groups × 3 paraphrases each against `ox-alpha-free` (the free
endpoint was flaky — 503s — so this is a 2-group subset; the script
`--paraphrase` mode skips failed requests instead of aborting):

| Config | Req | L1 hit | L2 hit | Upstream | Hit rate | P50 (ms) |
|---|---|---|---|---|---|---|
| fusion (L1+L3 only) | 6 | 0 | 0 | 6 | 0% | 21107.5 |
| **fusion+sem (L1+L2+L3)** | 6 | 0 | **5** | 1 | **83%** | **0.4** |

Without the semantic layer every paraphrase misses (different text → L1
can't match) and pays the ~21 s upstream. With it, 5/6 paraphrases are served
from cache in **0.4 ms**. The local deterministic embedder used here (char
n-grams) is a stand-in for any OpenAI-compatible `/embeddings` endpoint —
plug your own via `FUSION_EMBED_BASE_URL` / `FUSION_EMBED_API_KEY`.

### Methodology

1. **Workload:** N distinct tasks with shared system prompt + stable
   prompt-first layout, issued 3× each interleaved (warm/cold cache mix).
2. **Upstream:** any OpenAI-compatible endpoint (default `ox-alpha-free` on
   opencode.ai/zen/go/v1); pinned model + `temperature=0.2`, `stream=False`;
   no other middleware in the path.
3. **Warm vs cold:** cold = first run on an empty cache; warm = subsequent runs.
4. **Metrics:** exact/semantic/prefix hit counts, hit rate, per-request
   latency (P50/P95) and cost; cost = `sum(prompt_cache_miss_tokens × miss
   price + prompt_cache_hit_tokens × hit price + completion_tokens × output
   price)` from real usage fields.
5. **Report:** per-layer hit table, P50/P95 before/after, `$`/task
   before/after, and the percentage of cost reduction attributable to each
   layer.

Honest claims only: L3 savings are real upstream discounts (usage fields are
provider-returned); L1/L2 savings are upstream calls avoided. The original
decision-doc targets (`.00 → .18` USD/task, P95 4.2s → 1.7s) were goals for
this script to reproduce — actual numbers depend on the upstream's pricing
and latency; the framework-agnostic script lets you measure your own.

---

## Related work (academic basis)

The three layers map onto an active research lineage. If you're coming from
the literature, this is where each layer lives:

### Semantic caching (L2)

- **MeanCache: User-Centric Semantic Caching for LLM Web Services** —
  arXiv:2403.02694 (2024). Foundational semantic-cache work: embedding
  similarity + cached-response reuse. This is the direct ancestor of L2.
- **GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding
  Caching** — arXiv:2411.05276 (2024). Semantic embedding caching for cost and
  latency reduction — the same pitch fusion-cache makes.
- **From Exact Hits to Close Enough: Semantic Caching for LLM Embeddings** —
  arXiv:2603.03301 (2026). The exact→semantic progression, which is exactly
  the L1→L2 design here.
- **Continuous Semantic Caching for Low-Cost LLM Serving** —
  arXiv:2604.20021 (2026). Recent continuous-semantic-cache work — the
  direction is active.
- **Closing the Calibration Gap in Semantic Caching** — arXiv:2606.19719
  (2026). Calibrating semantic-cache thresholds to avoid wrong hits — the
  academic version of our false-positive guardrail.

### Prefix / KV caching (L3 relies on the upstream's mechanism)

- **Marconi: Prefix Caching for the Era of Hybrid LLMs** — arXiv:2411.19379
  (2024, MLSys '25 Outstanding Paper). System-level prefix caching; the
  academic counterpart of the automatic prefix caching DeepSeek/OpenAI
  provide, which L3 accounts for.
- **Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for
  LLM Prefix Caches** — arXiv:2605.18825 (2026). Semantic-aware eviction in
  prefix caches — semantic + prefix fusion in the same spirit as this project.

### Memory / context compression (future direction for L2)

- **AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems** —
  arXiv:2607.21604 (2026). KV reuse for agent memory — the same idea as
  observational-memory-style compression; a natural evolution path for the
  semantic layer on long-running agents.

---

## License

Apache-2.0. Built from scratch; reference material only was reused from
GPTCache/LiteLLM/semcache APIs (MIT/Apache-2.0) — no code copied.
