# Agent Integration Guide

How to put fusion-cache in front of an AI coding agent (or any LLM app) so
it spends less and responds faster — **without changing the agent's code**.

## The idea: two channels

```
agent (Claude Code / opencode / pi / Cursor / ...)
   │
   ├─ LLM requests ──▶ fusion-cache gateway ──▶ upstream LLM
   │                   (point base_url at it)    (DeepSeek/OpenAI/...)
   │
   └─ MCP tools ──▶ fusion-cache MCP server ──▶ observe / manage the cache
```

- **Caching happens in the gateway.** The agent just talks to a different
  URL. Zero code changes.
- **The MCP server lets the agent see the savings** and flush the cache.

## 1. Start the gateway

```bash
pip install "fusion-cache[gateway,redis]"

export FUSION_UPSTREAM_BASE_URL=https://api.deepseek.com
export DEEPSEEK_API_KEY=sk-...
export FUSION_GATEWAY_API_KEY=change-me   # protect /v1/*
fusion-cache serve --port 8000
```

## 2. Point your agent at it

Most agents read an OpenAI-compatible `base_url` / `OPENAI_BASE_URL`
environment variable. Set it and the agent's LLM traffic flows through the
cache:

### DeepSeek-based agents (pi, custom harnesses)

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=anything            # the gateway forwards its own key
```

### Claude Code

Claude Code uses its own gateway protocol; the closest integration is via
`ANTHROPIC_BASE_URL` + the gateway's `anthropic` provider mode:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=anything
# run the gateway with: FUSION_UPSTREAM_PROVIDER=anthropic
```

### opencode / Cursor / any OpenAI-compatible client

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
```

If the agent supports MCP servers, also register the MCP server (step 4) so
it can report savings.

## 3. Verify it's caching

```bash
curl http://localhost:8000/health          # → {"status":"ok"}
curl -H "Authorization: Bearer change-me" http://localhost:8000/metrics
# look at fusion_cache_hit_rate, fusion_cache_cost_saved_usd_total
```

Or open http://localhost:8000/dashboard in a browser.

## 4. Add the MCP server (optional but recommended)

```bash
pip install "fusion-cache[mcp]"
fusion-cache-mcp        # runs over stdio
```

Register it with your agent. In Claude Code:

```json
// .mcp.json
{
  "mcpServers": {
    "fusion-cache": { "command": "fusion-cache-mcp" }
  }
}
```

Then ask your agent:

- "How much has fusion-cache saved?" → `cache_stats`
- "Is the circuit breaker open?" → `cache_status`
- "Flush the cache, I changed the model" → `cache_invalidate {"all": true}`
- "What's the similarity threshold?" → `cache_config`

## 5. What you get

- **Fewer upstream calls**: identical requests replay from L1 (exact), similar
  requests from L2 (semantic). Measured at **67–97% hit rate** on real
  workloads.
- **Faster responses**: cache hits return in **~0.1 ms** vs seconds upstream.
- **Cheaper long contexts**: the gateway passes through and *accounts* the
  upstream prefix-cache discount (DeepSeek: ~31× cheaper cache-hit input
  tokens). The `cached-token ratio` on a real opencode workload was **53%**.
- **Visibility**: hit rate / cost saved / latency are all exposed via
  `/metrics` (Prometheus), `/dashboard` (HTML), and the MCP tools.

## Multi-agent / team setup

Run one shared gateway behind your team's agent configs and point everyone
at it. With `REDIS_URL` set, all agents share the cache (hit rate stays high
across instances). See [DEPLOYMENT.md](DEPLOYMENT.md).
