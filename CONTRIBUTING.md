# Contributing to fusion-cache

Thanks for your interest! This project is small and focused; here's how to
help without getting in the way.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,gateway,redis]"
pytest
```

Requires Python ≥ 3.10.

## Design principles

1. **Framework-agnostic core.** `fusion_cache.core` has zero dependency on
   FastAPI/uvicorn — the gateway is an optional extra. Keep it that way.
2. **Honest metrics.** Never fabricate savings. L3 savings must come from
   provider-returned usage fields (`prompt_cache_hit_tokens` / `cached_tokens`);
   L1/L2 savings are upstream calls avoided. If a provider doesn't expose
   cache fields, report 0 — don't guess.
3. **No vector DB in the core.** The semantic layer uses numpy cosine over
   an in-memory or Redis store. Avoid pulling in Milvus/FAISS.
4. **Streaming is buffered-then-replayed** (MVP tradeoff: higher TTFB, full
   savings). Document any change to this.

## What's good for a PR

- Bug fixes with a regression test.
- New store adapters (Redis is done; add your favorite if it follows the
  `Store` protocol).
- New provider adapters in `gateway/upstream.py` (normalize to the OpenAI
  chat-completions shape).
- Dashboard/metrics improvements that stay self-contained (no CDN).

## Before submitting

- `pytest` passes (the suite runs against a fake OpenAI server; no keys needed).
- No new hard dependencies in `fusion_cache.core`.
- If you changed behavior, update the README.

## Code style

- `ruff` line length 100. `python -m ruff check .` should be clean.
- Type hints on all public functions.
- `from __future__ import annotations` in every module.
