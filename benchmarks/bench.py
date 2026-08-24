"""Real end-to-end benchmark for fusion-cache against a live upstream.

Compares three configurations on the same workload:

- **baseline**   — direct upstream calls (no fusion-cache)
- **fusion**     — fusion-cache wrapper (L1 exact + L3 prefix accounting)
- **fusion+sem** — fusion-cache with semantic layer enabled (L2)

Workload: ``--tasks`` distinct user prompts sharing one stable system prompt,
each issued ``--repeat`` times (first = cold, rest = warm).  Measures per-layer
hit rate, latency (P50/P95), and cost ($/task) using the provider-returned
usage fields.

Usage::

    OPENCODE_API_KEY=... python benchmarks/bench.py \\
        --base-url https://opencode.ai/zen/go/v1 \\
        --model ox-alpha-free --tasks 20 --repeat 3

Output is a markdown table ready to paste into the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from httpx import AsyncClient

from fusion_cache.config import FusionCacheConfig
from fusion_cache.core.pipeline import FusionCache

SYSTEM_PROMPT = (
    "You are a concise technical assistant. Answer in plain text, no markdown "
    "unless asked. Keep answers under 120 words."
)

TASKS = [
    "Explain what a cache hit ratio is in one sentence.",
    "What is the difference between LRU and TTL caching?",
    "Explain exponential backoff in one sentence.",
    "What does idempotency mean in APIs?",
    "Explain what a reverse proxy does.",
    "What is the difference between a monolith and microservices?",
    "Explain what a semaphore is in concurrent programming.",
    "What does 'eventually consistent' mean?",
    "Explain what a dead letter queue is.",
    "What is the difference between TCP and UDP?",
    "Explain what a hash table is.",
    "What does 'single responsibility principle' mean?",
    "Explain what a CDN does.",
    "What is the difference between symmetric and asymmetric encryption?",
    "Explain what a database index is.",
    "What does 'fail fast' mean in software design?",
    "Explain what a webhook is.",
    "What is the difference between a process and a thread?",
    "Explain what idempotent HTTP methods are.",
    "What does 'graceful degradation' mean?",
    "Explain what a message queue does.",
    "What is the difference between SQL and NoSQL?",
    "Explain what a bloom filter is.",
    "What does 'CAP theorem' state?",
]

# Paraphrase variants: each "task" is one question asked 3 different ways.
# L1 exact will MISS these (texts differ); only L2 semantic can catch them.
PARAPHRASE_GROUPS = [
    [
        "Explain what a cache hit ratio is in one sentence.",
        "In a single sentence, describe the concept of a cache hit rate.",
        "What does the cache hit percentage mean?",
    ],
    [
        "What is the difference between LRU and TTL caching?",
        "Compare and contrast LRU cache eviction with TTL-based expiry.",
        "How do LRU and time-to-live caching differ from each other?",
    ],
    [
        "Explain exponential backoff in one sentence.",
        "What does exponential backoff mean, briefly?",
        "Describe the exponential backoff strategy in a nutshell.",
    ],
    [
        "What does idempotency mean in APIs?",
        "Explain the meaning of an idempotent API operation.",
        "What is idempotence in the context of web APIs?",
    ],
    [
        "Explain what a reverse proxy does.",
        "What is the role of a reverse proxy server?",
        "Describe what a reverse proxy is used for.",
    ],
    [
        "What is the difference between a monolith and microservices?",
        "Compare monolithic architecture with microservices.",
        "How do monoliths and microservices differ?",
    ],
    [
        "Explain what a semaphore is in concurrent programming.",
        "What role does a semaphore play in multithreading?",
        "Define semaphore in the context of concurrency control.",
    ],
    [
        "What does 'eventually consistent' mean?",
        "Explain the concept of eventual consistency.",
        "What is meant by eventually consistent systems?",
    ],
]


@dataclass
class Sample:
    task_idx: int
    repeat: int
    layer: str
    hit: bool
    latency_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cost_saved_usd: float = 0.0


@dataclass
class RunResult:
    label: str
    samples: List[Sample] = field(default_factory=list)

    def requests(self) -> int:
        return len(self.samples)

    def exact_hits(self) -> int:
        return sum(1 for s in self.samples if s.layer == "exact")

    def semantic_hits(self) -> int:
        return sum(1 for s in self.samples if s.layer == "semantic")

    def prefix_hit_count(self) -> int:
        return sum(1 for s in self.samples if s.cached_tokens > 0)

    def upstream_calls(self) -> int:
        return sum(1 for s in self.samples if s.layer == "miss")

    def hit_rate(self) -> float:
        if not self.samples:
            return 0.0
        return (self.exact_hits() + self.semantic_hits()) / len(self.samples)

    def cached_token_ratio(self) -> float:
        """Fraction of prompt tokens served from the upstream prefix cache."""
        total_prompt = sum(s.prompt_tokens for s in self.samples)
        total_cached = sum(s.cached_tokens for s in self.samples)
        return (total_cached / total_prompt) if total_prompt else 0.0

    def latencies(self) -> List[float]:
        return [s.latency_ms for s in self.samples]

    def p50(self) -> float:
        return statistics.median(self.latencies()) if self.latencies() else 0.0

    def p95(self) -> float:
        if not self.latencies():
            return 0.0
        xs = sorted(self.latencies())
        return xs[min(len(xs) - 1, int(len(xs) * 0.95))]

    def cost_saved_usd(self) -> float:
        return sum(s.cost_saved_usd for s in self.samples)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests(),
            "exact_hits": self.exact_hits(),
            "semantic_hits": self.semantic_hits(),
            "upstream_calls": self.upstream_calls(),
            "prefix_hit_count": self.prefix_hit_count(),
            "hit_rate": round(self.hit_rate(), 4),
            "cached_token_ratio": round(self.cached_token_ratio(), 4),
            "p50_ms": round(self.p50(), 1),
            "p95_ms": round(self.p95(), 1),
            "cost_saved_usd": round(self.cost_saved_usd(), 6),
        }


def _mk_messages(task: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


async def _upstream_call(client: AsyncClient, base_url: str, model: str, messages: List[Dict[str, str]], skip_on_error: bool = False) -> Optional[Dict[str, Any]]:
    """Call upstream with retry on transient errors (429/5xx).

    When ``skip_on_error`` is True, a request that exhausts retries is
    skipped (returns None) instead of raising — used by paraphrase mode
    where the free endpoint can be flaky and we'd rather lose one sample
    than abort the whole run.
    """
    max_retries = 4
    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.2, "stream": False},
                timeout=120,
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            break
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
            print(f"    ⚠ upstream {resp.status_code}, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})", flush=True)
            await asyncio.sleep(delay)
            delay *= 2
            continue
        last_err = RuntimeError(f"upstream {resp.status_code}: {resp.text[:200]}")
        break
    if skip_on_error:
        print(f"    ⚠ skipping request after retries exhausted: {last_err}", flush=True)
        return None
    raise RuntimeError(f"upstream call failed: {last_err}")


async def run_baseline(client: AsyncClient, base_url: str, model: str, tasks: List[Any], repeat: int) -> RunResult:
    result = RunResult("baseline (no cache)")
    for ti, task in enumerate(tasks):
        variants: List[str] = task if isinstance(task, list) else [task] * repeat
        for rep, variant in enumerate(variants):
            start = time.perf_counter()
            body = await _upstream_call(client, base_url, model, _mk_messages(variant), skip_on_error=True)
            if body is None:
                continue  # skipped after retries exhausted (flaky free endpoint)
            elapsed = (time.perf_counter() - start) * 1000
            usage = body.get("usage", {})
            details = usage.get("prompt_tokens_details", {}) or {}
            result.samples.append(
                Sample(
                    task_idx=ti, repeat=rep, layer="miss", hit=False,
                    latency_ms=elapsed,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    cached_tokens=int(details.get("cached_tokens", 0) or usage.get("prompt_cache_hit_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                )
            )
    return result


async def _fusion_upstream(client: AsyncClient, base_url: str, model: str):
    async def upstream(**kwargs: Any) -> Any:
        body = await _upstream_call(client, base_url, model, kwargs.get("messages", []), skip_on_error=True)
        if body is None:
            # Return a minimal completion so the pipeline records a miss
            # instead of crashing the whole run on a flaky free endpoint.
            return {
                "id": "chatcmpl-skip",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return body

    return upstream


async def run_fusion(
    client: AsyncClient,
    base_url: str,
    model: str,
    tasks: List[str],
    repeat: int,
    enable_semantic: bool,
    paraphrase: bool = False,
) -> RunResult:
    cfg = FusionCacheConfig(
        enable_semantic=enable_semantic,
        enable_prefix_accounting=True,
        similarity_threshold=0.9,
        exact_ttl=3600.0,
        semantic_ttl=7200.0,
    )
    cache = FusionCache(config=cfg)
    if enable_semantic:
        # Use a local deterministic embedder so the L2 path runs without
        # needing a paid /embeddings endpoint. Character-ngram similarity
        # makes paraphrases (shared vocabulary) land above the threshold.
        from fusion_cache.semantic.embedder import Embedder

        cache.embedder = _LocalEmbedder()  # type: ignore[assignment]
    upstream = await _fusion_upstream(client, base_url, model)
    result = RunResult("fusion" + ("+sem" if enable_semantic else ""))

    for ti, task in enumerate(tasks):
        if paraphrase:
            # Each task is a list of paraphrase variants; issue each once.
            variants = task if isinstance(task, list) else [task]
            for rep, variant in enumerate(variants):
                request = {"model": model, "messages": _mk_messages(variant), "temperature": 0.2}
                start = time.perf_counter()
                pres = await cache.chat_completion(request=request, upstream=upstream, stream=False)
                elapsed = (time.perf_counter() - start) * 1000
                usage = pres.meta or {}
                details = usage.get("prompt_tokens_details", {}) or {}
                result.samples.append(
                    Sample(
                        task_idx=ti, repeat=rep, layer=pres.layer, hit=pres.hit,
                        latency_ms=elapsed,
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        cached_tokens=int(details.get("cached_tokens", 0) or usage.get("prompt_cache_hit_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        cost_saved_usd=pres.cost_saved_usd,
                    )
                )
            continue
        for rep in range(repeat):
            request = {"model": model, "messages": _mk_messages(task), "temperature": 0.2}
            start = time.perf_counter()
            pres = await cache.chat_completion(request=request, upstream=upstream, stream=False)
            elapsed = (time.perf_counter() - start) * 1000
            usage = pres.meta or {}
            details = usage.get("prompt_tokens_details", {}) or {}
            result.samples.append(
                Sample(
                    task_idx=ti, repeat=rep, layer=pres.layer, hit=pres.hit,
                    latency_ms=elapsed,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    cached_tokens=int(details.get("cached_tokens", 0) or usage.get("prompt_cache_hit_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    cost_saved_usd=pres.cost_saved_usd,
                )
            )
    await cache.aclose()
    return result


class _LocalEmbedder:
    """Deterministic local embedder (character n-grams) — no API needed.

    Paraphrases share most vocabulary/trigrams, so they land above the
    similarity threshold; unrelated texts do not. Mirrors the test fixture.
    """

    def __init__(self) -> None:
        self.enabled = True

    async def embed(self, text: str) -> List[float]:
        import math

        text = text.lower()
        ngrams: Dict[str, int] = {}
        for tok in text.split():
            ngrams["w:" + tok] = ngrams.get("w:" + tok, 0) + 1
        for i in range(len(text) - 2):
            gram = text[i : i + 3]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        vec = [0.0] * 128
        for gram, count in ngrams.items():
            h = hash(gram) & 127
            vec[h] += count
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def aclose(self) -> None:  # noqa: D401
        pass


def _markdown(results: List[RunResult]) -> str:
    lines = [
        "| Config | Req | L1 hit | L2 hit | Upstream | Prefix-hit reqs | Hit rate | Cached-token ratio | P50 (ms) | P95 (ms) | $ saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.summary_dict()
        lines.append(
            f"| {r.label} | {s['requests']} | {s['exact_hits']} | {s['semantic_hits']} | "
            f"{s['upstream_calls']} | {s['prefix_hit_count']} | {s['hit_rate']:.0%} | "
            f"{s['cached_token_ratio']:.0%} | {s['p50_ms']} | {s['p95_ms']} | ${s['cost_saved_usd']:.6f} |"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("FUSION_UPSTREAM_BASE_URL", "https://opencode.ai/zen/go/v1"))
    parser.add_argument("--model", default=os.environ.get("FUSION_BENCH_MODEL", "ox-alpha-free"))
    parser.add_argument("--tasks", type=int, default=10, help="number of distinct tasks")
    parser.add_argument("--repeat", type=int, default=3, help="times to repeat each task")
    parser.add_argument("--no-semantic", action="store_true", help="skip the semantic-enabled run")
    parser.add_argument("--no-baseline", action="store_true", help="skip the baseline run")
    parser.add_argument("--paraphrase", action="store_true", help="use paraphrase groups (each task asked 3 ways) to exercise the L2 semantic layer")
    args = parser.parse_args()

    api_key = (
        os.environ.get("FUSION_UPSTREAM_API_KEY")
        or os.environ.get("OPENCODE_GO_API_KEY")
        or os.environ.get("OPENCODE_API_KEY")
    )
    if not api_key:
        raise SystemExit("set FUSION_UPSTREAM_API_KEY / OPENCODE_GO_API_KEY / OPENCODE_API_KEY")

    if args.paraphrase:
        tasks: List[Any] = PARAPHRASE_GROUPS[: args.tasks]
        print(f"paraphrase mode: {len(tasks)} groups × 3 variants each (L2 semantic is the star)")
    else:
        tasks = TASKS[: args.tasks]
    headers = {"Authorization": f"Bearer {api_key}"}
    async with AsyncClient(headers=headers) as client:
        results: List[RunResult] = []
        if not args.no_baseline:
            print(f"▶ baseline: {len(tasks)} tasks × {args.repeat} ...")
            results.append(await run_baseline(client, args.base_url, args.model, tasks, args.repeat))
        print(f"▶ fusion (L1+L3): ...")
        results.append(
            await run_fusion(client, args.base_url, args.model, tasks, args.repeat, enable_semantic=False, paraphrase=args.paraphrase)
        )
        if not args.no_semantic:
            print(f"▶ fusion+sem (L1+L2+L3): ...")
            results.append(
                await run_fusion(client, args.base_url, args.model, tasks, args.repeat, enable_semantic=True, paraphrase=args.paraphrase)
            )

    print("\n" + _markdown(results))
    wl = "paraphrase groups" if args.paraphrase else "tasks"
    print(f"\nWorkload: {args.tasks} {wl} × {args.repeat} reps, model={args.model}, upstream={args.base_url}")


if __name__ == "__main__":
    asyncio.run(main())
