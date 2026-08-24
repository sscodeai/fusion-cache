"""Minimal end-to-end demo: wrap an OpenAI client with fusion-cache and
show the before/after on repeated requests.

Run (no real API key needed — uses a fake upstream):

    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fusion_cache import FusionCache
from fusion_cache.core.pipeline import PipelineResult


class FakeCompletions:
    """Stand-in for openai.chat.completions that 'thinks' for 1 second."""

    def create(self, **kwargs: Any) -> Dict[str, Any]:
        time.sleep(1.0)  # pretend upstream is slow
        messages = kwargs.get("messages", [])
        last = messages[-1]["content"] if messages else ""
        return {
            "id": "fake",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"reply to: {last}"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }


async def main() -> None:
    cache = FusionCache()

    async def upstream(**kwargs: Any) -> Any:
        # Call the fake 'upstream' (1s sleep) in a thread so the async
        # pipeline doesn't block.
        return await asyncio.to_thread(FakeCompletions().create, **kwargs)

    request = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Explain caching in one sentence"}],
    }

    print("=== fusion-cache quickstart ===\n")

    # Cold call → upstream (1s)
    t0 = time.perf_counter()
    r1: PipelineResult = await cache.chat_completion(request=request, upstream=upstream, stream=False)
    t1 = time.perf_counter()
    print(f"[1] cold  → layer={r1.layer:<8} {t1 - t0:.2f}s  (upstream, slow)")

    # Identical call → L1 exact hit (instant)
    t0 = time.perf_counter()
    r2: PipelineResult = await cache.chat_completion(request=request, upstream=upstream, stream=False)
    t1 = time.perf_counter()
    print(f"[2] exact → layer={r2.layer:<8} {(t1 - t0) * 1000:.1f}ms  (cached, instant)")

    # Slightly rephrased → L2 semantic hit (needs an embedder; skipped if none)
    paraphrased = dict(request)
    paraphrased["messages"] = [{"role": "user", "content": "In one sentence, what is caching?"}]
    t0 = time.perf_counter()
    r3: PipelineResult = await cache.chat_completion(request=paraphrased, upstream=upstream, stream=False)
    t1 = time.perf_counter()
    print(f"[3] para  → layer={r3.layer:<8} {(t1 - t0) * 1000:.1f}ms  (semantic if embedder configured)")

    print("\n=== stats ===")
    stats = cache.stats_dict()
    print(f"  requests:      {stats['requests']}")
    print(f"  exact hits:    {stats['exact_hits']}")
    print(f"  semantic hits: {stats['semantic_hits']}")
    print(f"  hit rate:      {stats['hit_rate']:.0%}")
    print(f"  cost saved:    ${stats['cost_saved_usd']:.6f}")
    print("\nThe [2] exact call skipped the 1s upstream — that's the point.")


if __name__ == "__main__":
    asyncio.run(main())
