"""fusion-cache: a fusion caching layer for LLM APIs.

The package sits between your application and an OpenAI-compatible upstream
(OpenAI, DeepSeek, etc.) and serves cached responses across three layers:

- **L1 exact**: canonicalized request → identical-response cache (memory / redis).
- **L2 semantic**: embedding-based similarity search with a cosine threshold
  and a false-positive guardrail.
- **L3 prefix accounting**: DeepSeek-style automatic prefix caching is passed
  through and its ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
  usage fields are converted into an honest estimate of money saved.

Typical usage (drop-in wrapper)::

    import openai
    from fusion_cache import FusionCache
    from fusion_cache.wrapper.openai import CachedOpenAI

    cache = FusionCache()
    client = CachedOpenAI(openai.AsyncOpenAI(api_key=...), cache=cache)
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hello"}],
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import FusionCacheConfig
from .core.pipeline import FusionCache, FusionCacheStats

__all__ = ["FusionCache", "FusionCacheConfig", "FusionCacheStats", "__version__"]
