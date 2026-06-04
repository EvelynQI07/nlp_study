"""
SemanticCache.py
----------------
LLM response cache backed by Redis vector search.

Instead of exact-match caching, a query is considered a cache hit when
a semantically similar past query exists above a configurable threshold.
This dramatically increases hit-rate for paraphrased or near-duplicate prompts.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from redisvl.extensions.llmcache import SemanticCache as _RVLSemanticCache


class SemanticCache:
    """
    Thin wrapper around redisvl.extensions.llmcache.SemanticCache that adds:
      - hit/miss counters
      - optional TTL
      - a convenient ask() helper that handles the full cache-then-call flow
    """

    def __init__(
        self,
        name: str = "semantic_cache",
        distance_threshold: float = 0.10,   # lower = stricter match
        ttl: Optional[int] = 3600,
        redis_url: str = "redis://localhost:6379",
    ):
        self._cache = _RVLSemanticCache(
            name=name,
            distance_threshold=distance_threshold,
            ttl=ttl,
            redis_url=redis_url,
        )
        self.distance_threshold = distance_threshold
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Core cache operations
    # ------------------------------------------------------------------

    def check(self, prompt: str) -> Optional[str]:
        """
        Return a cached response string if a semantically similar prompt exists,
        otherwise return None.
        """
        results = self._cache.check(prompt=prompt, num_results=1)
        if results:
            self._hits += 1
            return results[0].get("response")
        self._misses += 1
        return None

    def store(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> str:
        """Store a prompt→response pair. Returns the Redis key."""
        return self._cache.store(
            prompt=prompt,
            response=response,
            metadata=metadata or {},
        )

    def ask(
        self,
        prompt: str,
        llm_fn: Callable[[str], str],
        metadata: Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        High-level helper:
          1. Check semantic cache.
          2. On hit  → return cached response, was_cached=True.
          3. On miss → call llm_fn(prompt), store result, return it.

        Returns (response_str, was_cached).
        """
        cached = self.check(prompt)
        if cached is not None:
            print(f"[SemanticCache] HIT  → {prompt[:60]!r}")
            return cached, True

        print(f"[SemanticCache] MISS → calling LLM")
        t0 = time.time()
        response = llm_fn(prompt)
        elapsed = time.time() - t0
        self.store(prompt, response, metadata)
        print(f"[SemanticCache] stored ({elapsed:.2f}s)")
        return response, False

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "distance_threshold": self.distance_threshold,
        }


# ---------------------------------------------------------------------------
# Demo  (requires a running Redis + OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import openai

    openai_client = openai.OpenAI()

    def call_gpt(prompt: str) -> str:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    sc = SemanticCache(distance_threshold=0.15, ttl=300)
    sc.clear()

    prompts = [
        "What is the capital of France?",
        "Tell me the capital city of France.",   # semantically very close → should hit
        "What is Redis used for?",
    ]

    for p in prompts:
        answer, from_cache = sc.ask(p, call_gpt)
        print(f"  Q: {p}")
        print(f"  A: {answer[:120]}")
        print(f"  cached={from_cache}\n")

    print("Stats:", sc.stats)
