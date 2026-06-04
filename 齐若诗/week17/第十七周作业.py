""
EmbeddingsCache.py
------------------
Cache embedding vectors in Redis to avoid redundant API calls.
Uses RedisVL's SemanticCache under the hood but focused on raw embedding storage.
"""

import hashlib
import json
import os
import numpy as np
import redis
from typing import List, Optional


class EmbeddingsCache:
    """
    A Redis-backed cache for embedding vectors.

    Stores embeddings keyed by a hash of the input text so that identical
    inputs skip the embedding model entirely.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "emb_cache",
        ttl: Optional[int] = None,
    ):
        self.client = redis.from_url(redis_url, decode_responses=False)
        self.prefix = prefix
        self.ttl = ttl  # seconds; None means no expiry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.prefix}:{digest}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[List[float]]:
        """Return cached embedding or None if not found."""
        raw = self.client.get(self._key(text))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding. Respects TTL if configured."""
        key = self._key(text)
        payload = json.dumps(embedding)
        if self.ttl:
            self.client.setex(key, self.ttl, payload)
        else:
            self.client.set(key, payload)

    def get_or_embed(self, text: str, embed_fn) -> List[float]:
        """
        Return cached embedding, or call embed_fn(text) → List[float],
        store the result, and return it.

        Parameters
        ----------
        text     : input string
        embed_fn : callable that accepts a string and returns List[float]
        """
        cached = self.get(text)
        if cached is not None:
            print(f"[EmbeddingsCache] cache hit  → {text[:60]!r}")
            return cached
        print(f"[EmbeddingsCache] cache miss → calling embed_fn")
        embedding = embed_fn(text)
        self.set(text, embedding)
        return embedding

    def batch_get_or_embed(self, texts: List[str], embed_fn) -> List[List[float]]:
        """
        Batch version: hit cache for every text, call embed_fn only for misses.
        embed_fn receives a List[str] of cache-miss texts.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices, miss_texts = [], []

        for i, t in enumerate(texts):
            cached = self.get(t)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(t)

        if miss_texts:
            new_embeddings = embed_fn(miss_texts)
            for idx, text, emb in zip(miss_indices, miss_texts, new_embeddings):
                self.set(text, emb)
                results[idx] = emb

        print(
            f"[EmbeddingsCache] batch: {len(texts)-len(miss_texts)} hits, "
            f"{len(miss_texts)} misses"
        )
        return results  # type: ignore[return-value]

    def delete(self, text: str) -> bool:
        """Remove a single cached embedding. Returns True if key existed."""
        return bool(self.client.delete(self._key(text)))

    def flush(self) -> int:
        """Delete all keys under this cache's prefix. Returns count deleted."""
        pattern = f"{self.prefix}:*"
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0

    def stats(self) -> dict:
        keys = self.client.keys(f"{self.prefix}:*")
        return {"prefix": self.prefix, "count": len(keys), "ttl": self.ttl}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Fake embed function — replace with openai / sentence-transformers etc.
    call_count = 0

    def fake_embed(text_or_texts):
        global call_count
        if isinstance(text_or_texts, str):
            call_count += 1
            return list(np.random.rand(4).astype(float))
        else:
            call_count += len(text_or_texts)
            return [list(np.random.rand(4).astype(float)) for _ in text_or_texts]

    cache = EmbeddingsCache(ttl=60)
    cache.flush()

    texts = ["Hello world", "Redis is fast", "Hello world"]  # duplicate on purpose

    print("=== Single get_or_embed ===")
    for t in texts:
        emb = cache.get_or_embed(t, lambda x: fake_embed(x))
        print(f"  {t!r}: {emb}")

    print(f"\nEmbed calls so far: {call_count}  (expected 2, not 3)")

    print("\n=== Batch get_or_embed ===")
    call_count = 0
    cache.flush()
    embs = cache.batch_get_or_embed(texts, fake_embed)
    print(f"Embed calls: {call_count}  (expected 2)")

    print("\n=== Stats ===")
    print(cache.stats())
