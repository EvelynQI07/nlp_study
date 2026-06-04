"""
SemanticMessageHistory.py
-------------------------
Conversation history stored in Redis with semantic search capability.

Extends the idea of a simple message log: you can retrieve not only the
chronological recent context but also the most semantically relevant past
messages for a given query — useful for long-running sessions.
"""

import time
from typing import Any, Dict, List, Optional

from redisvl.extensions.session_manager import SemanticSessionManager


class SemanticMessageHistory:
    """
    Manages multi-turn conversation history in Redis with semantic search.

    Each session has an isolated key-space.  Messages are stored with their
    embeddings so you can later retrieve the top-k semantically similar turns.
    """

    def __init__(
        self,
        session_id: str,
        name: str = "chat_history",
        distance_threshold: float = 0.30,
        top_k: int = 5,
        redis_url: str = "redis://localhost:6379",
    ):
        self.session_id = session_id
        self.top_k = top_k

        self._mgr = SemanticSessionManager(
            name=name,
            session_id=session_id,
            distance_threshold=distance_threshold,
            redis_url=redis_url,
        )

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Append a user turn to the session."""
        self._mgr.add_messages([
            {"role": "user", "content": content, **(metadata or {})}
        ])

    def add_ai_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Append an assistant turn to the session."""
        self._mgr.add_messages([
            {"role": "assistant", "content": content, **(metadata or {})}
        ])

    def add_exchange(self, user_msg: str, ai_msg: str) -> None:
        """Convenience: add a user+assistant pair atomically."""
        self._mgr.add_messages([
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": ai_msg},
        ])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 10) -> List[Dict[str, str]]:
        """Return the last n messages in chronological order."""
        return self._mgr.get_recent(top_k=n)

    def get_relevant(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """Return the top-k semantically relevant messages for a query."""
        return self._mgr.get_relevant(query, top_k=top_k or self.top_k)

    def get_context(
        self,
        query: str,
        recent_n: int = 4,
        relevant_k: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Combine recent + semantically relevant messages, deduplicated,
        for passing as context to an LLM call.
        """
        recent   = self.get_recent(recent_n)
        relevant = self.get_relevant(query, relevant_k)

        # Deduplicate by content, preserving order (relevant first, then recent)
        seen: set = set()
        combined: List[Dict] = []
        for msg in relevant + recent:
            key = (msg.get("role"), msg.get("content"))
            if key not in seen:
                seen.add(key)
                combined.append(msg)

        return combined

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all messages in this session."""
        self._mgr.clear()

    @property
    def message_count(self) -> int:
        return len(self._mgr.get_recent(top_k=10_000))

    def __repr__(self) -> str:
        return (
            f"SemanticMessageHistory("
            f"session_id={self.session_id!r}, "
            f"messages={self.message_count})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hist = SemanticMessageHistory(session_id="demo-session-001", top_k=3)
    hist.clear()

    exchanges = [
        ("What is Redis?",       "Redis is an in-memory data structure store."),
        ("How fast is Redis?",   "Redis can handle millions of requests per second."),
        ("Tell me about Python", "Python is a high-level programming language."),
        ("What is a vector DB?", "A vector database stores and queries embeddings."),
        ("Can Redis store vectors?", "Yes! Redis supports vector similarity search via RediSearch."),
    ]

    for user, ai in exchanges:
        hist.add_exchange(user, ai)
        print(f"  stored: {user!r}")

    print(f"\n{hist}\n")

    query = "Is Redis good for ML?"
    print(f"Query: {query!r}")

    print("\n--- Recent (last 3) ---")
    for m in hist.get_recent(3):
        print(f"  [{m['role']:9}] {m['content']}")

    print("\n--- Relevant ---")
    for m in hist.get_relevant(query):
        print(f"  [{m['role']:9}] {m['content']}")

    print("\n--- Combined context ---")
    for m in hist.get_context(query, recent_n=2):
        print(f"  [{m['role']:9}] {m['content']}")
