"""
memory.py  —  the "Fossil Floor" piece: a place for the AI to remember things.

PLAIN-LANGUAGE EXPLANATION (read this first):

  Every time something worth remembering happens, we write it down to a
  file on disk — one line per memory, never overwritten, never deleted.
  That's the "fossil" part: once it's written, it stays, layer on layer,
  like sediment.

  Later, when the AI needs to recall something, it searches back through
  everything it has ever written down and pulls out the entries that seem
  relevant to what's happening right now.

  That's it. No equations, no physics. Just "write it down" and "look it
  up later." This is the same basic idea real AI systems use — it's
  usually called a "memory store" or, in more technical writing,
  "retrieval-augmented memory."

WHAT THIS FILE GIVES YOU (three simple actions):

  1. remember(content, tags)   -> write a new memory
  2. recall(query, tags)        -> search past memories for relevant ones
  3. recent(n)                  -> just get the last n memories, no search

You do not need to understand the code below to use this. You only need
these three actions.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Shared word-matching helpers, used by both this module's recall() and
# ambiguity.py's ambiguity_from_memory() -- kept in one place so both stay
# consistent. Found necessary via a real false positive during testing:
# a query and a memory sharing only common function words (like "the")
# were being treated as related, and separately, real content words like
# "files." and "files" were failing to match due to trailing punctuation.
STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "of", "and", "or",
    "that", "this", "it", "you", "i", "me", "my", "we", "us", "for",
    "with", "as", "be", "are", "was", "were", "by", "from", "so", "but",
    "if", "then", "than", "not", "do", "does", "did", "will", "would",
    "can", "could", "should", "there", "here", "what", "who", "how",
}


def tokenize(text: str) -> set[str]:
    """Extract meaningful word tokens: lowercase, punctuation-stripped, stopwords removed."""
    return set(re.findall(r"[a-z0-9']+", text.lower())) - STOPWORDS


@dataclass
class Memory:
    id: str
    timestamp: float
    content: str
    tags: list[str]
    salience: float = 0.0   # t3: how emotionally weighted this memory is (see remember())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "timestamp": self.timestamp, "content": self.content,
            "tags": self.tags, "salience": self.salience,
        }

    @staticmethod
    def from_dict(d: dict) -> "Memory":
        return Memory(
            id=d["id"], timestamp=d["timestamp"], content=d["content"],
            tags=d.get("tags", []), salience=d.get("salience", 0.0),
        )


@dataclass
class MemoryStore:
    path: str | Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.touch(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. REMEMBER — write something down, permanently.
    # ------------------------------------------------------------------
    def remember(self, content: str, tags: list[str] | None = None, salience: float | None = None) -> Memory:
        """
        Write a new memory to the fossil floor.

        content: whatever you want the AI to remember, in plain text.
        tags:    optional short labels to help find it again later,
                 e.g. ["project-x", "decision"] or ["user-preference"].
        salience: t3 (Decay Rate) -- how emotionally weighted this memory is.
                  If not given, it's computed automatically from the content
                  using emotional_friction (see emotional_friction.py): strongly
                  positive or negative content gets a higher salience score,
                  which makes it survive decay() / prune() longer than
                  neutral content. Pass salience=0.0 explicitly to opt out.
        """
        if salience is None:
            from .emotional_friction import emotional_friction
            result = emotional_friction(content)
            salience = abs(result["mean_friction"])  # magnitude only -- strong feeling either direction matters

        mem = Memory(
            id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            content=content,
            tags=tags or [],
            salience=salience,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(mem.to_dict()) + "\n")
        return mem

    # ------------------------------------------------------------------
    # Internal helper: load every memory ever written.
    # ------------------------------------------------------------------
    def _all(self) -> list[Memory]:
        memories = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(Memory.from_dict(json.loads(line)))
        return memories

    # ------------------------------------------------------------------
    # 2. RECALL — search past memories for ones that seem relevant now.
    # ------------------------------------------------------------------
    def recall(self, query: str = "", tags: list[str] | None = None, limit: int = 10) -> list[Memory]:
        """
        Search everything ever remembered.

        query: plain text — memories that share words with this get ranked higher.
        tags:  if given, only memories carrying at least one of these tags are considered.
        limit: max number of memories to return, most relevant first.

        This uses simple word-overlap matching, not a language model — it's
        transparent and you can predict exactly why something matched.
        Common function words (the, a, is, and...) don't count toward a
        match on their own -- only meaningful shared words do.
        """
        candidates = self._all()

        if tags:
            candidates = [m for m in candidates if set(m.tags) & set(tags)]

        if not query.strip():
            # No search text given -> just return the most recent matching memories.
            return sorted(candidates, key=lambda m: m.timestamp, reverse=True)[:limit]

        query_words = tokenize(query)

        def overlap_count(m: Memory) -> int:
            return len(query_words & tokenize(m.content))

        # Only keep memories that actually share at least one word with the query —
        # this is the fix: a memory with zero relevant words should never be returned.
        matches = [(overlap_count(m), m) for m in candidates]
        matches = [pair for pair in matches if pair[0] > 0]

        # Among the real matches, higher word-overlap wins; ties broken by recency.
        matches.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        return [m for _, m in matches[:limit]]

    # ------------------------------------------------------------------
    # 3. RECENT — just the last n memories, no searching involved.
    # ------------------------------------------------------------------
    def recent(self, n: int = 10) -> list[Memory]:
        return sorted(self._all(), key=lambda m: m.timestamp, reverse=True)[:n]

    def count(self) -> int:
        return len(self._all())

    # ------------------------------------------------------------------
    # t3: DECAY — emotionally salient memories survive longer than neutral ones.
    # ------------------------------------------------------------------
    def _decay_score(self, m: Memory, now: float, decay_resistance: float = 5.0) -> float:
        """
        Lower score = survives longer. Age is discounted by salience: a highly
        salient memory's effective age is divided down, so it decays slower
        than a neutral memory of the same actual age. decay_resistance controls
        how much salience matters (higher = salience protects more).
        """
        age = max(0.0, now - m.timestamp)
        return age / (1.0 + m.salience * decay_resistance)

    def prune(self, keep_fraction: float = 0.5, now: float | None = None) -> tuple[list[Memory], list[Memory]]:
        """
        t3 in action: reduce the store down to keep_fraction of its memories,
        keeping the ones with the lowest decay score (least "decayed" —
        combination of being recent AND/OR emotionally salient), and actually
        rewriting the file to drop the rest.

        Returns (kept, dropped) so the caller can see exactly what survived.
        This does NOT run automatically — call it explicitly when you want
        to simulate forgetting.
        """
        now = now if now is not None else time.time()
        all_memories = self._all()
        if not all_memories:
            return [], []

        keep_n = max(1, int(len(all_memories) * keep_fraction))
        ranked = sorted(all_memories, key=lambda m: self._decay_score(m, now))
        kept, dropped = ranked[:keep_n], ranked[keep_n:]

        with self.path.open("w", encoding="utf-8") as f:
            for m in kept:
                f.write(json.dumps(m.to_dict()) + "\n")

        return kept, dropped

    # ------------------------------------------------------------------
    # t4: TIMED RECALL — measure real retrieval latency (The Fold).
    # ------------------------------------------------------------------
    def recall_timed(self, query: str = "", tags: list[str] | None = None, limit: int = 10) -> tuple[list[Memory], float]:
        """Same as recall(), but also returns real wall-clock retrieval time in seconds."""
        start = time.perf_counter()
        results = self.recall(query=query, tags=tags, limit=limit)
        elapsed = time.perf_counter() - start
        return results, elapsed


if __name__ == "__main__":
    # A tiny walkthrough you can run yourself: python3 -m autonomy_middleware.memory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memories.jsonl")

        store.remember("User prefers concise answers, no filler.", tags=["preference"])
        store.remember("Project deadline is end of August.", tags=["project"])
        store.remember("User's agent should always ask before deleting files.", tags=["preference", "safety"])

        print("All memories tagged 'preference':")
        for m in store.recall(tags=["preference"]):
            print(f"  - {m.content}")

        print("\nSearching for 'deadline':")
        for m in store.recall("deadline"):
            print(f"  - {m.content}")

        print(f"\nTotal memories stored: {store.count()}")
