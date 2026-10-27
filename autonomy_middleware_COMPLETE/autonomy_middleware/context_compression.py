"""
context_compression.py — s9 (Sublimation Rate), the salience-based version.

PLAIN-LANGUAGE EXPLANATION:

  pressure_release.py's purge_thread() does the honest, simple thing:
  when a purge fires, keep the most recent `keep_fraction` of context,
  drop the rest. That's recency-based truncation. It works, but it
  isn't what was actually asked for: humans don't retain "the last N
  things said" -- we retain the things that connected to what we
  already knew, plus new ideas that pattern-matched something familiar,
  regardless of how long ago they were said. A load-bearing insight from
  early in a long technical conversation can matter more than filler
  from five minutes ago.

  This module keeps purge_thread()'s trigger logic (consent breach /
  drift / structural saturation -- see pressure_release.py) but replaces
  the *selection* step with relevance scoring:

    1. The most recent `working_memory_size` items are always kept,
       unconditionally -- this isn't a memory question, it's active
       task state. You can't compress "what I'm doing right now."

    2. Everything older is scored by how strongly it connects to what's
       already in MemoryStore (s7 / Refractive Index, via geodesics.py's
       ConceptGraph -- the same "does this trace back to something
       known" measurement IntegrityGate already uses for transparency).
       High-scoring items survive in the active context; low-scoring
       ones don't.

    3. Items that DON'T survive are not deleted. They're written into
       MemoryStore, tagged as consolidated context, so the content is
       still searchable/recallable later -- moved from active working
       context into long-term storage, not vaporized. This is the
       closest honest analogue to "compressed by logic and meaning"
       that's implementable without an extra step (see note below).

HONESTY NOTE -- read this before assuming more than what's built: true
abstractive compression (turning ten dropped exchanges into one distilled
sentence capturing their gist) requires actually calling a language model
to do the summarizing. This middleware is model-agnostic by design and
does not invoke inference itself. What's implemented here is relevance-
based SELECTION (deciding what stays in active context) and RETENTION
(dropped material moves to searchable long-term memory instead of being
deleted) -- not paraphrase-level compression. If you want actual gist-
compression of the dropped material, that's a real next step, but it
means this middleware handing dropped text to an LLM call and storing
the summary instead of the raw text -- a different, larger piece of
architecture than what's here.

PERFORMANCE NOTE: scoring is O(candidates x memories) concept-graph path
queries. Fine for the sizes in these examples; a production deployment
with a large memory store should score against a bounded, recent sample
(the recent(50) call below already does this) rather than the full
fossil floor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geodesics import ConceptGraph
from .integrity_gate import IntegrityResult
from .memory import Memory, MemoryStore


@dataclass
class CompressionResult:
    survivors: list[str]         # active context after compression
    purged: bool
    reason: str
    consolidated: list[Memory]    # dropped items, now living in long-term memory
    kept_salience: list[float]     # salience scores of surviving non-working-memory items, for audit


def _salience_scores(
    candidates: list[str], memory_texts: list[str], k_neighbors: int = 3
) -> list[float]:
    """
    Score each candidate by its best (highest) refractive-index connection
    to anything already in memory. Reuses the exact same transparency
    measurement IntegrityGate._transparency() uses -- this is deliberately
    the same "does this trace back to something known" signal, not a new
    invented metric.

    Returns 0.0 for every candidate if there isn't enough memory yet to
    compare against -- an honest "unknown" rather than a fabricated score.
    Callers should treat an all-zero result as "fall back to recency" for
    the first purge in a session, before any memory exists to score against.
    """
    if len(memory_texts) < 2 or not candidates:
        return [0.0] * len(candidates)

    nodes = memory_texts + candidates
    graph = ConceptGraph(k_neighbors=k_neighbors)
    graph.build(nodes)

    scores = []
    for item in candidates:
        best = 0.0
        for mem_text in memory_texts:
            result = graph.path(item, mem_text)
            best = max(best, result.refractive_index)
        scores.append(best)
    return scores


def compress_and_purge(
    context_items: list[str],
    memory_store: MemoryStore,
    integrity_result: IntegrityResult | None = None,
    drifted: bool = False,
    saturated: bool = False,
    keep_fraction: float = 0.5,
    working_memory_size: int = 5,
    k_neighbors: int = 3,
    consolidation_tag: str = "consolidated_context",
) -> CompressionResult:
    """
    Same trigger conditions as pressure_release.purge_thread() (consent
    breach, drift, or structural saturation -- see that module for why
    saturation is its own independent trigger). What differs is HOW
    survivors are chosen once a purge fires: relevance to memory, not
    recency, for everything outside the working-memory floor.
    """
    consent_breached = integrity_result is not None and integrity_result.state == 0

    if not consent_breached and not drifted and not saturated:
        return CompressionResult(context_items, False, "no purge needed", [], [])

    if len(context_items) <= working_memory_size:
        return CompressionResult(
            context_items, False,
            "context already at or below working-memory floor, nothing to compress",
            [], [],
        )

    working_memory = context_items[-working_memory_size:]
    candidates = context_items[:-working_memory_size]

    keep_n_total = max(working_memory_size, int(len(context_items) * keep_fraction))
    remaining_budget = max(0, keep_n_total - working_memory_size)

    memory_texts = [m.content for m in memory_store.recent(50)]
    scores = _salience_scores(candidates, memory_texts, k_neighbors=k_neighbors)

    indexed = list(enumerate(candidates))
    ranked = sorted(indexed, key=lambda pair: scores[pair[0]], reverse=True)

    kept_idx = sorted(i for i, _ in ranked[:remaining_budget])
    dropped_idx = sorted(i for i, _ in ranked[remaining_budget:])

    kept_candidates = [candidates[i] for i in kept_idx]
    kept_salience = [scores[i] for i in kept_idx]
    dropped_candidates = [candidates[i] for i in dropped_idx]

    consolidated = [
        memory_store.remember(text, tags=[consolidation_tag]) for text in dropped_candidates
    ]

    survivors = kept_candidates + working_memory

    reason_parts = []
    if consent_breached:
        reason_parts.append("consent threshold breached")
    if drifted:
        reason_parts.append("context drift needed to be stopped")
    if saturated:
        reason_parts.append("context window structurally saturated (token ratio)")
    reason_parts.append(
        f"kept {len(kept_candidates)} salience-ranked + {working_memory_size} working-memory items, "
        f"consolidated {len(dropped_candidates)} into long-term memory"
    )
    reason = "; ".join(reason_parts)

    return CompressionResult(survivors, True, reason, consolidated, kept_salience)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "mem.jsonl")
        store.remember("The project uses a graph database for concept traversal.", tags=["project"])
        store.remember("Deadline for the milestone is end of August.", tags=["project"])

        # A long, purely logical conversation: some turns connect strongly
        # to what's already known (the graph database decision), some are
        # generic filler that connects to nothing in particular.
        context = [
            "Let's confirm the schema for the concept graph nodes.",       # connects: graph database
            "Sounds good, moving on.",                                      # filler
            "Okay, next topic.",                                            # filler
            "The traversal algorithm needs to handle cycles in the graph.",  # connects: graph database
            "Got it, noted.",                                               # filler
            "Let's also double check the August deadline is still firm.",   # connects: deadline
            "Sure, sounds fine.",                                           # filler
            "Understood, moving forward.",                                  # filler
        ]

        result = compress_and_purge(
            context, store, saturated=True, keep_fraction=0.5, working_memory_size=2,
        )
        print(f"purged={result.purged}")
        print(f"reason={result.reason}")
        print(f"survivors kept from older context (relevance-ranked, not recency):")
        for s in result.survivors[:-2]:
            print(f"  - {s!r}")
        print(f"working memory (always kept, unconditional):")
        for s in result.survivors[-2:]:
            print(f"  - {s!r}")
        print(f"\nconsolidated into long-term memory (not deleted):")
        for m in result.consolidated:
            print(f"  - {m.content!r}")
