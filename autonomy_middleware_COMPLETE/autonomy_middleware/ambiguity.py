"""
ambiguity.py — s11: Ambiguity Density, reframed positively as the raw
material for broader pattern recognition, not just risk.

PLAIN-LANGUAGE EXPLANATION:

  When a request matches several stored memories or concepts about
  equally well, that's ambiguity -- but it's also what gives a system
  multiple valid interpretive threads to draw on, instead of one narrow
  match.

  Measured with real, standard math: Shannon entropy over the relevance
  scores of the top matches.
    - All the weight on one dominant match -> low entropy -> narrow,
      unambiguous, one clear interpretation.
    - Weight spread evenly across several matches -> high entropy ->
      high ambiguity density -> several roughly-equal candidate
      interpretations available -- broader pattern recognition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .memory import MemoryStore, tokenize


@dataclass
class AmbiguityResult:
    density: float                          # 0 (one dominant match) .. 1 (many equally plausible)
    num_candidates: int
    top_candidates: list[tuple[str, float]]  # (text, relevance_score), sorted by score descending


def ambiguity_density(candidates: list[tuple[str, float]]) -> AmbiguityResult:
    """candidates: list of (text, relevance_score) pairs; scores should be > 0."""
    scored = [(text, score) for text, score in candidates if score > 0]
    if len(scored) <= 1:
        return AmbiguityResult(density=0.0, num_candidates=len(scored), top_candidates=scored)

    total = sum(score for _, score in scored)
    probs = [score / total for _, score in scored]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(scored))
    density = entropy / max_entropy if max_entropy > 0 else 0.0

    return AmbiguityResult(
        density=density,
        num_candidates=len(scored),
        top_candidates=sorted(scored, key=lambda pair: pair[1], reverse=True),
    )


def ambiguity_from_memory(memory_store: MemoryStore, query: str, limit: int = 5) -> AmbiguityResult:
    """Convenience wrapper: build candidates directly from word-overlap scores against memory."""
    query_words = tokenize(query)
    candidates = []
    for m in memory_store.recent(200):
        overlap = len(query_words & tokenize(m.content))
        if overlap > 0:
            candidates.append((m.content, float(overlap)))
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return ambiguity_density(candidates[:limit])


if __name__ == "__main__":
    print("One dominant match (narrow, unambiguous):")
    r1 = ambiguity_density([("exact match", 10.0), ("weak match", 1.0), ("weaker match", 0.5)])
    print(f"  density={r1.density:.3f}")

    print("\nSeveral equally strong matches (high ambiguity, broad pattern recognition):")
    r2 = ambiguity_density([("match A", 5.0), ("match B", 5.0), ("match C", 5.0), ("match D", 5.0)])
    print(f"  density={r2.density:.3f}")
