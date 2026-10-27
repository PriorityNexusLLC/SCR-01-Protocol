"""
gravity_model.py — the actual "Emotional Gravity" you described: the more
emotional weight a memory carries, the more MEANING/pull it has, making
it a more trustworthy anchor to trust and route toward. Distinct from
emotional_friction.py (resistance/heaviness that slows things down).

PLAIN-LANGUAGE EXPLANATION:

  This uses a real, named technique from network science and
  transportation/trade research: the "gravity model." The attraction
  between two things is proportional to their MASS, and falls off with
  DISTANCE -- literally borrowed from Newton's law of gravitation
  (force = mass_a * mass_b / distance^2).

  Here:
    - MASS of a memory = 1 (baseline, just for existing) + its salience
      (emotional weight -- see memory.py, computed by emotional_friction.py
      at the time it was remembered).
    - DISTANCE between the current text and a memory = the real geodesic
      path distance from geodesics.py -- how directly connected they are
      in the associative network, accounting for detours.

  A memory that is BOTH emotionally weighted AND semantically close pulls
  the hardest -- it's the most "trustworthy" pathway to anchor around,
  in exactly the sense you described: which vectors/pathways deserve
  more trust, driven by how much meaning (mass) they carry.

HONESTY NOTE: like emotional_friction.py, this doesn't predict anything
on its own -- it ranks candidate memories by a real, computable score.
Whether "highest gravity" actually correlates with "most useful to act
on" in practice is exactly the kind of thing worth testing against real
logged sessions, not assumed from the formula alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geodesics import ConceptGraph
from .memory import Memory, MemoryStore

BASE_MASS = 1.0  # every memory has at least this much "mass" just by existing at all


@dataclass
class GravityResult:
    memory: Memory
    distance: float   # geodesic path distance between the query and this memory
    mass: float         # 1 + salience
    gravity: float       # mass / distance^2 -- higher = more pull, more "trustworthy"


def gravity_rank(
    memory_store: MemoryStore,
    query_text: str,
    limit: int = 5,
    k_neighbors: int = 3,
    max_memories: int = 200,
) -> list[GravityResult]:
    """
    Rank stored memories by gravitational pull toward the current query --
    NOT by plain word-overlap (that's what recall() does), but by how much
    each memory's emotional mass and semantic closeness combine to make it
    a trustworthy anchor for the current context.

    Memories unreachable through the associative graph (no path at all)
    exert zero gravity and are excluded, rather than being scored as
    infinitely close or forced to a default.
    """
    memories = memory_store.recent(max_memories)
    if not memories:
        return []

    nodes = [m.content for m in memories] + [query_text]
    graph = ConceptGraph(k_neighbors=k_neighbors)
    graph.build(nodes)

    results = []
    for m in memories:
        path = graph.path(query_text, m.content)
        if path.path_distance == float("inf"):
            continue  # no route at all -> zero gravity, not "infinitely far"

        distance = max(path.path_distance, 1e-6)  # avoid divide-by-zero on identical text
        mass = BASE_MASS + m.salience
        gravity = mass / (distance ** 2)
        results.append(GravityResult(memory=m, distance=distance, mass=mass, gravity=gravity))

    results.sort(key=lambda r: r.gravity, reverse=True)
    return results[:limit]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "mem.jsonl")

        # Two memories with similar semantic closeness to the query below,
        # but very different emotional weight -- the salient one should win.
        store.remember("The project meeting went fine, nothing notable happened.")  # low salience
        store.remember("The project meeting was a disaster, everything fell apart.")  # high salience
        store.remember("Completely unrelated note about gardening tips.")  # should barely matter

        query = "How did the project meeting go?"
        ranked = gravity_rank(store, query)

        print(f"Query: {query!r}\n")
        for r in ranked:
            print(f"  gravity={r.gravity:.3f}  mass={r.mass:.2f}  distance={r.distance:.2f}  "
                  f"-> {r.memory.content!r}")
