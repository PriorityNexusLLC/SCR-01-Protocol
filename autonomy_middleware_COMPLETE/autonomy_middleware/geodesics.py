"""
geodesics.py — a concrete implementation of s7 (Refractive Index / "geometry warp").

PLAIN-LANGUAGE EXPLANATION:

  Take a batch of ideas/phrases ("concepts") and represent each one as a
  vector of the words it uses (TF-IDF — literally just weighted word
  counts, nothing hidden or black-box). Connect each concept to its
  nearest neighbors by similarity, forming a small network.

  To go from concept A to concept B there are two distances:
    - STRAIGHT LINE distance: how similar or different A and B actually
      are, measured directly, no detour.
    - PATH distance: the shortest route actually available through the
      network of intermediate, related concepts.

  If A and B are directly similar, the path distance matches the
  straight-line distance — an efficient, direct translation. If A and B
  are only connected through a long chain of loosely related concepts,
  the path distance balloons past the straight-line distance — meaning
  had to detour and bend to get from one to the other. That ratio
  (straight-line ÷ path, capped at 1.0) is the refractive index: 1.0 =
  perfectly direct, closer to 0 = a big detour was required.

  This is a real, named idea in network science — a "detour index" or
  "route factor" — normally used for things like measuring how indirect
  a road network is compared to flying in a straight line. Same math,
  applied to word-overlap similarity between phrases instead of roads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PathResult:
    concept_a: str
    concept_b: str
    straight_line_distance: float   # direct distance between A and B, no detour
    path_distance: float             # shortest distance actually available through the network
    refractive_index: float          # straight_line / path, capped at 1.0 -- 1.0 = direct, near 0 = big detour
    hops: list[str]                  # the actual intermediate concepts the path routed through


class ConceptGraph:
    """
    A small, transparent associative network built from TF-IDF word-overlap
    similarity. Every distance here is directly inspectable -- nothing in
    this class is a downloaded or black-box embedding model.
    """

    def __init__(self, k_neighbors: int = 3, max_distance: float = 0.9):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance  # don't connect nodes with no real similarity, even to fill k
        self.concepts: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._vectors = None
        self._graph: csr_matrix | None = None

    def build(self, concepts: list[str]) -> None:
        """(Re)build the associative graph from a list of concept texts."""
        if len(concepts) < 2:
            raise ValueError("Need at least 2 concepts to build a graph.")

        self.concepts = list(concepts)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectors = self._vectorizer.fit_transform(self.concepts).toarray()

        n = len(self.concepts)
        sim = cosine_similarity(self._vectors)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        k = min(self.k_neighbors, n - 1)
        graph = lil_matrix((n, n))
        for i in range(n):
            order = np.argsort(dist[i])[1:]  # skip 0 = itself, closest-first
            added = 0
            for j in order:
                if added >= k:
                    break
                d = float(dist[i, j])
                if d >= self.max_distance:
                    # no genuinely similar neighbors left -- stop instead of
                    # forcing a meaningless tied-distance connection
                    break
                graph[i, j] = d
                graph[j, i] = d  # undirected
                added += 1
        self._graph = graph.tocsr()

    def _straight_line(self, a: str, b: str) -> float:
        vecs = self._vectorizer.transform([a, b]).toarray()
        sim = cosine_similarity(vecs)[0, 1]
        return float(1.0 - sim)

    def _nearest_node_index(self, text: str) -> int:
        if text in self.concepts:
            return self.concepts.index(text)
        vec = self._vectorizer.transform([text]).toarray()
        sims = cosine_similarity(vec, self._vectors)[0]
        return int(np.argmax(sims))

    def path(self, concept_a: str, concept_b: str) -> PathResult:
        """Shortest path between two concepts, and the resulting refractive index."""
        if self._graph is None:
            raise RuntimeError("Call build(concepts) before computing paths.")

        idx_a = self._nearest_node_index(concept_a)
        idx_b = self._nearest_node_index(concept_b)

        dist_matrix, predecessors = dijkstra(
            self._graph, directed=False, indices=idx_a, return_predecessors=True
        )
        path_distance = float(dist_matrix[idx_b])
        straight = self._straight_line(concept_a, concept_b)

        if np.isinf(path_distance):
            # No route exists at all through the associative graph as built --
            # an honest result, not an error: these concepts aren't connected
            # by anything currently known. Refractive index is undefined (0).
            return PathResult(
                concept_a=concept_a,
                concept_b=concept_b,
                straight_line_distance=straight,
                path_distance=float("inf"),
                refractive_index=0.0,
                hops=[],
            )

        hops = []
        node = idx_b
        while node != idx_a and node != -9999:
            hops.append(self.concepts[node])
            node = predecessors[node]
        hops.append(self.concepts[idx_a])
        hops.reverse()

        if path_distance <= 1e-9:
            refractive_index = 1.0
        else:
            refractive_index = min(1.0, straight / path_distance)

        return PathResult(
            concept_a=concept_a,
            concept_b=concept_b,
            straight_line_distance=straight,
            path_distance=path_distance,
            refractive_index=refractive_index,
            hops=hops,
        )

    def k_paths(self, concept_a: str, concept_b: str, k: int = 3) -> list[list[str]]:
        """
        t5: Simulation Width. Instead of one shortest path, find up to k
        distinct paths between two concepts -- multiple parallel routes
        the associative network could take, ranked shortest-first. Real,
        standard algorithm (Yen's k-shortest loopless paths, via networkx),
        not a new invention. If the graph only supports fewer than k
        distinct routes, returns however many actually exist -- it does
        not fabricate paths that aren't there.
        """
        import networkx as nx

        if self._graph is None:
            raise RuntimeError("Call build(concepts) before computing paths.")

        idx_a = self._nearest_node_index(concept_a)
        idx_b = self._nearest_node_index(concept_b)

        g = nx.Graph()
        g.add_nodes_from(range(len(self.concepts)))
        sparse = self._graph.tocoo()
        for i, j, w in zip(sparse.row, sparse.col, sparse.data):
            if i < j:
                g.add_edge(i, j, weight=float(w))

        try:
            generator = nx.shortest_simple_paths(g, idx_a, idx_b, weight="weight")
            paths = []
            for i, path in enumerate(generator):
                if i >= k:
                    break
                paths.append([self.concepts[idx] for idx in path])
            return paths
        except nx.NetworkXNoPath:
            return []


if __name__ == "__main__":
    # A deliberate associative chain: each concept shares real vocabulary
    # with its neighbor, so there's an actual bridge to route through --
    # not just noise from tied distances on tiny, unrelated sentences.
    chain = [
        "the cat sat on a mat",                                   # cat
        "cat and dog are common household pets",                  # cat, pets
        "pets are often adopted from animal shelters",             # pets, shelters
        "animal shelters rely on charitable donations",            # shelters, donations
        "charitable donations often provide tax benefits",         # donations, tax
        "tax policy is shaped by economic committees",             # tax, economic
        "economic committees respond to interest rates changes",    # economic, interest, rates
        "interest rates are set by the central bank",              # interest, rate, central bank
    ]
    graph = ConceptGraph(k_neighbors=3)
    graph.build(chain)

    print("Adjacent concepts in the chain (should be close to direct):")
    r1 = graph.path(chain[0], chain[1])
    print(f"  refractive_index={r1.refractive_index:.3f}  hops={r1.hops}")

    print("\nOpposite ends of the chain (must route through every intermediate link):")
    r2 = graph.path(chain[0], chain[-1])
    print(f"  refractive_index={r2.refractive_index:.3f}")
    print(f"  straight_line={r2.straight_line_distance:.3f}  path={r2.path_distance:.3f}")
    print(f"  hops={r2.hops}")
