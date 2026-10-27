"""
topology.py — s1: the actual shape of the AI's associative memory, made
of real graph-theory measurements instead of a metaphor.

PLAIN-LANGUAGE EXPLANATION:

  Take everything in the memory store, connect related items (same
  ConceptGraph machinery as s7), and now look at the SHAPE of that
  network as a whole, not just one path through it:

    - HUB CONCEPTS (high-degree nodes): ideas that connect to many
      others -- the "well-worn pathways" or attractor craters, in
      the original vocabulary.
    - ISOLATED CONCEPTS (degree 0, no strong connections to anything
      else remembered): the "cliffs" -- content sitting on its own,
      disconnected from everything else.
    - CLUSTERS (connected components): separate islands of related
      memory that don't currently link to each other at all.
    - DENSITY / CLUSTERING COEFFICIENT: real network-science measures
      of how tightly interconnected the terrain is overall -- a
      literal (not metaphorical) stand-in for "flat plain" (sparse,
      low-density) vs. "rugged terrain" (dense, tightly clustered).

  render() actually draws this as a picture -- nodes and edges laid
  out with a force-directed layout -- so you can literally see the
  shape, the way the original description asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .geodesics import ConceptGraph


@dataclass
class TopologySnapshot:
    node_count: int
    edge_count: int
    density: float                    # 0 (sparse/flat) to 1 (fully connected/dense)
    avg_clustering: float              # how tightly local neighborhoods cluster together
    hub_concepts: list[tuple[str, int]]     # (text, degree) sorted by degree, most-connected first
    isolated_concepts: list[str]        # concepts with zero connections
    clusters: list[list[str]]            # connected components -- separate "islands"


class TopologyMap:
    def __init__(self, k_neighbors: int = 3, max_distance: float = 0.9):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance
        self._concept_graph: ConceptGraph | None = None
        self._nx_graph: nx.Graph | None = None

    def build(self, concepts: list[str]) -> TopologySnapshot:
        self._concept_graph = ConceptGraph(k_neighbors=self.k_neighbors, max_distance=self.max_distance)
        self._concept_graph.build(concepts)

        g = nx.Graph()
        g.add_nodes_from(range(len(concepts)))
        sparse = self._concept_graph._graph.tocoo()
        for i, j, w in zip(sparse.row, sparse.col, sparse.data):
            if i < j:  # undirected, avoid double-adding
                g.add_edge(i, j, weight=w)
        self._nx_graph = g

        return self._snapshot(concepts)

    def _snapshot(self, concepts: list[str]) -> TopologySnapshot:
        g = self._nx_graph
        n = g.number_of_nodes()

        degrees = dict(g.degree())
        hubs = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)
        hub_concepts = [(concepts[i], d) for i, d in hubs if d > 0][:5]
        isolated_concepts = [concepts[i] for i, d in degrees.items() if d == 0]

        components = [sorted(concepts[i] for i in comp) for comp in nx.connected_components(g)]
        components.sort(key=len, reverse=True)

        return TopologySnapshot(
            node_count=n,
            edge_count=g.number_of_edges(),
            density=nx.density(g),
            avg_clustering=nx.average_clustering(g) if n > 0 else 0.0,
            hub_concepts=hub_concepts,
            isolated_concepts=isolated_concepts,
            clusters=components,
        )

    def render(self, path: str, concepts: list[str] | None = None) -> str:
        """Draw the actual shape: nodes and edges, force-directed layout, saved as an image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._nx_graph is None:
            raise RuntimeError("Call build(concepts) before render().")

        g = self._nx_graph
        pos = nx.spring_layout(g, seed=42)
        degrees = dict(g.degree())
        sizes = [200 + 300 * degrees[n] for n in g.nodes()]

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color="#4C72B0", alpha=0.85)
        ax.set_title("Topology map (s1): node size = degree (well-connected concepts)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


if __name__ == "__main__":
    concepts = [
        "the cat sat on a mat",
        "cat and dog are common household pets",
        "pets are often adopted from animal shelters",
        "animal shelters rely on charitable donations",
        "charitable donations often provide tax benefits",
        "tax policy is shaped by economic committees",
        "economic committees respond to interest rates changes",
        "interest rates are set by the central bank",
        "an entirely unrelated note about volcanoes erupting",
    ]
    topo = TopologyMap(k_neighbors=3)
    snapshot = topo.build(concepts)
    print(f"nodes={snapshot.node_count} edges={snapshot.edge_count} density={snapshot.density:.3f}")
    print(f"avg_clustering={snapshot.avg_clustering:.3f}")
    print(f"hub_concepts={snapshot.hub_concepts}")
    print(f"isolated_concepts={snapshot.isolated_concepts}")
    print(f"clusters={snapshot.clusters}")
