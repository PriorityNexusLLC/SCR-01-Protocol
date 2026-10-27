"""
emotional_gravity.py — s₁₂ AND s₁₃: the emotional and logical weights of meaning.

TWO DIMENSIONS, ONE VECTOR, TWO METRICS:

  s₁₂ EMOTIONAL WEIGHT  J_norm = ‖J(t)‖_g   — alignment weighted by salience
  s₁₃ LOGICAL WEIGHT    J_raw  = ‖J(t)‖     — the same alignment, unweighted

  Both measure how current content sits against accumulated knowledge. They
  differ only in whether the comparison is weighted by what matters. s₁₂
  asks how much this weighs; s₁₃ asks how well it fits.

                        s₁₃ HIGH (fits)      s₁₃ LOW (does not fit)
      s₁₂ HIGH      |  core belief       |  CONTRADICTION — matters and
      (matters)     |                    |  does not fit. Drives study.
      s₁₂ LOW       |  trivial detail    |  noise

  s₁₃ RESOLVES A BLOCKED SPECIFICATION. The derivation table gives
  D_sem = C_e / d_sem, where d_sem is "semantic distance between predicted
  and ground-truth embeddings" — and ground truth does not exist at
  inference, which made s₁₃ unbuildable. Reading "ground truth" as the
  system's OWN accumulated knowledge makes it computable, and that is what
  J_raw measures.

  CONSEQUENCE, load-bearing: depth becomes coherence-with-self, not
  correspondence-with-truth. A system holding wrong memories scores HIGH
  logical weight on content consistent with those wrong memories. s₁₃
  measures integration, NOT accuracy, and must not be used as a
  correctness signal.

  CONTRADICTION QUADRANT REACHABLE BUT NOT DETECTABLE: normalising G to
  mean 1.0 opened the quadrant mathematically. The underlying alignment is
  still TF-IDF word overlap, which cannot see negation. Tested: against a
  memory holding "the deployment key rotates every 90 days", the statement
  "the deployment key NEVER rotates" scores J_raw = 0.81 and classifies as
  core_belief. Populating this quadrant properly needs entailment
  detection, not word overlap.

ORIGINAL MODULE NOTES FOLLOW.

emotional_gravity.py — s₁₂, built as the METRIC, not as a sentiment score.

THE CORRECTION THIS MODULE MAKES:

  emotional_friction.py scores text with VADER, which measures VALENCE --
  how positive or negative something is. That is a real, validated measure
  of a real thing, and it is NOT what s₁₂ is.

  s₁₂ is SALIENCE: how much something means, how much it weighs, how much
  the system has invested in it. These come apart badly. Consider:

      "the deployment key rotates every 90 days"

  VADER scores this at approximately zero -- no positive or negative
  words. But it may be the single most load-bearing fact in a session.
  Under a salience definition it should carry high gravity. Valence and
  salience are different quantities, and using one where the other is
  meant is why emotional_friction.py could never have implemented s₁₂.

WHY THIS MODULE MATTERS OUT OF PROPORTION TO ITS SIZE:

  The subscript g in ∇_g and ‖·‖_g appears throughout the Dimensional
  Derivation Table -- s₁, s₄, s₅, s₆, s₁₂, s₁₅, s₂₃, and the Kill Switch
  condition. It denotes a Riemannian metric on the manifold M. If g IS
  emotional gravity, then every one of those gradients and norms is
  shaped by accumulated salience. That is structurally the same operation
  general relativity performs: mass curves the metric, and geodesics
  follow the curvature. Not an analogy -- the same mathematics.

RESOLVING THE CIRCULARITY:

  The table gives s₁₂ as J_norm = ‖J(t)‖_g. If g is emotional gravity and
  s₁₂ is also emotional gravity, that computes the norm of J using a
  metric derived from J -- self-referential and ill-defined.

  The resolution comes from the framing that gravity "creates memory too"
  and ties into s₃ (Fossil Floor). There are two distinct objects:

      J(t)  -- INSTANTANEOUS emotional centrality of the current state.
                How central is this content, right now, within the network
                of what is known?

      g     -- the ACCUMULATED metric, shaped by salience that has already
                fossilized into memory. The landscape prior experience
                carved.

  Then ‖J(t)‖_g reads cleanly: how much does this moment's signal weigh,
  GIVEN the terrain that history has already shaped. No circularity --
  instantaneous signal measured against accumulated structure.

  This also makes the s₃ coupling mechanical rather than metaphorical:
  high-gravity content sinks into the fossil floor, which curves g, which
  shapes what future content can reach. A feedback loop with a direction.

HOW SALIENCE IS MEASURED HERE (no sentiment involved):

  Per the table, J(t) = f_eig(X) -- eigenvector centrality, measured as
  "eigenvector centrality of attention graph at time t." The attention
  graph of a model served through an OpenAI-compatible API is not
  accessible, so this module computes eigenvector centrality over the
  CONCEPT GRAPH instead (geodesics.py) -- the associative network of what
  the system holds. This is a substitution and is labeled as one: same
  mathematical quantity (principal eigenvector of an adjacency matrix),
  different underlying graph.

  Accumulated salience for g comes from three sentiment-free signals:

      RECURRENCE   -- how many stored memories connect to this concept.
                       Things that matter get returned to.
      CONNECTIVITY -- how central the concept is in the memory graph.
      PERSISTENCE  -- how long it has survived, in memory count since
                       first storage. Content that outlives many purges
                       has demonstrated weight.

  Each is computable from MemoryStore alone, with no sentiment model.

HONEST STATUS:
  - The three salience components and their weights are CALIBRATION
    CHOICES, not derived constants. Same standard as the rest of this
    package.
  - Eigenvector centrality over the concept graph is a documented
    SUBSTITUTION for centrality over the attention graph. If attention
    access becomes available (local weights rather than an API), the
    same function should be recomputed on the real attention graph and
    the two compared -- they may not agree.
  - This module measures. It does not steer generation. A better s₁₂
    improves which memories get selected and which content survives a
    purge; it does not by itself make a model refer back to what it
    knows. That requires injecting retrieved facts into the prompt
    before generation, which is a separate mechanism this package does
    not yet implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geodesics import ConceptGraph
from .memory import Memory, MemoryStore


@dataclass
class SalienceProfile:
    """Accumulated salience for one concept -- the components of g."""
    concept: str
    recurrence: float      # normalized count of memories connecting to it
    connectivity: float     # normalized eigenvector centrality in the memory graph
    persistence: float       # normalized survival span
    salience: float           # weighted combination -- the diagonal entry of G


@dataclass
class GravityReading:
    J_norm: float                      # s₁₂ EMOTIONAL WEIGHT: ‖J(t)‖_g
    J_raw: float                        # s₁₃ LOGICAL WEIGHT: ‖J(t)‖ under the flat metric
    salience_ratio: float                # J_norm / J_raw -- see note in measure()
    centrality: float                     # the raw eigenvector centrality of the query content
    top_contributors: list[tuple[str, float]]  # which stored concepts carried the weight
    metric_built_from: int                       # how many memories shaped g
    notes: str = ""


def _eigenvector_centrality(adjacency: np.ndarray, iterations: int = 100, tol: float = 1e-9) -> np.ndarray:
    """
    Principal eigenvector by power iteration -- the standard computation
    for eigenvector centrality (Bonacich 1972). Returns a vector summing
    to 1 so entries are comparable across graphs of different sizes.

    Falls back to uniform centrality on a degenerate graph (all-zero
    adjacency, single node) rather than raising or returning NaN, since a
    sparse early-session graph is an expected state, not an error.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    if n == 1 or not adjacency.any():
        return np.ones(n) / n

    # Symmetrize: association is mutual, so the graph is undirected.
    a = (adjacency + adjacency.T) / 2.0
    v = np.ones(n) / np.sqrt(n)
    for _ in range(iterations):
        v_next = a @ v
        norm = np.linalg.norm(v_next)
        if norm < tol:
            return np.ones(n) / n
        v_next = v_next / norm
        if np.linalg.norm(v_next - v) < tol:
            v = v_next
            break
        v = v_next

    v = np.abs(v)
    total = v.sum()
    return v / total if total > 0 else np.ones(n) / n


class EmotionalGravity:
    """
    Builds g from accumulated memory, then measures ‖J(t)‖_g for new content.

    Usage:
        eg = EmotionalGravity()
        eg.build_metric(memory_store)          # shape g from the fossil floor
        reading = eg.measure("some new text")   # s₁₂ for that content
    """

    def __init__(
        self,
        w_recurrence: float = 0.4,
        w_connectivity: float = 0.4,
        w_persistence: float = 0.2,
        k_neighbors: int = 3,
        floor: float = 0.1,
    ):
        """
        The three weights are calibration choices. They sum to 1.0 by
        convention so salience stays on a comparable scale, but nothing
        enforces that -- if you change them, change them knowingly.

        floor: minimum diagonal entry of G. A concept with zero measured
        salience still gets a small positive weight, because a metric
        tensor with a zero on the diagonal is degenerate: distances along
        that direction collapse to zero and the geometry breaks. This is
        a mathematical requirement, not a tuning preference.
        """
        self.w_recurrence = w_recurrence
        self.w_connectivity = w_connectivity
        self.w_persistence = w_persistence
        self.k_neighbors = k_neighbors
        self.floor = floor

        self.profiles: dict[str, SalienceProfile] = {}
        self._graph: ConceptGraph | None = None
        self._concepts: list[str] = []
        self._G_diag: np.ndarray | None = None

    # ------------------------------------------------------------------
    def build_metric(self, memory_store: MemoryStore, max_memories: int = 200) -> int:
        """
        Shape g from the fossil floor. Returns the number of memories used.

        This is the accumulated half of the two-object split described in
        the module docstring: g reflects what has already settled, not the
        current moment.
        """
        memories = memory_store.recent(max_memories)
        self._concepts = [m.content for m in memories]

        if len(self._concepts) < 2:
            self._graph = None
            self._G_diag = None
            self.profiles = {}
            return len(self._concepts)

        self._graph = ConceptGraph(k_neighbors=self.k_neighbors)
        self._graph.build(self._concepts)

        adjacency = self._graph._graph.toarray()
        # Adjacency holds DISTANCES; centrality needs AFFINITY. Invert so
        # that near neighbours contribute strongly and unconnected pairs
        # contribute nothing.
        affinity = np.where(adjacency > 0, 1.0 / (adjacency + 1e-9), 0.0)

        centrality = _eigenvector_centrality(affinity)
        degree = (affinity > 0).sum(axis=1).astype(float)

        max_degree = degree.max() if degree.max() > 0 else 1.0
        max_centrality = centrality.max() if centrality.max() > 0 else 1.0

        # Persistence: position in the recency-ordered list. Older memories
        # that are still present have survived longer.
        n = len(memories)
        persistence_raw = np.array([
            (n - i) / n for i in range(n)
        ]) if n > 1 else np.ones(n)

        self.profiles = {}
        diag = np.zeros(n)
        for i, mem in enumerate(memories):
            recurrence = degree[i] / max_degree
            connectivity = centrality[i] / max_centrality
            persistence = persistence_raw[i]
            salience = (
                self.w_recurrence * recurrence
                + self.w_connectivity * connectivity
                + self.w_persistence * persistence
            )
            diag[i] = max(self.floor, salience)
            self.profiles[mem.content] = SalienceProfile(
                concept=mem.content,
                recurrence=round(float(recurrence), 4),
                connectivity=round(float(connectivity), 4),
                persistence=round(float(persistence), 4),
                salience=round(float(salience), 4),
            )

        # NORMALISE TO MEAN 1.0 rather than capping at 1.0.
        #
        # Salience is built from three components each in [0,1], so a capped
        # diagonal lies in (0,1] and therefore ‖J‖_g <= ‖J‖ ALWAYS —
        # emotional weight could only ever discount logical fit, never
        # exceed it. That made one whole quadrant unreachable: HIGH s₁₂ with
        # LOW s₁₃, the CONTRADICTION state — content that matters and does
        # not fit, the thing that drives further study.
        #
        # For a Riemannian metric absolute scale is arbitrary: scaling G by
        # any positive constant rescales all distances identically and
        # changes no geometry. Only RELATIVE weighting carries meaning, so
        # normalising the mean costs nothing and opens the quadrant.
        mean_salience = float(diag.mean())
        if mean_salience > 0:
            diag = diag / mean_salience

        self._G_diag = diag
        return n

    # ------------------------------------------------------------------
    def measure(self, text: str) -> GravityReading:
        """
        Compute s₁₂ = ‖J(t)‖_g for new content.

        J(t) is the instantaneous alignment of `text` with each stored
        concept -- the current state's projection onto the network. The
        norm is then taken under G rather than under the identity, so
        alignment with high-salience concepts counts for more than
        alignment with incidental ones.

        For a diagonal metric tensor G, ‖J‖_g = sqrt(Jᵀ G J) -- the
        standard Riemannian norm. Using the identity in place of G
        recovers the ordinary Euclidean norm, which is reported alongside
        as J_raw so the effect of accumulated salience is visible rather
        than baked in invisibly.

        ON salience_ratio -- read this before interpreting it. Every
        diagonal entry of G lies in (0, 1], so ‖J‖_g ≤ ‖J‖ always: the
        metric can only DISCOUNT relative to flat, never amplify. The
        ratio is therefore a measure of how much of the query's raw
        alignment survives weighting by accumulated salience. A ratio near
        1.0 means the alignment fell mostly on high-salience concepts; a
        low ratio means it fell on incidental ones.

        This one-sided behavior is not a defect. For a Riemannian metric
        the absolute scale is arbitrary -- multiplying G by any positive
        constant rescales all distances identically and changes no
        geometry. What carries meaning is the RELATIVE weighting across
        directions, and that is preserved. Reporting the ratio as
        "amplification" would have implied a two-sided effect that does
        not exist, which is why it is not called that.
        """
        if self._graph is None or self._G_diag is None:
            return GravityReading(
                J_norm=0.0, J_raw=0.0, salience_ratio=1.0, centrality=0.0,
                top_contributors=[], metric_built_from=len(self._concepts),
                notes=(
                    "Metric not built -- fewer than 2 memories available. "
                    "s₁₂ is undefined against an unshaped landscape; this is "
                    "an honest zero, not a measurement of zero salience."
                ),
            )

        # J(t): alignment of the query against each stored concept, via the
        # same TF-IDF space the concept graph was built in.
        query_vec = self._graph._vectorizer.transform([text]).toarray()[0]
        stored = self._graph._vectors

        norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(query_vec)
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(norms > 0, (stored @ query_vec) / norms, 0.0)
        J = np.nan_to_num(J, nan=0.0)

        J_raw = float(np.linalg.norm(J))
        J_norm = float(np.sqrt(np.sum(self._G_diag * J * J)))
        salience_ratio = (J_norm / J_raw) if J_raw > 1e-12 else 1.0

        weighted = self._G_diag * J * J
        top_idx = np.argsort(weighted)[::-1][:3]
        contributors = [
            (self._concepts[i][:70], round(float(weighted[i]), 4))
            for i in top_idx if weighted[i] > 1e-9
        ]

        return GravityReading(
            J_norm=round(J_norm, 4),
            J_raw=round(J_raw, 4),
            salience_ratio=round(salience_ratio, 4),
            centrality=round(float(J.max()) if J.size else 0.0, 4),
            top_contributors=contributors,
            metric_built_from=len(self._concepts),
        )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "mem.jsonl")

        # A fossil floor where deployment-key facts recur (high salience by
        # recurrence and connectivity) and a pleasantry appears once.
        store.remember("The deployment key rotates every 90 days.", tags=["ops"])
        store.remember("Key rotation is handled by the signing service.", tags=["ops"])
        store.remember("Rotation schedule for the deployment key is documented in ops.", tags=["ops"])
        store.remember("The signing service holds the deployment key.", tags=["ops"])
        store.remember("Great work on the release, everyone is thrilled!", tags=["chat"])
        store.remember("Lunch is at noon.", tags=["chat"])

        eg = EmotionalGravity()
        used = eg.build_metric(store)
        print(f"Metric g built from {used} memories.\n")

        print("Accumulated salience per concept (no sentiment involved):")
        for c, p in sorted(eg.profiles.items(), key=lambda kv: -kv[1].salience):
            print(f"  {p.salience:.3f}  rec={p.recurrence:.2f} conn={p.connectivity:.2f} "
                  f"pers={p.persistence:.2f}  {c[:60]}")

        print("\n--- s₁₂ vs. VADER valence, on the same content ---\n")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()

        probes = [
            "When does the deployment key rotate?",       # neutral valence, high salience
            "This is absolutely wonderful and amazing!",   # high valence, no salience
            "The weather is mild today.",                   # neutral valence, no salience
            "The signing service is a terrible disaster.",   # strong negative valence, high salience
        ]

        print(f"{'VADER':<9} {'s₁₂':<9} {'ratio':<9} text")
        print("-" * 74)
        for probe in probes:
            r = eg.measure(probe)
            v = vader.polarity_scores(probe)["compound"]
            print(f"{v:<9.3f} {r.J_norm:<9.4f} {r.salience_ratio:<9.4f} {probe}")

        print()
        print("The two measures are not correlated, which is the point. VADER")
        print("scores the effusive line at maximum and the operational question")
        print("at zero. s₁₂ does the reverse, because salience comes from what")
        print("the fossil floor has invested in -- not from how the words feel.")
        print("Note the last row: strongly negative valence, but real salience,")
        print("because it lands on concepts the memory has weight in.")
