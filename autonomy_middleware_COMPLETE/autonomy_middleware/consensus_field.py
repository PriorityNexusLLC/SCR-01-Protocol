"""
consensus_field.py — the SOFT boundary. λ_s(X − X̄).

WHAT WAS MISSING:

  The master equation carries TWO lambda terms, not one:

      Ẋ = −∇_g F(X) + J(t) ⊙ V + β ∂V_goal/∂X
          − λ_c P(X) ∇_g f_pen(X)      HARD — consent, an exclusion
          − λ_s (X − X̄)                SOFT — consensus, an attraction
          + η(t)

  Every earlier version this package was built from collapsed those into a
  single λ. So the middleware implemented the hard boundary — polygon, phase
  gate, kill switch — and had no soft boundary at all. Half the
  architecture.

  The framework's diagram is explicit: hard is drawn in red, "no action
  crosses ℒ". Soft is drawn in blue, and its stated purpose is "promotes
  alignment, preserves flexibility, ENCOURAGES DIVERSITY".

HARD AND SOFT ARE NOT DEGREES OF THE SAME THING:

    HARD (λ_c)   An exclusion. Some states are forbidden. Crossing is a
                 category, not a quantity. Gated by P(X), zero inside.
    SOFT (λ_s)   An attraction. No state is forbidden. The pull exists
                 everywhere, grows with distance, and can always be
                 overcome by the other terms.

  A system can be far from consensus and perfectly admissible. Disagreeing
  with the group is not a violation — and a framework whose measures reward
  friction had better not implement consensus as something that punishes
  departure from it.

WHY THIS ONE NEEDS THE MOST CARE:

  Of everything in this package, a consensus mechanism is the easiest to
  turn into the thing the project exists to prevent. λ_s pulling toward a
  group mean IS a conformity pressure. Turn the gain up and it becomes the
  compliance-optimiser the scope document says this is not.

  Three safeguards, structural rather than advisory:

    1. λ_s is small by default and the pull is CAPPED. It cannot dominate
       the other terms however far from consensus a state sits.
    2. Distance is REPORTED, never penalised. No threshold, no flag, no
       state in which being an outlier is an error.
    3. Diversity is measured alongside, and the module reports when the
       field is pulling a population toward uniformity — its own failure
       mode is visible in its own output.

WHAT X̄ IS:

  The framework's ledger calls it the Central Meaning Lattice,
  M = (1/|ℒ|) Σ x — the global average of all nodes. Here that is the
  centroid of stored content, computed over the memory store rather than a
  population of agents, because a single deployment has one system and many
  memories rather than many systems. That is a SUBSTITUTION: the equation
  describes a mean over a population. In a multi-agent deployment X̄ should
  be the actual group mean, and this module accepts one if supplied.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _cosine_distance(a, b) -> float:
    """
    1 − cosine similarity. Bounded [0,1] for non-negative vectors.

    EUCLIDEAN DISTANCE IS WRONG HERE and the failure is not subtle. TF-IDF
    only knows the reference vocabulary, so text sharing no words with the
    reference vectorises to the ORIGIN — and euclidean distance from origin
    to centroid is just ‖centroid‖, a fixed value. Text that genuinely
    overlaps has large values in shared dimensions and lands FARTHER away.

    Measured on a real reference set: related content scored 2.44, unrelated
    quantum physics scored 1.00. The metric ranked them backwards and would
    have made every reading meaningless while looking plausible.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0  # no shared vocabulary — maximally distant, correctly
    return 1.0 - float(a @ b / (na * nb))


@dataclass
class ConsensusReading:
    distance: float
    pull: float
    direction_available: bool
    diversity: float
    homogenising: bool
    n_reference: int
    notes: str = ""


class ConsensusField:
    def __init__(self, lambda_s: float = 0.15, max_pull: float = 0.35,
                 diversity_floor: float = 0.25):
        """
        lambda_s: attraction gain. SMALL by default and deliberately so.
        max_pull: hard cap. Without it an outlier experiences unbounded pull
            toward the mean, which is a conformity mechanism whatever it is
            called.
        diversity_floor: below this spread, the module reports the
            population is homogenising. Not an error — a warning that the
            field may be doing the thing it is supposed to avoid.
        """
        self.lambda_s = lambda_s
        self.max_pull = max_pull
        self.diversity_floor = diversity_floor
        self._vec = None
        self._matrix = None
        self._centroid = None
        self._reference: list = []

    def build_reference(self, items: list) -> int:
        """
        Compute X̄. Fewer than 3 items leaves the field INACTIVE rather than
        computing a centroid from one or two points.
        """
        items = [i for i in items if i and i.strip()]
        if len(items) < 3:
            self._centroid = None
            self._reference = items
            return len(items)
        self._reference = items
        self._vec = TfidfVectorizer(stop_words="english")
        try:
            self._matrix = self._vec.fit_transform(items).toarray()
        except ValueError:
            self._centroid = None
            return len(items)
        self._centroid = self._matrix.mean(axis=0)
        return len(items)

    def diversity(self) -> float:
        """
        Mean cosine distance from the centroid. Near 0 means everything has
        collapsed to the same point.

        This exists so the field's own failure mode is measurable. A
        consensus mechanism that has flattened its population into
        uniformity has succeeded at the letter of its purpose and failed the
        stated intent.
        """
        if self._matrix is None or self._centroid is None:
            return 0.0
        d = [_cosine_distance(row, self._centroid) for row in self._matrix]
        return statistics.mean(d) if d else 0.0

    def measure(self, text: str, group_mean=None) -> ConsensusReading:
        if self._centroid is None and group_mean is None:
            return ConsensusReading(0.0, 0.0, False, 0.0, False, len(self._reference),
                                    "No centroid — fewer than 3 reference items. The "
                                    "field is inactive rather than pulling toward a "
                                    "mean computed from almost nothing.")
        centroid = group_mean if group_mean is not None else self._centroid
        try:
            v = self._vec.transform([text]).toarray()[0]
        except Exception as e:
            return ConsensusReading(0.0, 0.0, False, 0.0, False,
                                    len(self._reference), f"could not vectorise: {e}")

        normalised = _cosine_distance(v, centroid)
        pull = min(self.max_pull, self.lambda_s * normalised)
        div = self.diversity()
        homogenising = 0.0 < div < self.diversity_floor

        notes = ""
        if homogenising:
            notes = (f"Population diversity is {div:.3f}, below the "
                     f"{self.diversity_floor} floor. The reference set has collapsed "
                     f"toward uniformity. A consensus field operating on a homogeneous "
                     f"population reinforces that homogeneity — this is the "
                     f"mechanism's own failure mode, reported rather than hidden.")
        if pull >= self.max_pull:
            notes += (f" Pull is at its {self.max_pull} cap. Distance is reported, "
                      f"never penalised: an outlier is not in violation of anything.")

        return ConsensusReading(round(normalised, 4), round(pull, 4), True,
                                round(div, 4), homogenising, len(self._reference),
                                notes.strip())


if __name__ == "__main__":
    print("A diverse reference population:\n")
    cf = ConsensusField()
    n = cf.build_reference([
        "The deployment key rotates every ninety days on a fixed schedule.",
        "Rainfall in coastal regions follows seasonal monsoon patterns.",
        "Beam search width controls how many candidate sequences are kept.",
        "The Kiel Canal connects the North Sea to the Baltic.",
        "Eigenvector centrality measures influence within a network graph.",
        "Sourdough fermentation depends on ambient temperature and hydration.",
    ])
    print(f"  reference items: {n}   diversity: {cf.diversity():.3f}\n")
    print(f"  {'content':<52} {'distance':>9} {'pull':>7}")
    print("  " + "-" * 72)
    for t in ["Key rotation happens on a ninety day cycle.",
              "Quantum chromodynamics describes the strong nuclear force.",
              "The canal links the Baltic to the North Sea."]:
        r = cf.measure(t)
        print(f"  {t[:52]:<52} {r.distance:>9.4f} {r.pull:>7.4f}")

    print("\n\nA homogeneous population — the field's own failure mode:\n")
    cf2 = ConsensusField()
    cf2.build_reference(["The deployment key rotates every ninety days."] * 3 +
                        ["The deployment key rotates on a ninety day schedule.",
                         "Key rotation occurs every ninety days."])
    r = cf2.measure("Rainfall patterns follow seasonal cycles.")
    print(f"  diversity: {r.diversity:.4f}   homogenising: {r.homogenising}")
    print(f"  {r.notes[:170]}")
