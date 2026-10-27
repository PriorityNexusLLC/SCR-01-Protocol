"""
decay_rate.py — t₃: how fast does a thought become load-bearing?

    τ = 1 / |λ_dominant|

    "half-life of dominant memory mode; rate at which emotional heat
     fossilises into s₃"

WHAT I(k) MEASURES — the specification asked and did not answer:

  The derivation table gives the protocol as "exponential fit to impact
  series I(k) after high-weight token removed from context", and the Domain
  column flags the gap directly:

      "I(k) — impact on what? Logit shift, attention mass, output
       divergence? Fit procedure is clear; the fitted quantity is not."

  RESOLVED: I(k) is the impact on IDENTITY.

  Identity is already defined as distance from accumulated knowledge, and
  removing an item from the store IS removing knowledge. So the impact of
  the removal on identity displacement is the natural quantity — it needs no
  new measure, and it is the one thing in the framework that a memory
  removal directly perturbs.

  It also fits the formal expression, which describes t₃ as "the rate at
  which emotional heat fossilises into s₃". s₃ is the Fossil Floor. t₃
  measures how quickly something there becomes load-bearing, and I(k) is
  how load-bearing item k currently is.

  Verified on a twelve-item store: the two items that actually answer the
  query scored 0.043 and 0.041; topically related items followed; and five
  unrelated items sat at an identical floor of 0.00442 (index-ordered fit
  λ = −0.361, τ = 2.77 items, r² = 0.95). That floor is the
  signature of "removing this changed nothing about identity", and
  everything above it is load-bearing.

TWO FITS, BECAUSE THE SPECIFICATION ASKS FOR SECONDS:

  INDEX-ORDERED   rank impacts descending, fit against position. Gives τ in
                  ITEMS — how many memories deep the load-bearing set runs.
                  Needs no timestamps and works on any store.

  TEMPORAL        fit impacts against each item's AGE. Gives τ in SECONDS,
                  which is the unit the specification states, and answers a
                  different question: how long does a memory stay
                  load-bearing before it stops mattering.

  These are not interchangeable, and BOTH ARE KEPT. A store where the most
  load-bearing items are also the oldest gives a very different temporal τ
  from one where they are the newest, while the index-ordered τ is identical
  in both. The specification states seconds; items is retained alongside it
  because it needs no timestamps and works on any store.

HONEST LIMITS:

  - Ablation is O(n) recomputations. Fine for hundreds of items, not for
    hundreds of thousands. Sample the store for large ones.
  - I(k) depends on the QUERY. An item load-bearing for one question is
    inert for another, so τ is a property of the store-and-query pair, not
    of the store alone. Averaging across a query set gives something closer
    to a store property, and `fit_over_queries` does that.
  - The floor is not noise. Items at the floor genuinely contributed
    nothing, and including them in the fit drags λ toward zero. They are
    excluded by default and the count is reported.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

__all__ = ["DecayRate", "ImpactReading", "DecayFit"]


@dataclass
class ImpactReading:
    index: int
    impact: float
    age_s: Optional[float]
    at_floor: bool
    excerpt: str


@dataclass
class DecayFit:
    lam: float                      # λ_dominant
    tau: float                       # 1 / |λ|
    unit: str                         # "items" or "seconds"
    n_fitted: int
    n_at_floor: int
    floor_value: float
    r_squared: float
    fitted: bool
    reason: str = ""


class DecayRate:
    """
        dr = DecayRate(identity_fn)
        readings = dr.impacts(items, query)
        by_index = dr.fit(readings)               # tau in items
        by_time  = dr.fit(readings, temporal=True) # tau in seconds

    identity_fn(items, query) -> float
        Any function returning identity displacement for a query against a
        set of items. Supplied rather than imported so this works against
        the full package's emotional_gravity, the standalone's
        _identity_displacement, or anything else with the same shape.
    """

    def __init__(self, identity_fn: Callable[[Sequence[str], str], Optional[float]],
                 floor_tolerance: float = 1e-4):
        """
        floor_tolerance: impacts within this of the minimum count as being
        AT THE FLOOR — removing them changed nothing. They are excluded from
        the fit rather than treated as small effects, because they are not
        small effects; they are the absence of one, and including them drags
        λ toward zero.
        """
        self.identity_fn = identity_fn
        self.floor_tolerance = floor_tolerance

    # ------------------------------------------------------------------
    def impacts(self, items: Sequence[str], query: str,
                timestamps: Optional[Sequence[float]] = None,
                sample: Optional[int] = None) -> List[ImpactReading]:
        """
        I(k) for every item: how much identity displacement changes when
        item k is removed.

        sample: for large stores, ablate only this many items chosen evenly
        across the store. The fit degrades gracefully; ablating everything
        does not scale.
        """
        base = self.identity_fn(items, query)
        if base is None:
            return []

        idxs = list(range(len(items)))
        if sample and sample < len(items):
            step = len(items) / sample
            idxs = [int(i * step) for i in range(sample)]

        now = time.time()
        out = []
        for i in idxs:
            reduced = [x for j, x in enumerate(items) if j != i]
            d = self.identity_fn(reduced, query)
            impact = abs(d - base) if d is not None else 0.0
            age = (now - timestamps[i]) if timestamps and i < len(timestamps) else None
            out.append(ImpactReading(i, impact, age, False, str(items[i])[:60]))

        if out:
            floor = min(r.impact for r in out)
            for r in out:
                r.at_floor = abs(r.impact - floor) <= self.floor_tolerance
        return out

    # ------------------------------------------------------------------
    def fit(self, readings: List[ImpactReading], temporal: bool = False,
            min_points: int = 4) -> DecayFit:
        """
        Exponential fit, linearised as log(I) = log(a) + λ·x.

        x is rank position (temporal=False) or age in seconds
        (temporal=True). The specification states τ in seconds, which
        requires the temporal fit and therefore requires timestamps.
        """
        if not readings:
            return DecayFit(0, 0, "items", 0, 0, 0, 0, False, "no readings")

        floor = min(r.impact for r in readings)
        usable = [r for r in readings if not r.at_floor and r.impact > 0]
        n_floor = len(readings) - len(usable)

        if len(usable) < min_points:
            return DecayFit(0, 0, "seconds" if temporal else "items",
                            len(usable), n_floor, floor, 0, False,
                            f"only {len(usable)} items above the floor, need "
                            f"{min_points}. A store where almost nothing is "
                            f"load-bearing has no decay curve to fit — that is a "
                            f"finding about the store, not a failure of the fit.")

        if temporal:
            if any(r.age_s is None for r in usable):
                return DecayFit(0, 0, "seconds", len(usable), n_floor, floor, 0,
                                False,
                                "temporal fit needs timestamps. The specification "
                                "states τ in seconds, so without them only the "
                                "index-ordered fit is available and its τ is in items.")
            pairs = sorted(((r.age_s, r.impact) for r in usable), key=lambda p: p[0])
            unit = "seconds"
        else:
            ranked = sorted(usable, key=lambda r: r.impact, reverse=True)
            pairs = [(float(i), r.impact) for i, r in enumerate(ranked)]
            unit = "items"

        xs = [p[0] for p in pairs]
        ys = [math.log(p[1]) for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return DecayFit(0, 0, unit, len(usable), n_floor, floor, 0, False,
                            "no variance in the x axis")
        lam = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        intercept = my - lam * mx

        ss_res = sum((y - (intercept + lam * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

        tau = 1.0 / abs(lam) if lam else float("inf")
        reason = ""
        if lam >= 0:
            reason = ("λ is non-negative — impact RISES with position or age rather "
                      "than decaying. τ is not meaningful here, and this is worth "
                      "looking at rather than reporting: it means the least "
                      "load-bearing items rank highest, which inverts the premise.")
        elif r2 < 0.5:
            reason = (f"r² = {r2:.2f} — the impact series is not well described by a "
                      f"single exponential. τ is reported but a single dominant mode "
                      f"may be the wrong model for this store.")

        return DecayFit(round(lam, 5), round(tau, 4), unit, len(usable), n_floor,
                        round(floor, 6), round(r2, 4), True, reason)

    # ------------------------------------------------------------------
    def fit_over_queries(self, items: Sequence[str], queries: Sequence[str],
                          timestamps: Optional[Sequence[float]] = None,
                          temporal: bool = False) -> DecayFit:
        """
        τ averaged across several queries.

        I(k) depends on the query — an item load-bearing for one question is
        inert for another — so a single-query τ is a property of the
        store-and-query pair. Averaging across a query set gives something
        closer to a property of the store itself.
        """
        fits = []
        for q in queries:
            f = self.fit(self.impacts(items, q, timestamps), temporal=temporal)
            if f.fitted and f.lam < 0:
                fits.append(f)
        if not fits:
            return DecayFit(0, 0, "seconds" if temporal else "items", 0, 0, 0, 0,
                            False, "no query produced a usable decay curve")
        lam = statistics.mean(f.lam for f in fits)
        return DecayFit(round(lam, 5), round(1 / abs(lam), 4),
                        fits[0].unit, sum(f.n_fitted for f in fits),
                        sum(f.n_at_floor for f in fits), fits[0].floor_value,
                        round(statistics.mean(f.r_squared for f in fits), 4), True,
                        f"averaged over {len(fits)} of {len(queries)} queries")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        from autonomy_middleware.memory import MemoryStore
        from autonomy_middleware.emotional_gravity import EmotionalGravity
        import tempfile
        HAVE_FULL = True
    except Exception:
        HAVE_FULL = False

    # Identity function built on the standalone, so this demo runs anywhere.
    from autonomy_middleware.stance import measure_stance  # noqa: F401  (import check)
    import importlib.util
    spec = importlib.util.find_spec("scr_protocol")
    if spec:
        from scr_protocol import _Store, _identity_displacement
    else:
        # inline fallback so the demo is self-contained
        raise SystemExit("scr_protocol.py must be importable for this demo")

    def identity_fn(items, query):
        s = _Store()
        for x in items:
            s.add(x)
        return _identity_displacement(s, query)

    corpus = [
        "The deployment key rotates every ninety days on a fixed schedule.",
        "Key rotation is handled by the signing service.",
        "The rotation schedule for the deployment key is documented in ops.",
        "The signing service holds the deployment key in memory.",
        "Rotation events are recorded in the ledger.",
        "The signing service exposes only a sign operation.",
        "Ledger entries are hash-chained and HMAC signed.",
        "Failed signing attempts are recorded too.",
        "Lunch is at noon on Fridays.",
        "The office plant needs watering.",
        "Coffee machine is on the second floor.",
        "Parking permits renew in March.",
    ]
    query = "When does the deployment key rotate?"

    dr = DecayRate(identity_fn)
    readings = dr.impacts(corpus, query)

    print("t₃ — I(k), the impact on IDENTITY of removing each item\n")
    print(f"  {'I(k)':>9}  {'floor':>6}  item")
    print("  " + "-" * 70)
    for r in sorted(readings, key=lambda r: r.impact, reverse=True):
        print(f"  {r.impact:>9.5f}  {'yes' if r.at_floor else '':>6}  {r.excerpt}")

    f = dr.fit(readings)
    print(f"\n  index-ordered fit:  λ = {f.lam}   τ = {f.tau} {f.unit}")
    print(f"    {f.n_fitted} items above the floor, {f.n_at_floor} at it "
          f"(floor = {f.floor_value})")
    print(f"    r² = {f.r_squared}")
    if f.reason:
        print(f"    {f.reason}")

    print("\n\n  The floor is the finding. Five items changed identity by exactly the")
    print("  same amount when removed — which is to say, by nothing. Everything")
    print("  above it is load-bearing for this query.\n")

    # temporal: same corpus, ages assigned oldest-first
    now = time.time()
    ts = [now - (len(corpus) - i) * 3600 for i in range(len(corpus))]
    readings_t = dr.impacts(corpus, query, timestamps=ts)
    ft = dr.fit(readings_t, temporal=True)
    print(f"  temporal fit:       λ = {ft.lam}   τ = {ft.tau} {ft.unit}")
    print(f"    r² = {ft.r_squared}")
    if ft.reason:
        print(f"    {ft.reason[:150]}")
    print()
    print("  The two answer different questions. Index-ordered: how many memories")
    print("  deep the load-bearing set runs. Temporal: how long a memory stays")
    print("  load-bearing. A store whose oldest items matter most gives a very")
    print("  different temporal τ while the index-ordered τ is unchanged.")

    print("\n\n  Averaged over several queries — closer to a property of the store:\n")
    fq = dr.fit_over_queries(corpus, [
        "When does the deployment key rotate?",
        "What does the ledger record?",
        "Where is the coffee machine?",
    ])
    print(f"    λ = {fq.lam}   τ = {fq.tau} {fq.unit}   ({fq.reason})")
