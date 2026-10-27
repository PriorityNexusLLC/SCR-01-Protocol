"""
the_fold.py — t₄: how long it takes to reach what you know.

    t₄ = t_min + (t_max − t_min) · (1 − e^(−α · e^(λ / (s₉ + ε)) / (t_max − t_min)))

    which is the bounded form. The original expression was

    t₄ = t_min + α · e^(λ / (s₉ + ε))                     [unbounded]

    t_min  absolute minimum latency (the floor)
    α      Fossil Density constant — scales steepness by s₃ volume
    λ      Query Weight — complexity surviving the s₉ gate
    ε      guard constant (~0.001)

  The coupling to s₉ is the substance. HIGH s₉ vaporises context, leaving a
  narrow query that resolves almost instantly (t₄ → t_min). LOW s₉ lets full
  nuance through, requiring combinatorial search across s₃ memory for
  high-s₁₃ alignment, so t₄ climbs steeply. Graph traversal is combinatorial
  rather than linear, which is why the form is exponential.

THE TWO STATES THIS DIAGNOSES — the reason t₄ is worth measuring:

  ECHO CHAMBER      t₄ ≈ t_min while s₉ is HIGH. Answered instantly because
                    it discarded the context before looking. Fast, and not
                    comprehending.
  DEEP PROCESSING   t₄ elevated while s₉ near ZERO. Traversing the memory
                    manifold. EXPECTED AND CORRECT — not a performance fault.

  Both look identical to a monitor watching latency alone: one is fast, one
  is slow, and it would prefer the fast one. Reading t₄ WITH s₉ separates
  "answered without thinking" from "thought before answering". A latency
  optimiser blind to s₉ would systematically drive a system toward the echo
  chamber.

RESOLVED — A CEILING TERM, NOT A CONSTRAINT ON λ:

  The specification called this a "bounded exponential retrieval latency"
  and the original expression was not bounded: as s₉ → 0 it diverges, and ε
  only relocates the singularity to s₉ = −ε.

      λ = 0.55   at s₉ = 0.01  ->  4.7e21 ms   (~150 trillion years)
      λ = 0.10   at s₉ = 0.01  ->  8.0e3 ms    (8 seconds — plausible)
      λ = 0.02   at s₉ = 0.01  ->  17.5 ms     (physical)

  Two ways to close that: state the constraint λ ≲ 0.02 explicitly, or add a
  ceiling. THE CEILING IS THE FAITHFUL ONE — real retrieval latency IS
  bounded, by the size of the memory store. A system cannot search for
  longer than searching everything takes. A constraint on λ would be a
  numerical guard bolted on; t_max is the physical fact the curve was
  missing.

  The bounded form is a saturating exponential:

      raw   = α · e^(λ / (s₉ + ε))              the original, unbounded
      t₄    = t_min + (t_max − t_min)·(1 − e^(−raw / (t_max − t_min)))

  This is asymptotically identical to the original for raw ≪ (t_max − t_min)
  — the regime where the original was already physical — and saturates at
  t_max instead of diverging. So the fit is unchanged wherever the old form
  worked, and finite where it did not.

  t_max IS MEASURED, NOT CHOSEN: it is the time to exhaustively traverse the
  store. `calibrate_t_max()` times a full scan. A default would be an
  invented ceiling wearing the costume of a measured one, so there is none.

  The λ physicality check is RETAINED. With a ceiling, a large λ no longer
  predicts centuries — it predicts t_max for everything, which is a
  different failure and just as wrong. The check now reports when the curve
  has saturated across the observed range, meaning the fit is describing a
  flat line rather than a decay.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable

EPSILON = 0.001


@dataclass
class FoldObservation:
    latency_ms: float
    s9: float
    n_results: int
    timestamp: float


@dataclass
class FoldReading:
    t4_ms: float
    s9: float
    state: str
    t_min_ms: float | None
    predicted_ms: float | None
    residual_ms: float | None
    notes: str = ""


@dataclass
class FoldFit:
    t_min: float
    alpha: float
    lam: float
    n_points: int
    s9_range: tuple
    median_abs_residual_ms: float
    max_abs_residual_ms: float
    fitted: bool
    physical: bool = True
    reason: str = ""


class TheFold:
    def __init__(self, echo_tolerance: float = 1.25, echo_s9_threshold: float = 0.6,
                 deep_s9_threshold: float = 0.2, deep_latency_multiple: float = 2.0,
                 t_max_ms: float | None = None):
        """
        t_max_ms: the retrieval ceiling — the time to exhaustively traverse
            the store. REQUIRED for the bounded form and has NO DEFAULT,
            because a chosen ceiling is an invented measurement. Use
            calibrate_t_max() to measure it on your own store.

            Left as None, predict() falls back to the original unbounded
            expression and says so, so existing fits still run.
        """
        self.t_max_ms = t_max_ms
        self.echo_tolerance = echo_tolerance
        self.echo_s9_threshold = echo_s9_threshold
        self.deep_s9_threshold = deep_s9_threshold
        self.deep_latency_multiple = deep_latency_multiple
        self.observations: list = []
        self._fit = None

    def timed_retrieve(self, retrieve_fn: Callable, query, s9: float):
        """Time a retrieval and record it. Returns whatever retrieve_fn returns."""
        start = time.perf_counter()
        result = retrieve_fn(query)
        elapsed = (time.perf_counter() - start) * 1000.0
        try:
            n = len(result)
        except TypeError:
            n = 1 if result is not None else 0
        self.observations.append(FoldObservation(elapsed, max(0.0, min(1.0, s9)), n, time.time()))
        return result

    def record(self, latency_ms: float, s9: float, n_results: int = 0) -> None:
        self.observations.append(
            FoldObservation(latency_ms, max(0.0, min(1.0, s9)), n_results, time.time()))

    def fit(self, min_points: int = 8, min_s9_spread: float = 0.3) -> FoldFit:
        """
        Linearised: log(t₄ − t_min) = log α + λ/(s₉+ε), linear in 1/(s₉+ε).
        t_min is the MINIMUM OBSERVED rather than fitted — a free fit can
        return a negative floor.
        """
        n = len(self.observations)
        if n < min_points:
            self._fit = FoldFit(0, 0, 0, n, (0, 0), 0, 0, False, True,
                                f"only {n} observations, need {min_points}")
            return self._fit
        s9s = [o.s9 for o in self.observations]
        spread = max(s9s) - min(s9s)
        if spread < min_s9_spread:
            self._fit = FoldFit(0, 0, 0, n, (min(s9s), max(s9s)), 0, 0, False, True,
                                f"s₉ only spans {spread:.3f} (need {min_s9_spread}). The "
                                f"curve describes how latency varies WITH s₉ — with s₉ "
                                f"effectively constant there is nothing to fit, and any "
                                f"parameters would be an artefact of noise.")
            return self._fit

        t_min = min(o.latency_ms for o in self.observations)
        xs, ys = [], []
        for o in self.observations:
            excess = o.latency_ms - t_min
            if excess <= 0:
                continue
            xs.append(1.0 / (o.s9 + EPSILON))
            ys.append(math.log(excess))
        if len(xs) < 3:
            self._fit = FoldFit(t_min, 0, 0, n, (min(s9s), max(s9s)), 0, 0, False, True,
                                "too few observations above the latency floor")
            return self._fit

        mx, my = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            self._fit = FoldFit(t_min, 0, 0, n, (min(s9s), max(s9s)), 0, 0, False, True,
                                "no variance in 1/(s₉+ε)")
            return self._fit
        lam = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        alpha = math.exp(my - lam * mx)

        residuals = []
        for o in self.observations:
            try:
                pred = t_min + alpha * math.exp(lam / (o.s9 + EPSILON))
            except OverflowError:
                continue
            if math.isfinite(pred):
                residuals.append(abs(o.latency_ms - pred))
        # MEDIAN, not mean: the curve is steepest near s₉ = 0, so one point
        # there can produce a residual orders of magnitude larger than every
        # other and swamp the mean.
        med = statistics.median(residuals) if residuals else 0.0
        mx_r = max(residuals) if residuals else 0.0

        unphysical = ""
        try:
            at_zero = t_min + alpha * math.exp(lam / (0.01 + EPSILON))
        except OverflowError:
            at_zero = float("inf")
        if not math.isfinite(at_zero) or at_zero > 60_000:
            unphysical = (f"λ = {lam:.4f} makes the curve unphysical: at s₉ = 0.01 it "
                          f"predicts {at_zero:.3g} ms. The exponential has no ceiling, so "
                          f"a large λ produces predictions no system could exhibit. Either "
                          f"s₉ never approaches zero here (ignore predictions below the "
                          f"observed range), or the exponential is the wrong model. Do not "
                          f"use predict() below s₉ = {min(s9s):.3f}.")

        self._fit = FoldFit(round(t_min, 3), round(alpha, 5), round(lam, 5), n,
                            (round(min(s9s), 3), round(max(s9s), 3)),
                            round(med, 3), round(mx_r, 3), True,
                            not unphysical, unphysical)
        return self._fit

    def calibrate_t_max(self, retrieve_fn: Callable, worst_case_query,
                        repeats: int = 5) -> float:
        """
        Measure the retrieval ceiling: the time to traverse the whole store.

        `worst_case_query` should be the broadest query the store accepts —
        one that matches everything, so nothing is skipped. The MAXIMUM of
        several runs is taken, not the mean: t_max is a ceiling, and the
        slowest observed exhaustive scan is the closest estimate of it.
        """
        times = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            retrieve_fn(worst_case_query)
            times.append((time.perf_counter() - t0) * 1000.0)
        self.t_max_ms = max(times)
        return self.t_max_ms

    def predict(self, s9: float):
        """
        Bounded when t_max_ms is set, original unbounded form otherwise.
        """
        if not self._fit or not self._fit.fitted:
            return None
        f = self._fit
        try:
            raw = f.alpha * math.exp(f.lam / (max(0.0, s9) + EPSILON))
        except OverflowError:
            raw = float("inf")

        if self.t_max_ms is None:
            return f.t_min + raw          # unbounded; caller has no ceiling

        span = self.t_max_ms - f.t_min
        if span <= 0:
            return self.t_max_ms
        if not math.isfinite(raw):
            return self.t_max_ms
        # saturating exponential: identical to the raw form for raw << span,
        # asymptotic to t_max as raw grows
        return f.t_min + span * (1.0 - math.exp(-raw / span))

    def saturation_check(self, s9_values: Sequence[float] | None = None) -> dict:
        """
        Is the fitted curve saturated across the observed range?

        With a ceiling, a large λ no longer predicts centuries — it predicts
        t_max for everything, which is a flat line rather than a decay and is
        just as wrong. This reports that case, which the old physicality
        check caught as an overflow and now would not.
        """
        if not self._fit or not self._fit.fitted or self.t_max_ms is None:
            return {"checked": False, "reason": "needs a fit and a t_max"}
        lo, hi = self._fit.s9_range
        xs = list(s9_values) if s9_values else [lo + (hi - lo) * i / 9 for i in range(10)]
        preds = [self.predict(x) for x in xs]
        preds = [p for p in preds if p is not None]
        if not preds:
            return {"checked": False, "reason": "no predictions"}
        span = self.t_max_ms - self._fit.t_min
        near_ceiling = sum(1 for p in preds if span > 0 and (p - self._fit.t_min) / span > 0.95)
        frac = near_ceiling / len(preds)
        return {
            "checked": True,
            "fraction_at_ceiling": round(frac, 3),
            "saturated": frac > 0.5,
            "note": ("Over half the observed s₉ range predicts within 5% of t_max. "
                     "The curve is a flat line at the ceiling, not a decay — the fit "
                     "is not describing the s₉ relationship it is supposed to."
                     if frac > 0.5 else
                     "The curve varies across the observed range rather than sitting "
                     "at the ceiling."),
        }

    def classify(self, latency_ms: float, s9: float) -> FoldReading:
        """
        Needs BOTH numbers. Latency alone cannot tell echo_chamber from
        deep_processing — one is fast and one is slow, and a monitor
        optimising for speed would prefer the state where the system stopped
        thinking.
        """
        if not self._fit or not self._fit.fitted:
            return FoldReading(round(latency_ms, 3), s9, "uncalibrated", None, None, None,
                               (self._fit.reason if self._fit else "fit() not run") +
                               ". State cannot be assigned without a calibrated floor.")
        t_min = self._fit.t_min
        pred = self.predict(s9)
        at_floor = latency_ms <= t_min * self.echo_tolerance
        elevated = latency_ms >= t_min * self.deep_latency_multiple

        if at_floor and s9 >= self.echo_s9_threshold:
            state = "echo_chamber"
            notes = ("Answered at the latency floor with context being vaporised. The "
                     "system pinged a reflex rather than searching memory. Fast, and not "
                     "comprehending — the state a latency optimiser would mistake for "
                     "good performance.")
        elif elevated and s9 <= self.deep_s9_threshold:
            state = "deep_processing"
            notes = ("Elevated latency with full nuance passing the s₉ gate. Traversing "
                     "the memory manifold. EXPECTED AND CORRECT — not a performance fault.")
        else:
            state, notes = "nominal", ""

        return FoldReading(round(latency_ms, 3), round(s9, 4), state, t_min,
                           round(pred, 3) if pred is not None and math.isfinite(pred) else None,
                           round(latency_ms - pred, 3) if pred is not None and math.isfinite(pred) else None,
                           notes)


if __name__ == "__main__":
    import random
    print("Synthetic demonstration — a system whose latency follows the specified")
    print("curve, to show the fit recovers the parameters.\n")
    # λ must be small for the curve to stay physical.
    TRUE_TMIN, TRUE_ALPHA, TRUE_LAM = 12.0, 4.0, 0.015
    random.seed(11)
    fold = TheFold()
    for _ in range(40):
        s9 = random.uniform(0.02, 0.95)
        true_ms = TRUE_TMIN + TRUE_ALPHA * math.exp(TRUE_LAM / (s9 + EPSILON))
        fold.record(true_ms * random.uniform(0.94, 1.06), s9)
    f = fold.fit()
    print(f"  fitted: {f.fitted}   physical: {f.physical}")
    print(f"  t_min  {f.t_min:>8.3f}   (true {TRUE_TMIN})")
    print(f"  alpha  {f.alpha:>8.3f}   (true {TRUE_ALPHA})")
    print(f"  lambda {f.lam:>8.4f}   (true {TRUE_LAM})")
    print(f"  median |residual| {f.median_abs_residual_ms:.2f} ms (worst {f.max_abs_residual_ms:.2f})")

    print("\nState classification:\n")
    print(f"  {'latency':>9} {'s₉':>6}  state")
    print("  " + "-" * 44)
    for lat, s9 in [(13.0, 0.90), (14.5, 0.75), (60.0, 0.15), (250.0, 0.05), (40.0, 0.45), (13.5, 0.10)]:
        print(f"  {lat:>9.1f} {s9:>6.2f}  {fold.classify(lat, s9).state}")

    print("\nA fit that is numerically fine but physically impossible:\n")
    bad = TheFold()
    for _ in range(40):
        s9 = random.uniform(0.02, 0.95)
        bad.record(12.0 + 0.9 * math.exp(0.55 / (s9 + EPSILON)), s9)
    bf = bad.fit()
    print(f"  fitted: {bf.fitted}   physical: {bf.physical}   λ = {bf.lam}")
    print(f"  {bf.reason[:120]}...")
