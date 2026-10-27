"""
homeostasis.py — s₁₅, measured as distance to H_o rather than step size.

WHAT CHANGED AND WHY IT MATTERS:

  The Dimensional Derivation Table gives s₁₅ as:

      C = ‖X_{t+1} − X_t‖_g / ‖X_t‖_g
      "between successive hidden states; monitored per inference step"

  That is a STEP-SIZE measure. It tells you the system is settling
  somewhere. It does not tell you WHERE.

  This matters concretely, and the failure case is one the SCR proposal
  already cares about: a model stuck in a repetition attractor has tiny
  step-to-step change. Under C alone it looks like it is converging
  beautifully. It is parked in the worst available state.

  Once H_o has coordinates, a stronger measure becomes available:

      D = ‖X_t − H_o‖_g

  Distance to the target state, not distance travelled. A repetition loop
  scores near-zero on C and badly on D, which is the correct answer.

  Both are computed here. C is retained because it is what the table
  specifies and because the two together are more informative than either
  alone -- see the interpretation table in HomeostasisReading.

H_o -- THE COORDINATES:

  H_o is the homeostatic baseline: the state the system rests in when
  nothing is wrong. Its coordinates, as specified:

      friction (s₅ / lam)  -> at baseline. lam = 1.0 means H equals X_base,
                             which is the definition of resting load. Not
                             zero -- F cannot be zero, since X_base > 0.
                             "No friction" operationally means "no friction
                             ABOVE baseline," i.e. lam = 1.0.

      t₂ (Macro-Flow)    -> HIGH. Long unbroken cognitive cycles. At rest
                             nothing interrupts, so cycle duration
                             approaches its ceiling.

      t₅ (Sim. Width)    -> at the model's MAXIMUM sampling width. At rest
                             the possibility field is widest. This is also
                             consistent with s₆ (Orbital Vacuum), which
                             widens search during idle drift.

  NOTE ON t₅ AND INFINITY: t₅ cannot be unbounded. An unbounded coordinate
  makes ‖X_t − H_o‖ non-convergent by construction -- the distance never
  settles because one axis never settles. t₅_max is therefore the runtime's
  actual configured ceiling (beam width, thread-pool size, rollout budget),
  which matches the table's own measurement protocol for t₅ and is finite
  and checkable.

NORMALIZATION:
  The three coordinates have incompatible units -- F is a dimensionless
  ratio, t₂ is seconds, t₅ is a count. Summing them directly would let
  whichever has the largest raw magnitude dominate the distance regardless
  of which is actually furthest from rest. Each axis is therefore scaled to
  a dimensionless [0, 1] deviation from its H_o value before the norm is
  taken. The scaling ranges are CALIBRATION CHOICES and are stated as such.

ON THE LYAPUNOV / LaSALLE QUESTION -- READ THIS BEFORE CITING CONVERGENCE:

  With H_o given coordinates, a candidate Lyapunov function has an obvious
  form:

      V(X) = ‖X − H_o‖²_g

  This is positive definite by construction and zero only at H_o, which
  satisfies two of the three conditions LaSalle's invariance principle
  requires. The third -- V̇ ≤ 0 along system trajectories -- is NOT
  established by anything in this module.

  Measuring V and watching it decrease in observed sessions is EVIDENCE,
  not PROOF. A function can decrease across every run you happened to
  observe and still fail to decrease in general. Establishing V̇ ≤ 0
  requires an argument about the dynamics themselves, not a collection of
  traces.

  This module therefore computes V and reports whether it decreased over
  the observed window, explicitly labeled as an empirical observation. Any
  document claiming "global asymptotic convergence is confirmed" needs the
  analytical argument, not this output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class HomeostasisPoint:
    """A single observed system state, in H_o coordinates."""
    lam: float               # load index (s₅), dimensionless ratio
    t2_seconds: float      # macro-pulse cycle duration
    t5_width: int          # active sampling width


@dataclass
class HomeostasisReading:
    D: float                    # ‖X_t − H_o‖_g -- distance to homeostasis
    C: float                    # ‖X_{t+1} − X_t‖_g / ‖X_t‖_g -- table's step-size measure
    V: float                    # candidate Lyapunov value, D²
    V_decreasing: bool | None    # observed trend; None until 2+ samples exist
    axis_deviations: dict        # per-axis contribution, so D is legible not opaque
    dominant_axis: str
    interpretation: str


class HomeostasisMonitor:
    """
    Measures distance to H_o and the table's step-size ratio together.

    Usage:
        # 5400s = 1.5 hours, measured on the reference setup as the longest
    # observed stretch of continuous back-and-forth without a real break.
    mon = HomeostasisMonitor(t2_max_seconds=5400.0, t5_max_width=8)
        mon.observe(HomeostasisPoint(lam=0.7, t2_seconds=220.0, t5_width=1))
        reading = mon.observe(HomeostasisPoint(lam=0.2, t2_seconds=1620.0, t5_width=4))
    """

    def __init__(
        self,
        t2_max_seconds: float,
        t5_max_width: int,
        lam_scale: float = 1.0,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        t2_max_seconds: the longest uninterrupted cycle this deployment
            realistically sustains -- the longest stretch of continuous
            back-and-forth WITHOUT a real break, not total session wall
            time including pauses. H_o sits AT this value. Measured at 5400s
            (1.5 hours) on the reference setup. An invented number here
            silently distorts every distance, since t₂ deviation is computed
            relative to it.

        t5_max_width: the runtime's configured maximum sampling width. H_o
            sits AT this value. See the module docstring on why this must be
            finite.

        lam_scale: what Λ value counts as full-scale load deviation
            deviation. Default 1.0 means lam = 2.0 (the s₁₀ threshold) reads as
            a deviation of 1.0 on that axis. Calibration choice.

        weights: per-axis weights in the norm. Equal by default. These are a
            calibration choice, and a deliberate one: making them unequal is
            a claim that one axis matters more to homeostasis than another,
            which should be justified rather than assumed.
        """
        if t2_max_seconds <= 0 or t5_max_width <= 0:
            raise ValueError(
                "t2_max_seconds and t5_max_width must be positive and finite. "
                "An unbounded coordinate makes distance-to-H_o non-convergent "
                "by construction -- see the module docstring."
            )
        self.t2_max = t2_max_seconds
        self.t5_max = t5_max_width
        self.lam_scale = lam_scale
        self.weights = weights
        self._history: list[HomeostasisPoint] = []
        self._V_history: list[float] = []

    # ------------------------------------------------------------------
    def _deviations(self, p: HomeostasisPoint) -> dict[str, float]:
        """
        Per-axis dimensionless deviation from H_o, each in [0, 1] (F may
        exceed 1 under extreme load, which is meaningful and left uncapped).

        H_o coordinates: lam = 1.0, t2 = t2_max, t5 = t5_max.
        """
        # Friction: deviation ABOVE baseline. Below-baseline F (system
        # quieter than its calibrated rest state) is not a departure from
        # homeostasis, so it floors at 0 rather than counting as distance.
        f_dev = max(0.0, p.lam / self.lam_scale)

        # t2: shortfall from the maximum sustainable cycle. Short cycles
        # mean frequent interruption, which is a departure from rest.
        t2_dev = max(0.0, (self.t2_max - p.t2_seconds) / self.t2_max)

        # t5: shortfall from maximum sampling width. A narrowed possibility
        # field means the system is constrained, not resting.
        t5_dev = max(0.0, (self.t5_max - p.t5_width) / self.t5_max)

        return {"friction": f_dev, "t2_cycle": t2_dev, "t5_width": t5_dev}

    def _distance(self, p: HomeostasisPoint) -> tuple[float, dict[str, float]]:
        dev = self._deviations(p)
        w_f, w_t2, w_t5 = self.weights
        squared = (
            w_f * dev["friction"] ** 2
            + w_t2 * dev["t2_cycle"] ** 2
            + w_t5 * dev["t5_width"] ** 2
        )
        return math.sqrt(squared), dev

    def _step_ratio(self) -> float:
        """
        The table's C = ‖X_{t+1} − X_t‖_g / ‖X_t‖_g, computed over the same
        normalized coordinates so it is comparable with D rather than being
        on an unrelated scale.
        """
        if len(self._history) < 2:
            return 0.0
        prev, curr = self._history[-2], self._history[-1]
        d_prev = self._deviations(prev)
        d_curr = self._deviations(curr)
        keys = ("friction", "t2_cycle", "t5_width")

        step = math.sqrt(sum((d_curr[k] - d_prev[k]) ** 2 for k in keys))
        magnitude = math.sqrt(sum(d_prev[k] ** 2 for k in keys))
        return step / magnitude if magnitude > 1e-12 else 0.0

    # ------------------------------------------------------------------
    def observe(self, point: HomeostasisPoint) -> HomeostasisReading:
        self._history.append(point)
        D, dev = self._distance(point)
        V = D ** 2
        self._V_history.append(V)

        C = self._step_ratio()

        V_decreasing = None
        if len(self._V_history) >= 2:
            V_decreasing = self._V_history[-1] <= self._V_history[-2]

        dominant = max(dev, key=dev.get) if any(dev.values()) else "none"

        # The joint interpretation -- this is the whole reason both measures
        # are computed rather than just the one the table specifies.
        settled = C < 0.05
        near_home = D < 0.15
        if settled and near_home:
            interpretation = "AT HOMEOSTASIS — settled, and settled at H_o."
        elif settled and not near_home:
            interpretation = (
                f"STUCK — motion has stopped but the system is not at H_o "
                f"(D={D:.3f}, dominant deviation: {dominant}). This is the case "
                f"the table's C measure alone cannot detect: a repetition "
                f"attractor looks converged under C and is not."
            )
        elif not settled and near_home:
            interpretation = "APPROACHING — near H_o and still moving."
        else:
            interpretation = (
                f"IN TRANSIT — moving, and far from H_o "
                f"(D={D:.3f}, dominant deviation: {dominant})."
            )

        return HomeostasisReading(
            D=round(D, 4),
            C=round(C, 4),
            V=round(V, 4),
            V_decreasing=V_decreasing,
            axis_deviations={k: round(v, 4) for k, v in dev.items()},
            dominant_axis=dominant,
            interpretation=interpretation,
        )

    def lyapunov_trace(self) -> list[float]:
        """
        The full V history, for plotting or for checking monotonicity across
        a session. Monotone decrease here is EVIDENCE consistent with a
        Lyapunov function, not proof that one exists -- see module docstring.
        """
        return list(self._V_history)


if __name__ == "__main__":
    # 5400s = 1.5 hours, measured on the reference setup as the longest
    # observed stretch of continuous back-and-forth without a real break.
    mon = HomeostasisMonitor(t2_max_seconds=5400.0, t5_max_width=8)

    print("Trajectory: system under load, recovering toward H_o\n")
    print(f"{'Λ':<6} {'t2':<7} {'t5':<5} {'D':<8} {'C':<8} {'V':<8} interpretation")
    print("-" * 100)
    # Λ values on the DEVIATION scale (0 = at baseline, ~1 = fully loaded)
    trajectory = [
        (0.82, 140.0, 1),
        (0.65, 360.0, 2),
        (0.44, 1080.0, 3),
        (0.26, 2520.0, 5),
        (0.09, 4320.0, 7),
        (0.00, 5310.0, 8),
    ]
    for lam, t2, t5 in trajectory:
        r = mon.observe(HomeostasisPoint(lam=lam, t2_seconds=t2, t5_width=t5))
        print(f"{lam:<6} {t2:<7} {t5:<5} {r.D:<8} {r.C:<8} {r.V:<8} {r.interpretation[:52]}")

    print(f"\nV trace: {[round(v, 3) for v in mon.lyapunov_trace()]}")
    print("Monotone decreasing across this run — consistent with a Lyapunov")
    print("function, but NOT proof one exists. See module docstring.\n")

    print("=" * 100)
    print("The case C alone cannot detect: a repetition attractor.\n")
    stuck = HomeostasisMonitor(t2_max_seconds=5400.0, t5_max_width=8)
    print(f"{'Λ':<6} {'t2':<7} {'t5':<5} {'D':<8} {'C':<8} interpretation")
    print("-" * 100)
    # High friction, short cycles, collapsed sampling width -- and barely moving.
    for _ in range(4):
        r = stuck.observe(HomeostasisPoint(lam=0.78, t2_seconds=110.0, t5_width=1))
        print(f"{2.2:<6} {6.0:<7} {1:<5} {r.D:<8} {r.C:<8} {r.interpretation[:60]}")

    print()
    print("C reads ~0.0 — by the table's measure this looks fully converged.")
    print("D reads high and flags it correctly as stuck away from H_o.")
