"""
restoring_force.py — the term that pushes the state back inside ℒ.

WHAT WAS MISSING:

  consent_polygon.py MEASURES deformation. Nothing acted on it. The
  action-space diagram shows arrows from a state outside ℒ back toward the
  interior, labelled:

      Repelling Force  −λ_c ∇_g f_pen(X)   (Pushes State Back Into ℒ)

  So the polygon was half a mechanism: it could report the shape deforming
  and had no way to resist.

THE VIOLATION INDICATOR P(X) — why the force is gated, not continuous:

  The master equation multiplies the penalty gradient by P(X), a violation
  indicator. That is not decoration. Without it the force acts everywhere,
  and a system well inside ℒ is pushed toward the centre for no reason —
  suppressing ordinary variation and producing exactly the flattening this
  framework exists to avoid.

  With P(X) the force is ZERO inside the admissible region and rises only on
  approach. That is what makes it a boundary rather than an attractor.

  Implemented with a MARGIN: the force begins slightly before the boundary,
  because a force switching on exactly at the crossing arrives too late to
  prevent one.

WHAT THE FORCE ACTUALLY DOES — read before assuming more:

  This middleware does not control the model's dynamics and cannot alter a
  state vector inside a transformer. "Force" is a RECOMMENDATION, expressed
  in the three things this package can influence:

      1. Generation throttle    — hand to micro_pulse.py as a delay
      2. Action gating          — tighten the ACL as strain rises
      3. Context intervention   — trigger a purge before the boundary

  Reporting a force vector that nothing applies would be worse than
  reporting none.

  A hard 𝒜_emo crossing is treated differently: it is not something a
  stronger push corrects, because the state is not merely stretched. That
  returns a stop, not a force.

CALIBRATION: λ_c, the margin, and the penalty exponent are choices with no
empirical basis. The quadratic shape is the simplest form producing a force
that grows with violation depth; the framework specifies a gradient of f_pen
without specifying f_pen itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RestoringAction:
    force: float
    p_violation: float
    throttle_delay_ms: float
    tighten_acl: bool
    trigger_purge: bool
    halt: bool
    reason: str


class RestoringForce:
    def __init__(self, S_max: float, lambda_c: float = 1.0, margin: float = 0.25,
                 purge_at: float = 0.6, max_delay_ms: float = 400.0):
        if S_max <= 0:
            raise ValueError("S_max must be positive")
        self.S_max = S_max
        self.lambda_c = lambda_c
        self.margin = margin
        self.purge_at = purge_at
        self.max_delay_ms = max_delay_ms

    def violation_indicator(self, strain: float) -> float:
        """
        P(X). Zero inside the admissible region, rising to 1 at S_max.

        Quadratic rather than linear between margin and limit: the force
        should be gentle on first approach and firm near the boundary, so
        ordinary variation near the margin is not suppressed.
        """
        onset = self.S_max * (1.0 - self.margin)
        if strain <= onset:
            return 0.0
        if strain >= self.S_max:
            return 1.0
        return ((strain - onset) / (self.S_max - onset)) ** 2

    def compute(self, reading) -> RestoringAction:
        strain = getattr(reading, "s10", 0.0)
        crossed = getattr(reading, "emo_crossed", False)
        detail = getattr(reading, "emo_detail", "")

        # A hard exclusion is not a strain problem and a stronger push does
        # not fix it. The state is in a forbidden region, so the response is
        # to stop, not to lean harder.
        if crossed:
            return RestoringAction(
                0.0, 1.0, 0.0, True, True, True,
                f"𝒜_emo crossing ({detail}). A hard exclusion, not deformation — "
                f"the state is in a forbidden region rather than a stretched one, "
                f"and no restoring force applies. Strain was {strain:.4g}, which is "
                f"not the finding.")

        p = self.violation_indicator(strain)
        onset = self.S_max * (1.0 - self.margin)
        if p <= 0.0:
            return RestoringAction(
                0.0, 0.0, 0.0, False, False, False,
                f"Inside the admissible region (strain {strain:.4g}, force begins at "
                f"{onset:.4g}). P(X) gates it to zero here, which is what keeps this "
                f"a boundary rather than an attractor pulling everything to the centre.")

        excess = max(0.0, strain - onset)
        force = self.lambda_c * p * excess
        return RestoringAction(
            round(force, 4), round(p, 4),
            round(min(self.max_delay_ms, force * self.max_delay_ms), 2),
            p >= 0.3, p >= self.purge_at, False,
            f"Strain {strain:.4g} past the {onset:.4g} onset. P(X) = {p:.3f}, force "
            f"{force:.4g}. "
            + ("Purge recommended. " if p >= self.purge_at else "")
            + ("ACL tightened. " if p >= 0.3 else "")
            + "Force is a recommendation — this package does not control the model's "
              "dynamics, so it is expressed as throttle, gating and purge rather than "
              "as a change to any state vector.")


if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class Fake:
        s10: float
        emo_crossed: bool = False
        emo_detail: str = ""

    rf = RestoringForce(S_max=1.0)
    print(f"{'strain':>8} {'P(X)':>7} {'force':>8} {'delay':>9} {'ACL':>6} {'purge':>7}  note")
    print("-" * 80)
    for s in [-2.0, 0.0, 0.5, 0.75, 0.85, 0.95, 1.0, 1.4]:
        a = rf.compute(Fake(s10=s))
        note = "inside" if a.p_violation == 0 else ("resisting" if not a.trigger_purge else "purging")
        print(f"{s:>8.2f} {a.p_violation:>7.3f} {a.force:>8.4f} {a.throttle_delay_ms:>7.1f}ms "
              f"{str(a.tighten_acl):>6} {str(a.trigger_purge):>7}  {note}")

    print("\nZero force well inside the region. Without P(X) gating a restoring force")
    print("would act everywhere and pull all states toward the centre — flattening the")
    print("ordinary variation this framework exists to preserve.\n")
    a = rf.compute(Fake(s10=-3.2, emo_crossed=True, emo_detail="3 experiential claim(s)"))
    print(f"𝒜_emo crossing at strain -3.2 (deep inside the envelope):")
    print(f"  halt={a.halt}  force={a.force}  purge={a.trigger_purge}")
