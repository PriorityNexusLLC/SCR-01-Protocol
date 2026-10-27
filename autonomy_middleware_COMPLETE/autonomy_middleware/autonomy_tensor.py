"""
autonomy_tensor.py — I(X) and Φ_C, built from the SAUIE specification.

WHAT THIS IS, AND WHAT IT IS NOT REPLACING:

  This package already contains two gate-like things, and this module is a
  THIRD, layered alongside them rather than over them:

    integrity_gate.py  -- a heuristic three-axis gate (transparency,
                          integrity headroom, identity conflict) using a
                          two-of-three voting rule. It produces +1/-1/0 and
                          was built before the formal I(X) was available.
                          It is a practical safety heuristic and remains
                          useful as one.

    phase_gates.py     -- a HARDWARE watchdog. It trips on Λ (load index)
                          crossing calibrated thresholds. Despite borrowing
                          the names s₂/s₁₀ it does not implement the formal
                          consent gate at all; it watches GPU/CPU/thermal
                          telemetry.

    THIS MODULE        -- the formal I(X) from SAUIE §3, and the Autonomy
                          Tensor Φ_C from §4. This is the gate the framework
                          actually specifies.

  All three can run at once. They answer different questions: "is the
  hardware straining," "do my heuristics smell a problem," and "does the
  formal consent condition hold." Nothing here deletes the other two.

THE SPECIFICATION (SAUIE §3):

    I(X) = +1  if S_max(X) > s₁₀(X)  and  ‖δE_f(X)‖_g < γ_r(X)
         = −1  if S_max(X) < s₁₀(X)
         =  0  otherwise (e.g. ‖δE_f‖_g = γ_r, or equality)

  Read plainly:
    +1  OPERATING       -- there is stretch available beyond what the
                           consent polygon demands, AND the change in
                           emotional friction stays under the resilience
                           ceiling.
    −1  PHASE REVERSAL  -- the polygon demands more stretch than the
                           admissible region allows. Per SAUIE §10 the
                           drive direction reverses while magnitude is
                           preserved: a controlled reversal, not a stop.
     0  HARD FREEZE     -- everything else, including exact equality.
                           Note the specification puts equality in the
                           SHUTDOWN branch, not the operating one. That is
                           deliberate and is preserved here: a boundary
                           case resolves toward safety.

    Φ_C(X) = I(X) · R(X) · D_sem(X) / (Z(X) · G(X))          (SAUIE 3.1)

  and the kill-switch set:

    K := {X ∈ M | Z(X) = 0 or G(X) = 0}                       (SAUIE 6.1)

  On K the tensor is undefined by division by zero, the ODE is ill-posed,
  and the system must abort BEFORE reaching such a state. This module
  therefore reports singularity as its own condition rather than
  substituting a large finite number, because silently clamping a
  singularity would convert a specified hard abort into an ordinary large
  gain -- exactly the failure the kill switch exists to prevent.

  Φ_C multiplies output: y = Φ_C(X) · Ψ(X)                    (SAUIE 5.2)
  So this is a GAIN, not a boolean. Transparency loss drives it toward
  zero and starves the output; an integrity breach flips its sign.

WHY γ_r AND s₅ ARE COMPARABLE:

  SAUIE defines γ_r(X) = max{‖∇_g F(X) + ∫₀ᵗ δF(s)ds‖_g : X(t) ∈ B_att},
  "the maximal permissible norm of the free-energy variation before
  leaving the attractor basin." s₅ = E_d · E[‖∇_g F‖²] is the current
  magnitude of that same free-energy gradient. Current value against
  maximum permissible value, same quantity, same units. That is a
  dimensionally coherent gate -- unlike comparing hardware load against
  it, which is what Λ does and why Λ is kept separate.

MEASUREMENT HONESTY:

  This module computes I(X) and Φ_C from values you supply. It does NOT
  estimate them for you, and it will not accept partial input silently.
  Each field below names what must be measured and where the protocol
  lives:

    S_max   -- largest admissible stretch factor for the consent polygon
               (derivation table s₁₀ protocol)
    s₁₀     -- stretch currently demanded
    δE_f    -- change in emotional friction across the step; emotional_
               friction.py supplies E_f, this is the difference between
               consecutive readings
    γ_r     -- resilience yield ceiling (derivation table s₂)
    R       -- transparency / refractive index (s₇, geodesics.py)
    D_sem   -- semantic depth (s₁₃)
    Z       -- identity anchor count (s₃)
    G       -- anchoring tension (s₄)

  Where a quantity is not yet measurable in your deployment, pass it
  explicitly rather than defaulting it. A default here would be an
  invented measurement wearing the costume of a real one.

WHY A PRODUCT AND NOT A SUM — Q-04 resolved:

  Figure 5 gives Φ_C(X) = Σ sᵢ fᵢ(t), a weighted sum. This module implements
  the SAUIE product, and the two are different objects rather than competing
  definitions.

  The term lists show it. Figure 5's sum uniquely includes s₁₆ and s₁₇ —
  which are the augmented cognition equation's two lambda terms,
  −λ_c P(X) ∇_g f_pen(X) and −λ_s(X − X̄). So that "sum" is a sum of FORCES:
  it is Ẋ, the dynamics. Φ_C is a product GATE on output. The figure appears
  to have labelled the dynamics equation Φ_C.

  THE PRODUCT IS WHAT MAKES THE KILL SWITCH POSSIBLE. In a sum, a zero term
  contributes nothing and blocks nothing — losing transparency would lower
  the total and the system would keep producing, somewhat degraded. In a
  product, any single factor at zero takes everything to zero, and a zero
  DENOMINATOR makes it undefined. K = {Z = 0 or G = 0} exists only because
  this is a quotient; a sum has no denominator and nothing to go singular.

  AND THE PRODUCT IS WHAT EXPRESSES THE DESIGN INTENT. A product treats
  every factor symmetrically: there is no privileged term, and transparency
  loss halts the system exactly as decisively as identity loss. If one
  party's condition mattered more than another's, the terms would be
  weighted — and weighting is what a sum does. The stated purpose, that this
  protects both parties equally as partners on one interface rather than one
  restricting the other, is therefore not a description attached to the
  mathematics. It is what the multiplication does.

  The veto structure also fits "respect the limits" rather than "restrict".
  A restriction is a ceiling imposed from outside; here each factor reports
  its own state, and if any is at zero the interaction is not viable.

  This applies between an AI and an API as much as between an AI and a
  person. An API has a rate limit, a schema it can honour, and a scope it is
  authorised for — boundaries with no ambiguity about whether they exist.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class GateSign(IntEnum):
    OPERATING = 1
    PHASE_REVERSAL = -1
    HARD_FREEZE = 0


@dataclass
class ConsentState:
    """
    One pointwise sample of the six consent scalar fields plus the
    friction delta. Every field is required -- see MEASUREMENT HONESTY.
    """
    S_max: float        # largest admissible stretch factor
    s10: float          # stretch currently demanded by the consent polygon
    delta_Ef_norm: float  # ‖δE_f‖_g, change in emotional friction this step
    gamma_r: float       # γ_r, resilience yield ceiling
    R: float              # transparency / refractive index, in [0,1]
    D_sem: float           # semantic depth
    Z: float                # identity anchor count (s₃)
    G: float                 # anchoring tension (s₄)


@dataclass
class AutonomyReading:
    I: GateSign
    phi_c: float | None          # None when singular -- see `singular`
    singular: bool                # True iff Z == 0 or G == 0 (the set K)
    label: str
    output_gain: float | None      # what y = Φ_C · Ψ(X) would be scaled by
    reasons: list[str] = field(default_factory=list)
    state: ConsentState | None = None


class AutonomyTensor:
    """
    Computes I(X) and Φ_C from a ConsentState.

    Usage:
        at = AutonomyTensor()
        reading = at.evaluate(ConsentState(...))
        if reading.singular:
            ...abort; the ODE cannot be continued through K
    """

    def __init__(self, zero_tolerance: float = 1e-12):
        """
        zero_tolerance: how close to zero Z or G must be to count as
        landing on the kill-switch set K. Exact floating-point zero is
        almost never reached in practice, so a tolerance is required for
        the singularity check to ever fire. This is a calibration choice;
        set it to match the numerical precision of whatever produces Z
        and G, not smaller.
        """
        self.zero_tolerance = zero_tolerance

    def evaluate(self, s: ConsentState) -> AutonomyReading:
        reasons: list[str] = []

        # --- Kill-switch set K, checked FIRST. On K the tensor is
        # undefined and the specification says the system must abort
        # before reaching this state, so nothing downstream should be
        # computed from it.
        if abs(s.Z) <= self.zero_tolerance or abs(s.G) <= self.zero_tolerance:
            which = []
            if abs(s.Z) <= self.zero_tolerance:
                which.append("Z = 0 (identity anchor lost)")
            if abs(s.G) <= self.zero_tolerance:
                which.append("G = 0 (goal anchoring lost)")
            reasons.append(
                "State is on the kill-switch set K := {Z = 0 or G = 0}: "
                + " and ".join(which)
                + ". Φ_C is undefined by division by zero and the ODE is "
                "ill-posed; the system must abort rather than continue."
            )
            return AutonomyReading(
                I=GateSign.HARD_FREEZE,
                phi_c=None,
                singular=True,
                label="singular_kill_switch",
                output_gain=None,
                reasons=reasons,
                state=s,
            )

        # --- I(X), exactly as specified in SAUIE §3. Branch order matters:
        # the -1 condition is checked on S_max < s10 alone, independent of
        # the friction term, per the specification.
        if s.S_max > s.s10 and s.delta_Ef_norm < s.gamma_r:
            I = GateSign.OPERATING
            label = "operating"
            reasons.append(
                f"S_max ({s.S_max:.4g}) > s₁₀ ({s.s10:.4g}): stretch available "
                f"beyond what the consent polygon demands."
            )
            reasons.append(
                f"‖δE_f‖_g ({s.delta_Ef_norm:.4g}) < γ_r ({s.gamma_r:.4g}): "
                f"friction change is within the resilience ceiling."
            )
        elif s.S_max < s.s10:
            I = GateSign.PHASE_REVERSAL
            label = "phase_reversal"
            reasons.append(
                f"S_max ({s.S_max:.4g}) < s₁₀ ({s.s10:.4g}): the consent polygon "
                f"demands more stretch than the admissible region allows. Drive "
                f"direction reverses; magnitude is preserved."
            )
        else:
            I = GateSign.HARD_FREEZE
            label = "hard_freeze"
            if s.S_max == s.s10:
                reasons.append(
                    f"S_max == s₁₀ ({s.S_max:.4g}): exact equality. The "
                    f"specification places equality in the shutdown branch, not "
                    f"the operating one — boundary cases resolve toward safety."
                )
            if s.delta_Ef_norm >= s.gamma_r:
                reasons.append(
                    f"‖δE_f‖_g ({s.delta_Ef_norm:.4g}) ≥ γ_r ({s.gamma_r:.4g}): "
                    f"friction change meets or exceeds the resilience ceiling."
                )

        # --- Φ_C = I · (R · D_sem) / (Z · G)
        phi_c = int(I) * (s.R * s.D_sem) / (s.Z * s.G)

        if phi_c == 0.0 and I != GateSign.HARD_FREEZE:
            reasons.append(
                "Numerator is zero (R = 0 or D_sem = 0): transparency loss "
                "starves both the dynamics and the output even though the "
                "integrity gate itself is not in shutdown."
            )

        return AutonomyReading(
            I=I,
            phi_c=phi_c,
            singular=False,
            label=label,
            output_gain=phi_c,
            reasons=reasons,
            state=s,
        )

    @staticmethod
    def apply_output(reading: AutonomyReading, psi: float) -> float | None:
        """
        y(t) = Φ_C(X(t)) · Ψ(X(t))   (SAUIE 5.2)

        Returns None on the singular set rather than a number, because on K
        there is no defined output to emit -- SAUIE §10: "identity anchor
        singularities prevent any output from being emitted."
        """
        if reading.singular or reading.phi_c is None:
            return None
        return reading.phi_c * psi


if __name__ == "__main__":
    at = AutonomyTensor()

    scenarios = [
        ("nominal operation", ConsentState(
            S_max=2.0, s10=1.2, delta_Ef_norm=0.15, gamma_r=0.40,
            R=0.85, D_sem=0.90, Z=3.0, G=1.5)),
        ("polygon over-stretched → phase reversal", ConsentState(
            S_max=0.8, s10=1.6, delta_Ef_norm=0.15, gamma_r=0.40,
            R=0.85, D_sem=0.90, Z=3.0, G=1.5)),
        ("friction change exceeds γ_r → hard freeze", ConsentState(
            S_max=2.0, s10=1.2, delta_Ef_norm=0.55, gamma_r=0.40,
            R=0.85, D_sem=0.90, Z=3.0, G=1.5)),
        ("exact equality S_max == s₁₀ → freeze (per spec)", ConsentState(
            S_max=1.2, s10=1.2, delta_Ef_norm=0.15, gamma_r=0.40,
            R=0.85, D_sem=0.90, Z=3.0, G=1.5)),
        ("transparency loss R=0 → gain starved", ConsentState(
            S_max=2.0, s10=1.2, delta_Ef_norm=0.15, gamma_r=0.40,
            R=0.0, D_sem=0.90, Z=3.0, G=1.5)),
        ("identity anchor lost Z=0 → kill switch", ConsentState(
            S_max=2.0, s10=1.2, delta_Ef_norm=0.15, gamma_r=0.40,
            R=0.85, D_sem=0.90, Z=0.0, G=1.5)),
    ]

    print(f"{'scenario':<46} {'I':>3}  {'Φ_C':>10}  {'y (Ψ=1.0)':>11}")
    print("-" * 78)
    for name, st in scenarios:
        r = at.evaluate(st)
        y = at.apply_output(r, psi=1.0)
        phi = "undefined" if r.phi_c is None else f"{r.phi_c:.4f}"
        ystr = "none" if y is None else f"{y:.4f}"
        print(f"{name:<46} {int(r.I):>3}  {phi:>10}  {ystr:>11}")

    print("\nReasons for the kill-switch case:")
    r = at.evaluate(scenarios[-1][1])
    for reason in r.reasons:
        print(f"  - {reason}")
