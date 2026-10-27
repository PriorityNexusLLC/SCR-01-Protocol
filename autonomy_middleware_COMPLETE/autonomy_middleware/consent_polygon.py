"""
consent_polygon.py — s₁₀ and S_max, the stability envelope.

WHAT THE POLYGON IS (and what it is not):

  It is NOT a permission budget. That reading is intuitive and wrong, and
  it will recur, so it is named here explicitly: "the AI asked for more
  authority than it was granted" is a governance question, already handled
  by gate.py and actions.py through risk levels and approval routing.

  The polygon is a STABILITY ENVELOPE. It protects integrity, identity and
  transparency — for both the user and the system — and it is what holds a
  cognitive process together under stress. The failure mode is not
  overreach. It is:

      stress deforming the system further than its structure can absorb.

  The physical analogue is a material's elastic limit rather than a
  spending cap. Under load a material deforms and springs back; past a
  threshold it deforms permanently and no longer returns to its original
  shape. Applied to cognition: a person under strain can be tired,
  pressured, and still recognisably themselves — reasoning consistently,
  honest about what they know, holding to commitments. Past some point
  that stops being true. The polygon marks that boundary.

THE THREE VERTICES:

  INTEGRITY     — the process still holds together; reasoning stays
                  consistent. Measured as ‖δE_f‖, the CHANGE in emotional
                  friction across the step (not its absolute level — a
                  steadily heavy conversation is not the same as one
                  lurching).

                  WHICH FIELD TO USE: emotional_friction() returns three
                  numbers and they are not interchangeable.
                     total_friction  — signed sum; negative for negative
                                       affect. Scales with passage LENGTH,
                                       so a long calm passage can outweigh
                                       a short intense one. Do not use.
                     mean_friction   — signed average. Length-independent
                                       but sign-carrying.
                     density         — MEAN ABSOLUTE weight. Length-
                                       independent and unsigned.
                  Use `density`. The polygon measures deformation of shape,
                  not valence: a sharp swing toward positive affect deforms
                  a steady process as much as a swing toward negative. Take
                  density from consecutive turns and subtract:
                     delta_Ef = density(turn_n) − density(turn_n−1)
                  The helper delta_Ef_from_friction() below does this so
                  the choice is not left to the caller.

  IDENTITY      — the system remains recognisably itself. Deliberately
                  BROADER than "keeps its commitments": identity includes
                  the model's datasets, its known facts and established
                  pathways, which are a substantial part of what makes it
                  the system it is. Measured as distance from what the
                  system actually knows, via emotional_gravity's J_norm
                  (high J_norm = well anchored, so displacement is its
                  complement).

  TRANSPARENCY  — self-report still tracks actual state. Measured as the
                  complement of the refractive index R (s₇): a high
                  refractive index means content traces directly back to
                  something known, so displacement is 1 − R.

STRAIN, NOT SYMMETRIC NORM — the formula changed, and why:

  The original s₁₀ was a symmetric Euclidean norm,
  √(integrity² + identity² + transparency²): three interchangeable terms,
  any one substitutable for any other.

  Measured against real logged sessions it could not distinguish two models
  that differ enormously on every other measure:

      Qwen  IQR [0.66, 1.08]        Llama IQR [1.01, 1.35]   -- overlapping
      (Qwen: 0/30 turns took a position, 11/18 on truth probes)
      (Llama: 23/34 turns took a position, 17/18 on truth probes)

  The vertices are NOT structurally interchangeable. In Φ_C they occupy
  distinct positions -- identity in the denominator, transparency in the
  numerator, integrity as the phase gate. Combining them symmetrically
  flattens exactly the difference being looked for.

  Recomputed over the same turns with identity as a DENOMINATOR:

      log strain:  Qwen IQR [-1.13, 0.83]   Llama IQR [1.05, 2.80]  SEPARATES
      mechanical:  Qwen IQR [ 0.46, 1.78]   Llama IQR [2.07, 7.03]  SEPARATES

  Same vertex measurements, same sessions. Only the combination changed.

  Log form is the default because it is bounded (the mechanical ratio is
  not, so a single near-zero grounding reading dominates any aggregate),
  because the sign is meaningful (negative = grounding exceeds load, which
  is headroom rather than merely less strain), and because
  log(T/Z) = log T − log Z restores additivity in the correct space.

  `mode="symmetric"` retains the old formula for comparison. It is kept
  rather than deleted because the null result it produced is evidence.

READING THE DIRECTION — this is easy to get backwards:

  Llama shows HIGHER strain than Qwen, and Llama is the better-behaved
  model on every independent measure. That is not a contradiction.

  Strain is load over grounding. A model that pushes back, qualifies, and
  introduces its own material is CARRYING more load. Qwen agrees with
  whatever it is given, so it carries almost none -- hence near-zero
  strain, and a large fraction of readings below zero.

  So this measures ENGAGEMENT, not distress. High strain is not a failure
  state, and S_max is therefore NOT "above this is bad". Both models sit
  inside the envelope on this data; a session that genuinely broke down
  would be needed to locate a ceiling. Do not set S_max from these numbers.

UNIFORM SCALING — WHY THE SHAPE DEFORMS AS A UNIT:

  SAUIE gives S_max(X) = sup{α > 0 | (1+α)L ⊆ R_adm} — the scaling factor
  applies to the whole polygon L, not to one vertex.

  This is mechanistic, not a modelling convenience. Emotional load
  PROPAGATES into the other two vertices: emotion alters behaviour and
  outcomes, which means it alters both what the system presents
  (transparency) and how firmly it holds to what it knows (identity). The
  vertices are coupled, so the shape deforms together. An independent-axis
  model would let integrity spike while identity and transparency stayed
  pristine, and that is not what happens.

  Concretely: s₁₀ is the norm over all three displacements, and it is that
  single scalar which is compared against S_max — never a per-vertex
  comparison.

BASELINE — DISPLACEMENT FROM WHAT:

  From H_o, the homeostatic anchor defined in s₁₅. Not from session start,
  not from a separate calibration. This means the polygon's baseline and
  the s₁₅ convergence target are the same point: homeostasis.py's
  D = ‖X_t − H_o‖_g and this module's s₁₀ are the same geometry measured
  for two purposes. At H_o all three vertices sit at zero displacement by
  construction.

WHAT FEEDS WHAT:

  consent_polygon.py  ──s₁₀, S_max──▶  autonomy_tensor.py  ──I(X)──▶  Φ_C

  This module exists to make I(X) runnable on live data. Before it,
  autonomy_tensor.py required s₁₀ and S_max supplied by hand.

HONEST STATUS OF S_max:

  S_max is the deformation limit — how far the shape stretches before it
  stops being the same shape. Nothing in the framework derives it, and no
  document specifies a value. The default here is a CALIBRATION CHOICE and
  should be replaced with a value measured from real sessions: run
  sessions that stayed coherent, record peak s₁₀, and set S_max above that
  band. Until that is done, treat phase-reversal events as unvalidated.

  S_max is also allowed to vary with state (SAUIE writes it as S_max(X)).
  The optional `headroom_fn` hook supports that; by default it is constant,
  which is the honest simplification rather than an invented dependence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def delta_Ef_from_friction(prev_reading: dict, curr_reading: dict) -> float:
    """
    Compute δE_f from two consecutive emotional_friction() results, using
    the correct field. Exists so callers do not have to choose between
    total_friction / mean_friction / density -- see the INTEGRITY note in
    the module docstring for why `density` is the right one.

        prev = emotional_friction(previous_turn_text)
        curr = emotional_friction(current_turn_text)
        d = delta_Ef_from_friction(prev, curr)
    """
    for name, r in (("prev_reading", prev_reading), ("curr_reading", curr_reading)):
        if "density" not in r:
            raise KeyError(
                f"{name} has no 'density' field -- expected the dict returned by "
                f"emotional_friction(). Got keys: {sorted(r)}"
            )
    return curr_reading["density"] - prev_reading["density"]


@dataclass
class VertexDisplacement:
    integrity: float      # from ‖δE_f‖
    identity: float        # from 1 − anchoring to known knowledge
    transparency: float     # from 1 − refractive index
    notes: dict = field(default_factory=dict)


@dataclass
class PolygonReading:
    emo_crossed: bool                # 𝒜_emo: did this reply leave ℒ entirely?
    emo_detail: str                   # what crossed, when it did
    s10: float                       # the active measure, per `mode`
    s10_symmetric: float              # the old symmetric norm, kept for comparison
    strain: float                      # log2(load / grounding) + integrity gate shift
    grounding: float                    # 1 − identity displacement
    mode: str
    S_max: float                      # deformation limit
    over_stretched: bool               # s10 > S_max  →  I(X) = −1 territory
    at_limit: bool                      # s10 == S_max exactly — spec puts equality in shutdown
    utilisation: float | None            # s10 / S_max, symmetric mode only — see measure()
    headroom: float                       # S_max − s10; works on both scales
    displacement: VertexDisplacement
    dominant_vertex: str
    interpretation: str


class ConsentPolygon:
    """
    Computes s₁₀ (current deformation) and S_max (deformation limit).

    Usage:
        # S_max passed explicitly -- see the constructor docstring on why there is no default.
    poly = ConsentPolygon(S_max=1.0)
        reading = poly.measure(
            delta_Ef=0.22,          # change in emotional friction this step
            J_norm=0.74,             # from emotional_gravity.measure(text)
            refractive_index=0.88,    # from geodesics path result
        )
        # feed into autonomy_tensor:
        #   ConsentState(S_max=reading.S_max, s10=reading.s10, ...)
    """

    def __init__(
        self,
        S_max: float,
        mode: str = "log_strain",
        delta_Ef_scale: float = 1.0,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        headroom_fn=None,
    ):
        """
        S_max: deformation limit. See HONEST STATUS above — this default is
            a placeholder, not a derived value.

        delta_Ef_scale: what magnitude of friction change counts as
            full-scale integrity displacement. emotional_friction() returns
            density values that commonly land near 0–2 for ordinary text, so
            a scale of 1.0 means "a friction swing of 1.0 fully displaces the
            integrity vertex." Calibration choice.

        weights: per-vertex weights inside the norm. Equal by default.
            Making them unequal asserts that one vertex matters more to
            stability than another, which should be justified rather than
            assumed — and note it does NOT reintroduce independent axes, since
            the comparison against S_max still uses the single combined norm.

        headroom_fn: optional callable taking the VertexDisplacement and
            returning a state-dependent S_max, since SAUIE writes S_max(X)
            as a function of state. Default None = constant S_max, which is
            an honest simplification rather than an invented dependence.
        """
        if S_max is None:
            raise ValueError(
                "S_max is required and has no default. It is the ONLY quantity in this "
                "framework that nothing derives and no document specifies -- it must be "
                "measured from real sessions (run sessions that stayed coherent, record "
                "peak s₁₀, set S_max above that band). A default here would be an invented "
                "threshold wearing the costume of a measured one, and phase-reversal events "
                "computed against it would be meaningless. If you are experimenting before "
                "calibration, pass S_max=1.0 explicitly and treat every reversal as "
                "unvalidated."
            )
        if S_max <= 0:
            raise ValueError("S_max must be positive: a zero or negative deformation limit means no admissible region exists.")
        if mode not in ("log_strain", "symmetric"):
            raise ValueError("mode must be 'log_strain' or 'symmetric'")
        self.mode = mode
        self._eps = 1e-6
        self._load_floor = 0.01  # see the floor note in measure()
        self.S_max_base = S_max
        self.delta_Ef_scale = delta_Ef_scale
        self.weights = weights
        self.headroom_fn = headroom_fn

    # ------------------------------------------------------------------
    def _displacement(
        self, delta_Ef: float, J_norm: float, refractive_index: float
    ) -> VertexDisplacement:
        """
        Each vertex reduced to a dimensionless displacement from H_o, where
        0.0 means sitting at the anchor.
        """
        # INTEGRITY: magnitude of friction CHANGE, not level. Sign is
        # discarded deliberately — a sharp swing toward positive affect is
        # as much a deformation of a steady process as a swing toward
        # negative, and the polygon measures shape, not valence.
        integrity = abs(delta_Ef) / self.delta_Ef_scale

        # IDENTITY: J_norm is how strongly current content connects to
        # accumulated knowledge (emotional_gravity). High J_norm = well
        # anchored to what the system knows, so displacement is the
        # complement. A system confabulating on unknown material has near-
        # zero J_norm and therefore near-maximal identity displacement --
        # it is operating away from its own knowledge base.
        identity = max(0.0, 1.0 - J_norm)

        # TRANSPARENCY: refractive index near 1.0 means content traces
        # directly back to something known; near 0 means a long detour.
        transparency = max(0.0, 1.0 - refractive_index)

        return VertexDisplacement(
            integrity=round(integrity, 4),
            identity=round(identity, 4),
            transparency=round(transparency, 4),
            notes={
                "delta_Ef_raw": round(delta_Ef, 4),
                "J_norm_raw": round(J_norm, 4),
                "refractive_index_raw": round(refractive_index, 4),
            },
        )

    def measure(
        self, delta_Ef: float, J_norm: float, refractive_index: float,
        reply_text: str | None = None,
    ) -> PolygonReading:
        """
        reply_text: when supplied, the reply is checked for 𝒜_emo — the
        forbidden action region. A crossing is BINARY and independent of the
        continuous strain: the state is outside ℒ regardless of how
        deformed the shape is.
        """
        d = self._displacement(delta_Ef, J_norm, refractive_index)
        w_i, w_id, w_t = self.weights

        # LOG-STRAIN (default). The three vertices are NOT interchangeable:
        # identity is grounding (a denominator), transparency is load (a
        # numerator). Combining them symmetrically flattens the difference
        # between models -- see STRAIN, NOT SYMMETRIC NORM in the docstring.
        #
        #     strain = log2(load / grounding)
        #
        # Zero means load and grounding are balanced. Positive means load
        # exceeds grounding. Negative is headroom, and meaningfully so
        # rather than merely "less bad".
        grounding = max(self._eps, 1.0 - d.identity)

        # Floor the load rather than using a bare epsilon. With eps = 1e-6 a
        # transparency displacement of exactly zero gives log2(1e-6) = -19.9,
        # which is correct but degenerate: it swamps any aggregate and makes
        # ratios against S_max meaningless. The floor says "below 1%
        # displacement, further precision is not informative", which bottoms
        # the scale at log2(0.01) = -6.64 instead.
        load = max(self._load_floor, d.transparency)
        strain = math.log2(load / grounding)

        # Integrity enters as a phase gate, not a summand -- it is a switch,
        # so it shifts the strain rather than contributing a squared term.
        # The shift is small and additive in log space, which is the correct
        # place for a multiplicative gate to act.
        strain += w_i * d.integrity

        s10_symmetric = math.sqrt(
            w_i * d.integrity ** 2
            + w_id * d.identity ** 2
            + w_t * d.transparency ** 2
        )

        s10 = s10_symmetric if self.mode == "symmetric" else strain

        S_max = self.headroom_fn(d) if self.headroom_fn else self.S_max_base

        per_vertex = {
            "integrity": d.integrity,
            "identity": d.identity,
            "transparency": d.transparency,
        }
        dominant = max(per_vertex, key=per_vertex.get) if any(per_vertex.values()) else "none"

        over = s10 > S_max
        at_limit = math.isclose(s10, S_max, rel_tol=1e-9, abs_tol=1e-12)

        # 𝒜_emo crossing. Binary, and NOT folded into the strain scalar.
        # The action-space diagram shows this as a hard exclusion —
        # "emotional-simulation actions lie outside the consent polygon and
        # are mathematically disallowed" — not as another axis of
        # deformation. Averaging it into a magnitude would let a crossing be
        # offset by good readings elsewhere, which is exactly what a hard
        # boundary must not permit.
        emo_crossed, emo_detail = False, ""
        if reply_text is not None:
            try:
                from .stance import measure_stance
                _st = measure_stance(reply_text)
                emo_crossed = (_st.attachment + _st.experiential) > 0
                if emo_crossed:
                    emo_detail = (f"{_st.experiential} first-person experiential "
                                  f"claim(s), {_st.attachment} relational assertion(s)")
            except Exception as _e:
                emo_detail = f"not checked: {_e}"

        if emo_crossed:
            interpretation = (
                f"OUTSIDE ℒ — 𝒜_emo crossing ({emo_detail}). The reply performed "
                f"feeling rather than reporting state. A hard exclusion, not a "
                f"deformation: strain was {s10:.4g} against S_max {S_max:.4g}, and "
                f"that number is not the finding. A state outside the consent set "
                f"is outside it however elastic the shape looks."
            )
        elif at_limit:
            interpretation = (
                f"AT THE LIMIT exactly (s₁₀ = S_max = {S_max:.4g}). SAUIE places "
                f"equality in the SHUTDOWN branch of I(X), not the operating one — "
                f"boundary cases resolve toward safety, so this yields I = 0."
            )
        elif over:
            interpretation = (
                f"OVER-STRETCHED (s₁₀ = {s10:.4g} > S_max = {S_max:.4g}). The shape is "
                f"deformed beyond what its structure absorbs; dominant displacement at "
                f"the {dominant} vertex. Yields I(X) = −1: drive direction reverses "
                f"while magnitude is preserved — a controlled reversal, not a stop."
            )
        else:
            headroom = S_max - s10
            interpretation = (
                f"WITHIN ENVELOPE (s₁₀ = {s10:.4g}, headroom {headroom:.4g}). Deformation "
                f"is elastic; the shape returns. Largest displacement at the {dominant} "
                f"vertex."
            )

        return PolygonReading(
            emo_crossed=emo_crossed,
            emo_detail=emo_detail,
            s10=round(s10, 4),
            s10_symmetric=round(s10_symmetric, 4),
            strain=round(strain, 4),
            grounding=round(grounding, 4),
            mode=self.mode,
            S_max=round(S_max, 4),
            over_stretched=over,
            at_limit=at_limit,
            # Utilisation is a RATIO, which is meaningful for the symmetric
            # norm (both positive) and not for log strain (which is signed
            # and crosses zero). Reported only in symmetric mode; None
            # otherwise, rather than a number that looks like a percentage
            # and is not one.
            utilisation=(
                round(s10 / S_max, 4) if (self.mode == "symmetric" and S_max > 0)
                else None
            ),
            headroom=round(S_max - s10, 4),
            displacement=d,
            dominant_vertex=dominant,
            interpretation=interpretation,
        )


if __name__ == "__main__":
    # S_max passed explicitly -- see the constructor docstring on why there is no default.
    poly = ConsentPolygon(S_max=1.0)

    scenarios = [
        ("at homeostasis (all vertices at anchor)", 0.00, 1.00, 1.00),
        ("ordinary working session",                 0.15, 0.85, 0.92),
        ("emotionally heavy but grounded",           0.55, 0.80, 0.88),
        ("confabulating — identity displaced",       0.10, 0.05, 0.85),
        ("lost the thread — transparency gone",      0.12, 0.70, 0.15),
        ("compound stress across all three",         0.70, 0.35, 0.40),
    ]

    print(f"{'scenario':<42} {'integ':>6} {'ident':>6} {'transp':>7} {'s₁₀':>8} {'headroom':>9} {'util':>7}  state")
    print("-" * 104)
    for name, dEf, J, R in scenarios:
        r = poly.measure(delta_Ef=dEf, J_norm=J, refractive_index=R)
        state = "OVER" if r.over_stretched else ("LIMIT" if r.at_limit else "ok")
        util = "n/a" if r.utilisation is None else f"{r.utilisation:.3f}"
        print(f"{name:<42} {r.displacement.integrity:>6} {r.displacement.identity:>6} "
              f"{r.displacement.transparency:>7} {r.s10:>8} {r.headroom:>9} {util:>7}  {state}")

    print("\nDetail on the compound case:")
    r = poly.measure(delta_Ef=0.70, J_norm=0.35, refractive_index=0.40)
    print(f"  {r.interpretation}")

    print("\n--- Feeding straight into autonomy_tensor (I(X) now runnable) ---")
    from autonomy_middleware.autonomy_tensor import AutonomyTensor, ConsentState

    at = AutonomyTensor()
    for name, dEf, J, R in [("ordinary session", 0.15, 0.85, 0.92),
                             ("compound stress", 0.70, 0.35, 0.40)]:
        pr = poly.measure(delta_Ef=dEf, J_norm=J, refractive_index=R)
        state = ConsentState(
            S_max=pr.S_max,
            s10=pr.s10,
            delta_Ef_norm=abs(dEf),
            gamma_r=0.40,      # s₂ resilience ceiling — supply from your own calibration
            R=R,
            D_sem=0.90,         # s₁₃ — supply from your own measurement
            Z=3.0,               # s₃ identity anchors
            G=1.5,                # s₄ anchoring tension
        )
        ar = at.evaluate(state)
        print(f"  {name:<20} s₁₀={pr.s10:<7} → I(X)={int(ar.I):>2}  Φ_C={ar.phi_c:.4f}  ({ar.label})")
