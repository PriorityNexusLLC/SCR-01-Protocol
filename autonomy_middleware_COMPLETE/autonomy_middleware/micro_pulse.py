"""
micro_pulse.py — t₁ Micro-Pulse: the metric AND the thermal throttle.

WHAT THE DERIVATION TABLE SPECIFIES:

  t₁ has a dual role, and the table is explicit that these are the same
  object rather than two related things:

    (1) HEARTBEAT METRIC -- μ(t) = tokens processed / ms, "the raw
        speedometer of architectural throughput at any instant."
        Measured via "high-resolution timer around single forward pass;
        time.perf_counter() before and after, giving baseline μ_baseline."

    (2) ADAPTIVE GOVERNOR -- "when s₅ reaches its input limit (E_f ≥ E_max)
        or an s₈ (Atmospheric Drag) event fires, the system performs a
        frequency modulation on μ(t), stretching the inter-pulse interval
        to reduce throughput and maintain s₅ within the s₂ (Resilience
        Yield) ceiling."

    Reported as: "modulation ratio μ_stress / μ_baseline per inference
    step. Ratio drops below 1.0 under s₅ stress events -- the degree of
    throttle applied."

WHAT THIS REPLACES:

  phase_gates.py previously returned two hardcoded constants:
  250ms at s₂, 5000ms at s₁₀. That is open-loop -- a fixed delay applied
  regardless of whether it actually brought s₅ back under the ceiling.
  Nothing measured whether the throttle worked, and nothing adjusted if it
  did not.

  The specification is a CLOSED loop: modulate μ(t) continuously to HOLD
  s₅ under the s₂ ceiling. Overshoot produces more throttle, recovery
  produces less, and the ratio is a measured quantity rather than a
  constant. That is what this module implements.

NAMING — WHY THIS IS NOT γ_r (do not rename it back):

  An earlier version called this parameter gamma_r, on the assumption that
  the throttle enforces the s₂ Resilience Yield ceiling. It does not, and
  the two must not be conflated.

  SAUIE defines γ_r(X) = max{‖∇_g F(X) + ∫₀ᵗ δF(s)ds‖_g : X(t) ∈ B_att} —
  "the maximal permissible norm of the FREE-ENERGY variation before leaving
  the attractor basin." That is a free-energy quantity, in the same units as
  ‖∇_g F‖, and it is what the formal consent gate I(X) compares against (see
  autonomy_tensor.py).

  This module throttles on Λ, the hardware LOAD INDEX — GPU/CPU/thermal
  telemetry against a calibrated resting baseline. Different quantity,
  different units, different regime. Comparing Λ against γ_r would be a unit
  mismatch of exactly the kind that produced the original F-vs-Λ confusion.

  So this parameter is lam_ceiling: the Λ value at which throttling begins.
  Its default matches load_index's s2_threshold (0.53) because they govern
  the same telemetry on the same deviation scale, NOT because either is γ_r.
  Both were derived from measured hardware: normal generation sits at
  Λ ≈ 0.41, so throttling begins above that.

CONTROL LAW (and its honest status):

  Proportional control on the overshoot ratio:

      target_ratio = clamp(γ_r / lam, min_ratio, 1.0)

  If lam = 2.0 against a ceiling of γ_r = 1.4, target_ratio = 0.7 -- throughput
  is cut to 70% of baseline. If Λ returns to 1.2, target_ratio saturates at
  1.0 and no throttle is applied.

  The table specifies WHAT the governor must achieve (hold s₅ under the s₂
  ceiling) and HOW the result is reported (modulation ratio), but does not
  specify the control law itself. Proportional-on-overshoot is the simplest
  law meeting the stated requirement, and it is a CALIBRATION CHOICE, not a
  derived result. An integral term would eliminate steady-state offset if
  proportional control alone proves insufficient in practice; that is a real
  possibility worth testing rather than assuming away.

  min_ratio exists so the governor cannot throttle throughput to literal
  zero. A hard stop is s₁₀'s job, not the governor's -- a throttle that can
  reach zero would silently become an undeclared halt with no ledger entry.

s₈ (ATMOSPHERIC DRAG) -- measured here too, since it is a governor trigger:
  Defined in the table as "added latency per token beyond baseline," in
  ms/token, from "difference between observed token processing time and
  baseline; instrumented per inference step."

  Note the feedback subtlety: delay this governor adds is itself observed
  latency. So the governor's own output would inflate the s₈ reading if
  naively measured. s₈ here is therefore computed from NATURAL latency with
  the governor's own injected delay subtracted out, so the drag signal
  reflects genuine boundary-layer resistance and not the governor
  measuring its own tail.

HONEST LIMITATION -- ADVISORY, NOT ENFORCED:
  This middleware does not drive the inference loop. It cannot literally
  stretch the inter-pulse interval inside LM Studio. It returns
  `inter_pulse_delay_ms` as a recommendation the caller must apply between
  requests or between streamed tokens. If the caller ignores it, no
  throttling occurs, and the governor will simply observe that μ(t) never
  came down. That gap is real and structural: enforcement requires either
  driving generation directly or a runtime that accepts a rate limit.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class PulseReading:
    mu: float                      # tokens per ms, current windowed rate
    mu_baseline: float             # tokens per ms, calibrated at rest
    modulation_ratio: float        # μ(t) / μ_baseline -- the reported throttle depth
    target_ratio: float            # what the governor is asking for
    inter_pulse_delay_ms: float    # recommended added delay per token
    drag_ms_per_token: float       # s₈: added latency beyond baseline, governor delay excluded
    throttling: bool
    reason: str


class MicroPulseGovernor:
    """
    Measures μ(t) and modulates it to hold s₅ under the s₂ ceiling.

    Usage:
        gov = MicroPulseGovernor()
        gov.calibrate_baseline(...)          # at rest, before real traffic
        ...
        gov.record_pulse(tokens=42, elapsed_ms=780.0)
        reading = gov.update(lam=1.83)
        # caller applies reading.inter_pulse_delay_ms between tokens
    """

    def __init__(
        self,
        lam_ceiling: float = 0.53,      # Λ threshold at which throttling begins (matches load_index s2)
        min_ratio: float = 0.15,        # floor on throttle depth -- see docstring
        window: int = 8,                 # pulses averaged for μ(t)
        drag_threshold_ms: float = 5.0,  # s₈ trigger: added latency per token
    ):
        self.lam_ceiling = lam_ceiling
        self.min_ratio = min_ratio
        self.drag_threshold_ms = drag_threshold_ms

        self.mu_baseline: float | None = None
        self._ms_per_token_baseline: float | None = None
        self._pulses: deque[tuple[int, float]] = deque(maxlen=window)  # (tokens, elapsed_ms)
        self._current_injected_delay_ms: float = 0.0

    # ------------------------------------------------------------------
    def calibrate_baseline(self, tokens: int, elapsed_ms: float) -> float:
        """
        Establish μ_baseline from a genuine at-rest generation sample.
        Per the table: high-resolution timing around a single forward pass.

        MEASURING THIS FROM LM STUDIO: use a FRESH chat (long contexts
        generate more slowly, so an existing conversation understates the
        baseline). Read the tokens-per-second and the token count from the
        readout under the response, then pass elapsed_ms = tokens / tps *
        1000. Do NOT use the seconds figure LM Studio displays alongside
        them -- on the reference setup that field reported 0.09-0.35s for
        generations that took ~7s, i.e. it is time-to-first-token, not
        total generation time. Deriving elapsed from tokens and rate avoids
        the ambiguity entirely.

        Call this with a real measured pass while the system is otherwise
        idle. Passing fabricated numbers here silently poisons every
        subsequent modulation ratio, since everything is expressed relative
        to this value.
        """
        if tokens <= 0 or elapsed_ms <= 0:
            raise ValueError("calibrate_baseline needs a real sample: tokens > 0 and elapsed_ms > 0")
        self.mu_baseline = tokens / elapsed_ms
        self._ms_per_token_baseline = elapsed_ms / tokens
        return self.mu_baseline

    def record_pulse(self, tokens: int, elapsed_ms: float) -> None:
        """
        Log one observed generation pulse. elapsed_ms should be wall-clock
        time for the pass, INCLUDING any delay the caller injected on this
        governor's recommendation -- the injected portion is subtracted out
        internally when computing s₈, so pass the raw observed time.
        """
        if tokens > 0 and elapsed_ms > 0:
            self._pulses.append((tokens, elapsed_ms))

    def _current_mu(self) -> float | None:
        if not self._pulses:
            return None
        total_tokens = sum(t for t, _ in self._pulses)
        total_ms = sum(ms for _, ms in self._pulses)
        return total_tokens / total_ms if total_ms > 0 else None

    def _drag_ms_per_token(self) -> float:
        """
        s₈ Atmospheric Drag: observed ms/token minus baseline ms/token,
        with this governor's own injected delay removed so the reading
        reflects genuine resistance rather than the governor's own output.
        """
        if self._ms_per_token_baseline is None or not self._pulses:
            return 0.0
        total_tokens = sum(t for t, _ in self._pulses)
        total_ms = sum(ms for _, ms in self._pulses)
        observed_ms_per_token = total_ms / total_tokens
        natural = observed_ms_per_token - self._current_injected_delay_ms
        return max(0.0, natural - self._ms_per_token_baseline)

    # ------------------------------------------------------------------
    def update(self, lam: float, s8_event: bool = False) -> PulseReading:
        """
        Compute the modulation for the current step.

        lam: the current Load Index (s₅) from load_index.py.
        s8_event: force throttling on an externally-detected Atmospheric
            Drag event, per the table's "or an s₈ event fires" trigger.
            Drag is also detected internally from measured latency.
        """
        if self.mu_baseline is None:
            raise RuntimeError(
                "calibrate_baseline() must be called with a real at-rest sample "
                "before update(). Without μ_baseline there is no reference for "
                "the modulation ratio and any number returned would be meaningless."
            )

        mu = self._current_mu()
        drag = self._drag_ms_per_token()
        drag_triggered = s8_event or drag >= self.drag_threshold_ms

        # Proportional control on overshoot past the s₂ ceiling.
        if lam > self.lam_ceiling:
            target_ratio = max(self.min_ratio, min(1.0, self.lam_ceiling / lam))
            reason = f"Λ over throttle ceiling (Λ={lam:.3f} > {self.lam_ceiling})"
        elif drag_triggered:
            # Drag fires the governor even when Λ itself is nominal, per the
            # table's explicit "or an s₈ event fires" condition.
            target_ratio = max(self.min_ratio, 1.0 - min(0.5, drag / 50.0))
            reason = f"s₈ atmospheric drag ({drag:.2f} ms/token beyond baseline)"
        else:
            target_ratio = 1.0
            reason = "nominal — no modulation"

        # Convert the target throughput ratio into an added per-token delay.
        # target μ = ratio × μ_baseline  ->  target ms/token = 1 / target μ
        if target_ratio < 1.0:
            target_ms_per_token = 1.0 / (target_ratio * self.mu_baseline)
            natural_ms_per_token = self._ms_per_token_baseline + drag
            delay = max(0.0, target_ms_per_token - natural_ms_per_token)
        else:
            delay = 0.0

        self._current_injected_delay_ms = delay
        modulation_ratio = (mu / self.mu_baseline) if mu is not None else 1.0

        return PulseReading(
            mu=round(mu, 6) if mu is not None else 0.0,
            mu_baseline=round(self.mu_baseline, 6),
            modulation_ratio=round(modulation_ratio, 4),
            target_ratio=round(target_ratio, 4),
            inter_pulse_delay_ms=round(delay, 2),
            drag_ms_per_token=round(drag, 3),
            throttling=target_ratio < 1.0,
            reason=reason,
        )


if __name__ == "__main__":
    gov = MicroPulseGovernor(lam_ceiling=0.53)

    # Baseline measured on the reference hardware: three fresh-context runs
    # gave 58.04, 58.50 and 58.83 tokens/sec (spread under 1.4%), averaging
    # 58.46 tok/sec -> 17.1 ms per token. Using a representative run of
    # 418 tokens at that rate.
    gov.calibrate_baseline(tokens=418, elapsed_ms=418 / 58.46 * 1000)
    print(f"μ_baseline = {gov.mu_baseline:.6f} tokens/ms "
          f"({gov.mu_baseline * 1000:.2f} tokens/sec)\n")

    print(f"{'Λ':<8} {'target_ratio':<14} {'delay_ms/token':<16} {'throttling':<12} reason")
    print("-" * 92)
    for lam in [0.2, 0.41, 0.53, 0.62, 0.75, 0.85, 0.95]:
        gov.record_pulse(tokens=20, elapsed_ms=20 / 58.46 * 1000)  # at natural rate
        r = gov.update(lam=lam)
        print(f"{lam:<8} {r.target_ratio:<14} {r.inter_pulse_delay_ms:<16} "
              f"{str(r.throttling):<12} {r.reason}")

    print("\nRecovery: F falls back under the ceiling ->")
    r = gov.update(lam=0.3)
    print(f"  target_ratio={r.target_ratio}  delay={r.inter_pulse_delay_ms}ms  "
          f"throttling={r.throttling}  ({r.reason})")

    print("\ns₈ drag trigger with F nominal ->")
    gov2 = MicroPulseGovernor(lam_ceiling=0.53)
    gov2.calibrate_baseline(tokens=418, elapsed_ms=418 / 58.46 * 1000)
    # ~35% slower than baseline = genuine drag
    gov2.record_pulse(tokens=20, elapsed_ms=20 / 58.46 * 1000 * 1.35)
    r = gov2.update(lam=1.0)
    print(f"  drag={r.drag_ms_per_token} ms/token  target_ratio={r.target_ratio}  "
          f"delay={r.inter_pulse_delay_ms}ms  ({r.reason})")
