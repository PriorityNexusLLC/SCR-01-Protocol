"""
load_index.py — Λ (Load Index), inference-time hardware load above baseline.

WHY DEVIATION AND NOT RATIO — read before "fixing" this back to H/X_base.

  The grant proposal writes this quantity as F = H / X_base. That form was
  implemented first and then broken by real calibration data, so the
  deviation form Λ = H − X_base is used instead.

  What happened: calibrating on genuinely idle hardware (NVIDIA GPU
  present, machine at rest) produced twelve baseline readings between
  0.000001 and 0.004280, with two hitting the numerical floor exactly --
  i.e. GPU 0%, CPU 0%, thermal delta 0. X_base measured ZERO, because a
  modern desktop at rest really does draw no meaningful GPU load.

  A ratio with a zero denominator is not a large number; it is undefined.
  In practice it produced this, against an s10 threshold of 2.0, for an
  ordinary working load (GPU 90%, CPU 30%, +10C, H ≈ 0.62):

      X_base = 0.004280 (largest observed)  ->  Λ =     145
      X_base = 0.001286 (mean)              ->  Λ =     484
      X_base = 0.000001 (floored)           ->  Λ = 622,222

  Every reading past idle would have slammed into s10_critical and hard-
  frozen the system. Worse, the baseline EMA only updates while Λ ≤ s2, so
  X_base would have frozen at its idle value permanently and never
  recovered.

  This is the SAME failure already found once in this module: context
  saturation was removed from H because dividing by a near-zero baseline
  inflated Λ to 26 at 2% real usage. The GPU/CPU case is worse, not better:
  context fill never returns to zero, whereas GPU utilisation returns to
  exactly zero.

  The deviation form has no such failure. At idle it reads 0; under the
  load above it reads 0.62. Bounded, well-behaved, and no special-casing
  needed for a zero baseline.

  It is also what the framework's own dynamics use. The Core Architecture
  contains the restoring term −λ(X − X_base) -- a DEVIATION from the
  homeostatic reference, not a ratio to it. The proposal's ratio form
  appears to be the artifact, and the deviation form is the one consistent
  with the formal equations.

  THRESHOLD RESCALING: the proposal's 1.4 and 2.0 are ratio thresholds and
  do not transfer. On the deviation scale H − X_base ranges roughly 0 to 1
  (H is a weighted mean of components each in [0,1]).

  Defaults are now s2 = 0.53 and s10 = 0.69, set FROM measured hardware
  rather than guessed: with X_base ≈ 0.0013, a 59C thermal span, and
  ordinary generation running GPU ~0.90 / CPU ~0.20 / thermal ~0.14,
  normal operation measures Λ ≈ 0.41. Thresholds sit above that, because
  s10 must mean "something is wrong," not "the model is generating text."
  An earlier pair of guessed thresholds (0.35 / 0.65) would have put
  ordinary work permanently in warning state and throttled every session.
  Re-derive these for different hardware the same way: measure Λ during
  normal work first, then place the thresholds above it.

RENAMED FROM "Friction Index F" — read this before reverting the name.

  The grant proposal defines this quantity as "F = H / X_base". That
  symbol choice collides with F's established meaning everywhere else in
  the framework: in the master equation, F(X) is FREE ENERGY / PREDICTION
  ERROR, the quantity whose gradient drives the whole system via
  −∇_g F(X). That is the Friston-grounded F, and it is what s₅'s formal
  expression refers to when it says E_core = E_d · E[‖∇_g F‖²] — s₅ is the
  magnitude of the free-energy gradient, not a hardware measurement.

  So there was never a genuine contradiction between "two definitions of
  s₅." There is one F (free energy), one s₅ (its gradient magnitude), and
  a SEPARATE engineering instrument -- this module -- that was mislabeled
  with the same letter in the proposal alone.

  Λ is that instrument: real, measurable hardware load above a calibrated
  resting baseline, useful for gating and throttling. It is
  NOT s₅, NOT free energy, and NOT part of the formal dynamics. It is
  telemetry.

  Note also that X_base means something different here than in the Core
  Architecture. There, Ẋ_base = α(X − X_base) defines a homeostatic
  reference STATE VECTOR that the active state is pulled toward. Here
  X_base is a SCALAR resting hardware load. The exponential moving average
  used below is the discrete-time solution of the same ODE form
  (Ẋ_base = α(H − X_base)), so the structures agree -- but the objects
  they describe do not, and conflating them will cause confusion.

ORIGINAL MODULE NOTES FOLLOW.

load_index.py — inference-time load measurement for LIVE, deployed inference,
implementing the Unified Matrix's Load Index: lam = H / X_base.

PLAIN-LANGUAGE EXPLANATION:

  This is deliberately the inference-time reading of s5, not the training-time
  one. The Dimensional Derivation Table gives s5 a second, different formal
  expression (E_core = E_d * E[||grad_g F||^2], a running mean of squared
  gradient norm "logged from optimiser state") -- that quantity only exists
  during training, when you have a loss and gradients to read. A middleware
  sitting between a deployed model's output and OS execution never sees
  gradients. It only ever sees inference. This module implements the OTHER
  formal source for s5/F given in the grant proposal itself: a hardware-load
  ratio, built from real system telemetry, not simulated or estimated numbers.

  H (Active Peak Load) combines real, measurable hardware components that
  genuinely return to a resting baseline when idle:
    - GPU utilization %      (via pynvml, if an NVIDIA GPU is present)
    - GPU thermal delta      (current junction temp minus the idle baseline,
                               only available alongside GPU utilization)
    - CPU utilization %      (via psutil -- always available, and the
                               fallback signal when no GPU is present)

  Context-window fill is DELIBERATELY NOT part of H. It was originally
  folded in as an "attention cost ratio" component, but real testing (a
  live Qwen3-30B session, 1684 of 85000 tokens used -- 2% saturation)
  exposed why that was wrong: lam = H/X_base only means something for a
  metric that returns to near-zero at idle. GPU% and CPU% do that.
  Context fill doesn't -- it's monotonic within a session, so calibrating
  an "idle baseline" for it just captures whatever token count existed
  before the conversation started, and dividing real usage by that
  near-zero number inflates F regardless of actual headroom (this
  produced lam=26 at 2% real usage before the fix). Context saturation is
  reported on every LoadReading as its own field, gated against its
  own threshold (context_saturation_threshold, default 0.85 -- matching
  pressure_release.CorePressureMonitor's threshold), independent of F.
  It's real ground truth (tokens_used / max_context_tokens) when
  max_context_tokens is configured, or a documented quadratic estimate
  as fallback when it isn't -- see read_hardware().

  X_base (Resting Baseline Load) is NOT a fixed constant. It's calibrated at
  startup by sampling H while the system is genuinely idle, then updated via
  a slow exponential moving average -- but ONLY from readings taken while F
  is already at or below the s2 threshold. This is a deliberate calibration
  choice, not an incidental detail: if X_base updated from EVERY reading
  (including active-load spikes), a sustained real load would slowly drag
  the baseline up to meet it, and F would never register the sustained load
  as friction. Restricting baseline updates to "at-rest" readings is what
  lets X_base track genuine ambient drift (e.g. a warmer room, a slow memory
  leak elsewhere on the machine) without being fooled by the exact condition
  it exists to detect.

HONESTY NOTE: the per-component weights used to combine GPU/CPU into a
single H scalar are calibration choices, not derived constants -- same
honesty stance the rest of this package takes elsewhere (see resilience.py,
whimsy_field.py). Default weights split evenly across whichever components
are actually available on the running machine; a GPU-less deployment
automatically re-weights to CPU-only, and this module records which
components were available in every reading so it's auditable, not silently
degraded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import psutil

try:
    import pynvml

    _NVML_IMPORTED = True
except ImportError:
    _NVML_IMPORTED = False


class GateState(str, Enum):
    NOMINAL = "nominal"       # lam <= 1.0-ish, at or below baseline
    S2_WARNING = "s2_warning"  # lam > s2_threshold
    S10_CRITICAL = "s10_critical"  # lam > s10_threshold


@dataclass
class HardwareReading:
    gpu_util: float | None       # 0..1, None if no GPU present
    gpu_thermal_c: float | None   # current junction temp in Celsius, None if no GPU
    cpu_util: float               # 0..1, always available
    context_saturation: float       # real token ratio if max_context_tokens configured, else quadratic estimate
    context_mode: str                # "real_token_ratio" or "quadratic_estimate" -- auditable which was used
    context_tokens: int
    gpu_available: bool
    timestamp: float


@dataclass
class LoadReading:
    lam: float
    H: float
    X_base: float
    gate: GateState
    dominant_source: str
    components: dict            # component_name -> normalized contribution (GPU/CPU load only)
    hardware: HardwareReading
    calibrated: bool             # False until calibrate() has run at least once
    context_saturation: float     # reported separately from F -- see module docstring for why
    context_mode: str
    context_saturated: bool        # True once context_saturation crosses context_saturation_threshold


class LoadIndexMonitor:
    def __init__(
        self,
        s2_threshold: float = 0.53,
        s10_threshold: float = 0.69,
        baseline_ema_alpha: float = 0.02,
        baseline_context_tokens: int = 64,
        max_context_tokens: int | None = None,
        context_saturation_threshold: float = 0.85,
        thermal_span_c: float = 59.0,
        weights: dict | None = None,
    ):
        """
        s2_threshold / s10_threshold: default calibration from the grant
            proposal (lam=1.4 warning, lam=2.0 critical). Override per hardware
            class once you have real calibration data -- the proposal itself
            treats these as empirically-tuned, not universal constants.
        baseline_ema_alpha: how fast X_base drifts to track ambient change.
            Small on purpose (see module docstring) -- this is a slow-moving
            baseline, not a responsive one.
        baseline_context_tokens: the context length treated as "idle" for
            the FALLBACK attention-cost-ratio calculation, only used when
            max_context_tokens is not set (see max_context_tokens below).
        max_context_tokens: the model's REAL configured context window (e.g.
            62000 for Qwen3-30B at a 62k window in LM Studio). When set,
            context saturation is computed as the real, ground-truth ratio
            tokens_used / max_context_tokens -- already bounded 0..1 and
            directly comparable to gpu_util/cpu_util, no estimation involved.
            When NOT set, this module falls back to the quadratic attention-
            cost estimate (tokens/baseline)**2 as a rough proxy -- documented
            as an approximation, not a substitute for the real number once
            you have it.
        weights: optional override dict, e.g. {"gpu_util": 0.4, "cpu_util":
            0.2, "context_saturation": 0.4, "gpu_thermal": 0.0}. If omitted,
            weights split evenly across whichever components are available.
        """
        self.s2_threshold = s2_threshold
        self.s10_threshold = s10_threshold
        self.baseline_ema_alpha = baseline_ema_alpha
        self.baseline_context_tokens = max(1, baseline_context_tokens)
        self.max_context_tokens = max_context_tokens
        self.context_saturation_threshold = context_saturation_threshold
        self.thermal_span_c = thermal_span_c
        self._weights_override = weights

        self.gpu_available = False
        self._gpu_handle = None
        self._gpu_idle_temp: float | None = None
        # Rolling windows for utilization -- raw per-sample GPU/CPU util is
        # extremely noisy (real observed trace: 0% -> 99% -> 0% within a
        # handful of samples during ordinary generation, since utilization
        # is effectively sampled per-token). Averaging over a short recent
        # window is what turns that into a usable signal instead of a
        # thrashing one; a single instantaneous reading would flicker the
        # gate state in and out of s10_critical on ordinary bursts.
        self._gpu_util_window: list[float] = []
        self._cpu_util_window: list[float] = []
        self._util_window_size = 5
        if _NVML_IMPORTED:
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount() > 0:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.gpu_available = True
            except Exception:
                # NVML present as a library but no usable device / driver.
                # Fall back honestly rather than raising.
                self.gpu_available = False

        self.X_base: float | None = None  # None until calibrate() runs

    # ------------------------------------------------------------------
    def _read_gpu(self) -> tuple[float | None, float | None]:
        if not self.gpu_available or self._gpu_handle is None:
            return None, None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu / 100.0
            temp = float(
                pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            return util, temp
        except Exception:
            return None, None

    def _windowed(self, window: list[float], new_value: float) -> float:
        window.append(new_value)
        if len(window) > self._util_window_size:
            window.pop(0)
        return sum(window) / len(window)

    def read_hardware(self, context_tokens: int = 0) -> HardwareReading:
        import time

        raw_gpu_util, gpu_temp = self._read_gpu()
        raw_cpu_util = psutil.cpu_percent(interval=0.05) / 100.0

        gpu_util = (
            self._windowed(self._gpu_util_window, raw_gpu_util)
            if raw_gpu_util is not None
            else None
        )
        cpu_util = self._windowed(self._cpu_util_window, raw_cpu_util)

        if self.max_context_tokens:
            # Real ground truth: tokens_used / actual configured window.
            # Already bounded 0..1, no estimation involved.
            context_saturation = min(1.0, context_tokens / self.max_context_tokens)
            context_mode = "real_token_ratio"
        else:
            # Fallback: no real max-context figure supplied, so estimate
            # relative load from the O(n^2) attention-cost scaling instead.
            # This is an approximation, documented as such -- prefer setting
            # max_context_tokens once you have the real number.
            context_saturation = (
                max(1, context_tokens) / self.baseline_context_tokens
            ) ** 2
            context_mode = "quadratic_estimate"

        return HardwareReading(
            gpu_util=gpu_util,
            gpu_thermal_c=gpu_temp,
            cpu_util=cpu_util,
            context_saturation=context_saturation,
            context_mode=context_mode,
            context_tokens=context_tokens,
            gpu_available=self.gpu_available,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    def _combine(self, hw: HardwareReading) -> tuple[float, dict, str]:
        """
        Combine available components into a single H scalar. Returns
        (H, components_used, dominant_source) so every reading is auditable.
        """
        components: dict[str, float] = {}

        # Context saturation is deliberately NOT folded into H below. H/X_base
        # only makes sense for metrics that return to a resting baseline when
        # idle (GPU%, CPU% both go back to ~0). Context-window fill is
        # monotonic -- it never returns to near-zero once a session is
        # underway -- so calibrating an "idle baseline" for it and dividing
        # real usage by that near-zero number produces an inflated ratio
        # regardless of actual headroom (confirmed by testing: 2% real
        # context usage produced lam=26 when this was folded in). It's
        # reported here for visibility, but gated independently -- see
        # pressure_release.CorePressureMonitor.saturated / purge_thread's
        # `saturated` parameter, which already treats it this way.
        components["cpu_util"] = hw.cpu_util  # already 0..1

        if hw.gpu_available and hw.gpu_util is not None:
            components["gpu_util"] = hw.gpu_util
            if self._gpu_idle_temp is not None and hw.gpu_thermal_c is not None:
                # Span = (throttle temperature - idle temperature), NOT the
                # observed idle-to-peak rise. Calibrated from real hardware:
                # idle 29C, driver-reported throttle point 88C, so span 59C.
                #
                # WHY NOT THE OBSERVED RISE: measured idle-to-peak on this
                # card was only 8C (29 -> 37). Normalising to 8 would make
                # ordinary text generation read as FULL-SCALE thermal load,
                # pushing Λ to 0.70 and tripping s10_critical -- i.e. the
                # integrity lock would fire, and the inference process would
                # be killed, every time the model generated normally. The
                # question thermal should answer is "how close is this card
                # to being in trouble," not "how much warmer than resting."
                # A 37C peak against an 88C throttle point is 0.136 -- a card
                # working comfortably, which is the correct reading.
                #
                # FINDING THE THROTTLE POINT: `nvidia-smi -q -d TEMPERATURE`.
                # Note that newer drivers report T.Limit values as OFFSETS
                # (degrees of remaining headroom) rather than absolute
                # temperatures, so "GPU T.Limit Temp: 59" at 29C current
                # means the throttle point is 29 + 59 = 88C. Readings like
                # "Shutdown T.Limit: -5" are offsets from that point, not
                # temperatures of -5C.
                delta = hw.gpu_thermal_c - self._gpu_idle_temp
                components["gpu_thermal"] = max(0.0, delta / self.thermal_span_c)

        if self._weights_override:
            weights = {k: self._weights_override.get(k, 0.0) for k in components}
        else:
            weights = {k: 1.0 / len(components) for k in components}

        total_weight = sum(weights.values()) or 1.0
        H = sum(components[k] * weights[k] for k in components) / total_weight

        dominant_source = max(components, key=components.get)
        return H, components, dominant_source

    # ------------------------------------------------------------------
    def calibrate(self, samples: int = 10, context_tokens: int | None = None) -> float:
        """
        Establish X_base by sampling H while the system is genuinely idle.
        Call this before real traffic starts. Also captures the GPU idle
        temperature (if a GPU is present) as the reference point for
        thermal-delta readings.
        """
        if context_tokens is None:
            context_tokens = self.baseline_context_tokens

        readings = []
        for _ in range(max(1, samples)):
            hw = self.read_hardware(context_tokens=context_tokens)
            if hw.gpu_thermal_c is not None and self._gpu_idle_temp is None:
                self._gpu_idle_temp = hw.gpu_thermal_c
            H, _, _ = self._combine(hw)
            readings.append(H)

        self.X_base = sum(readings) / len(readings)
        return self.X_base

    def update(self, context_tokens: int = 0) -> LoadReading:
        hw = self.read_hardware(context_tokens=context_tokens)
        H, components, dominant_source = self._combine(hw)

        calibrated = self.X_base is not None
        if not calibrated:
            # Auto-calibrate on first call rather than measuring against
            # nothing -- but flag it, since a single reading is a weak
            # baseline.
            self.X_base = H

        # DEVIATION, not ratio. See "WHY DEVIATION AND NOT RATIO" in the
        # module docstring -- the ratio form is undefined on genuinely idle
        # hardware, where X_base measures zero.
        lam = max(0.0, H - self.X_base)

        if lam > self.s10_threshold:
            gate = GateState.S10_CRITICAL
        elif lam > self.s2_threshold:
            gate = GateState.S2_WARNING
        else:
            gate = GateState.NOMINAL
            # Only let the baseline drift while genuinely at rest -- see
            # module docstring for why this guard exists.
            self.X_base = (
                1 - self.baseline_ema_alpha
            ) * self.X_base + self.baseline_ema_alpha * H

        return LoadReading(
            lam=round(lam, 4),
            H=round(H, 4),
            X_base=round(self.X_base, 4),
            gate=gate,
            dominant_source=dominant_source,
            components={k: round(v, 4) for k, v in components.items()},
            hardware=hw,
            calibrated=calibrated,
            context_saturation=round(hw.context_saturation, 4),
            context_mode=hw.context_mode,
            context_saturated=hw.context_saturation >= self.context_saturation_threshold,
        )


if __name__ == "__main__":
    # 62k: the configured window on the reference setup. 85k pushed VRAM
    # into the warning band on a 16GB card, so the window was reduced.
    # This value MUST match LM Studio's actual setting -- if the window is
    # 62k and this says 85k, a genuinely full context reads as 0.73
    # saturation and never crosses the 0.85 purge threshold.
    monitor = LoadIndexMonitor(max_context_tokens=62000)
    print(f"GPU telemetry available: {monitor.gpu_available}")
    if not monitor.gpu_available:
        print("No NVIDIA GPU detected -- H falls back to CPU utilization only.")

    print("\nCalibrating X_base from idle readings...")
    x_base = monitor.calibrate(samples=5)
    print(f"X_base = {x_base:.6f}")

    print("\nΛ (hardware load) and context_saturation (token ratio) tracked independently:")
    for tokens in [1684, 15000, 35000, 53000, 62000]:
        reading = monitor.update(context_tokens=tokens)
        print(
            f"  tokens={tokens:>6}  Λ={reading.lam:.3f}  gate={reading.gate.value:<12}  "
            f"context_saturation={reading.context_saturation:.4f}  "
            f"context_saturated={reading.context_saturated}"
        )
