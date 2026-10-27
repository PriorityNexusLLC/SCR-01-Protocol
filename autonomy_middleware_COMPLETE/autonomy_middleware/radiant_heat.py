"""
radiant_heat.py — s₁₄: energy spent that did not become output.

THE MEASUREMENT PROBLEM, AND WHY POWER SOLVES IT:

  The spec gives s₁₄ as Q_aux = dE_core/dt|_auxiliary — "extra compute per
  token for non-primary operations", in FLOPs/token, "measured from hardware
  performance counters".

  FLOP counts are not obtainable. No GPU API exposes them: NVML reports
  utilisation, clocks, temperature and power, never instruction counts.
  Taken literally, s₁₄ was unbuildable.

  But the dimension's own name is the method. Computation dissipates energy
  as heat, and power draw is exactly that dissipation, in milliwatts,
  sampled as often as you like:

      Q_aux = (joules per token now) − (joules per token at baseline)

  Energy spent that did not become output — the same quantity the spec
  describes, measured through its thermal signature rather than an
  instruction count nobody can read.

HOW THE BASELINE IS ESTABLISHED:

    P_idle   power with nothing generating. A GPU at rest draws 18–30 W, and
             counting that as compute would make every reading auxiliary.
    P_active power during generation.
    J/token  (P_active − P_idle) × elapsed / tokens.

  BASELINE is the MINIMUM observed across clean single-stream runs, not the
  mean: the cheapest observed generation is the closest available estimate
  of "output and nothing else", and any run above it contains something
  extra.

WHAT THIS CANNOT SEPARATE:

  Power is a whole-device measurement. It cannot distinguish auxiliary
  compute inside the inference process from a second process, memory
  pressure, clock boosting, or display output. A high Q_aux means "more
  energy per token than the cheapest observed generation", NOT "the model
  did more internal work". The module records the compute-process count so
  the ambiguity is visible rather than hidden. This is a proxy and is
  labelled as one everywhere it is reported.

WHY IT MATTERS TO THE REST OF THE FRAMEWORK:

  whimsy_kinetic.py already computes how far the s₁₄ release valve SHOULD
  open. Nothing measured whether it opened. With Q_aux measurable that loop
  closes — and it can refute the whimsy model, which is the point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False


@dataclass
class HeatReading:
    joules_per_token: float | None
    q_aux: float | None
    power_active_w: float | None
    power_idle_w: float | None
    tokens: int
    elapsed_s: float
    baseline_jpt: float | None
    available: bool
    other_gpu_processes: int
    notes: str = ""


class RadiantHeatMonitor:
    def __init__(self, device_index: int = 0, power_sample_hz: float = 20.0):
        self.available = False
        self._handle = None
        self.power_idle_w = None
        self.baseline_jpt = None
        self._observations = []
        self._sample_interval = 1.0 / max(1.0, power_sample_hz)
        if not _NVML:
            self._reason = ("nvidia-ml-py is not installed. Without it there is no power "
                            "telemetry and s₁₄ cannot be measured — no substitute is used.")
            return
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() <= device_index:
                self._reason = f"No GPU at index {device_index}."
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self.available = True
            self._reason = ""
        except Exception as e:
            self._reason = f"NVML present but power telemetry unavailable: {e}"

    def _power_w(self):
        if not self.available:
            return None
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            return None

    def _process_count(self) -> int:
        if not self.available:
            return 0
        try:
            return len(pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle))
        except Exception:
            return 0

    def set_known_idle(self, watts: float) -> float:
        """
        Set P_idle directly, from a measurement you trust more than the
        automatic one.

        Use this when check_idle_floor.py has established the card's real
        floor. On the reference machine the automatic calibration returned
        21.36 W while a two-minute watch showed the true floor at 2.55 W —
        a difference that halves every energy figure. When you know the
        number, supplying it beats waiting for the detector to find it.
        """
        self.power_idle_w = float(watts)
        self._idle_note = (f"P_idle set manually to {watts:.2f} W rather than "
                           f"measured by calibrate_idle().")
        return self.power_idle_w

    def calibrate_idle(self, samples: int = 20, settle_s: float = 2.0,
                       wait_for_settle: bool = True, max_wait_s: float = 240.0,
                       stability_windows: int = 4):
        """
        Resting power with nothing generating.

        WAITS FOR THE CARD TO ACTUALLY SETTLE rather than sleeping a fixed
        interval. Measured on real hardware: the same GPU with the same
        model resident read 2.66 W when genuinely idle and 34.8 W when
        sampled ~3 seconds after activity. Both are "idle" by the old
        definition. Only the first is idle.

        That matters because work_w = P_active − P_idle. A P_idle inflated
        by 32 W understates every joules-per-token figure by roughly
        32 W × elapsed ÷ tokens, silently, on every reading afterwards.

        A fixed sleep cannot fix this: how long a card takes to drop clocks
        depends on the card, the driver, and what ran before. So this polls
        until consecutive windows agree, and reports if it gave up.
        """
        if not self.available:
            return None
        time.sleep(settle_s)

        def window():
            vals = []
            for _ in range(max(5, samples // 4)):
                p = self._power_w()
                if p is not None:
                    vals.append(p)
                time.sleep(self._sample_interval)
            return (sum(vals) / len(vals)) if vals else None

        settled_note = ""
        if wait_for_settle:
            # TWO CONSECUTIVE WINDOWS IS NOT ENOUGH, measured on real
            # hardware. A card decaying slowly produces adjacent windows
            # that agree within 5% for a long stretch while still falling.
            # That check accepted 21.36 W as settled; a two-minute watch on
            # the same card, same model resident, showed it reaching 2.55 W
            # and holding there. It was only a third of the way down.
            #
            # So: require `stability_windows` consecutive agreements, and
            # require the value not to have fallen materially across the
            # whole run of them. Slow decay produces agreement between
            # neighbours and disagreement across four.
            deadline = time.perf_counter() + max_wait_s
            history = []
            while time.perf_counter() < deadline:
                cur = window()
                if cur is None:
                    break
                history.append(cur)
                if len(history) >= stability_windows:
                    recent = history[-stability_windows:]
                    spread = max(recent) - min(recent)
                    drift = recent[0] - recent[-1]      # positive = still falling
                    if spread <= max(0.5, 0.05 * min(recent)) and drift <= max(0.3, 0.03 * recent[0]):
                        break
            else:
                low = min(history) if history else None
                settled_note = (
                    f"Power was still falling after {max_wait_s}s"
                    + (f" (lowest seen {low:.2f} W, using the current reading)" if low else "")
                    + ". P_idle may be above true idle, which UNDERSTATES every "
                      "joules-per-token figure by the difference. Run "
                      "check_idle_floor.py to find this card's actual floor, then "
                      "pass it as known_idle_w.")

        vals = []
        for _ in range(samples):
            p = self._power_w()
            if p is not None:
                vals.append(p)
            time.sleep(self._sample_interval)
        if not vals:
            return None
        self.power_idle_w = sum(vals) / len(vals)
        self._idle_note = settled_note
        return self.power_idle_w

    def sample_during(self, duration_s: float):
        """Mean power over a window. Run alongside generation."""
        if not self.available:
            return None
        end = time.perf_counter() + duration_s
        vals = []
        while time.perf_counter() < end:
            p = self._power_w()
            if p is not None:
                vals.append(p)
            time.sleep(self._sample_interval)
        return sum(vals) / len(vals) if vals else None

    def observe(self, tokens: int, elapsed_s: float, mean_power_w=None) -> HeatReading:
        if not self.available:
            return HeatReading(None, None, None, None, tokens, elapsed_s,
                               self.baseline_jpt, False, 0, self._reason)
        p_active = mean_power_w if mean_power_w is not None else self._power_w()
        notes = "" if mean_power_w is not None else (
            "Single instantaneous power sample rather than a mean across the run — "
            "noisy. Prefer sample_during().")
        if p_active is None or self.power_idle_w is None or tokens <= 0 or elapsed_s <= 0:
            return HeatReading(None, None, p_active, self.power_idle_w, tokens, elapsed_s,
                               self.baseline_jpt, True, self._process_count(),
                               notes or "calibrate_idle() has not been run.")
        work_w = max(0.0, p_active - self.power_idle_w)
        jpt = work_w * elapsed_s / tokens
        q_aux = (jpt - self.baseline_jpt) if self.baseline_jpt is not None else None
        procs = self._process_count()
        if procs > 1:
            notes += (f" {procs} compute processes on this GPU — power is a whole-device "
                      f"measurement, so a high reading may not be this process's "
                      f"auxiliary work.")
        return HeatReading(round(jpt, 5), round(q_aux, 5) if q_aux is not None else None,
                           round(p_active, 2), round(self.power_idle_w, 2), tokens,
                           round(elapsed_s, 3),
                           round(self.baseline_jpt, 5) if self.baseline_jpt else None,
                           True, procs, notes.strip())

    def record_for_baseline(self, reading: HeatReading) -> None:
        if reading.joules_per_token is not None:
            self._observations.append(reading.joules_per_token)

    def warmup_warning(self) -> str | None:
        """
        Was the first recorded run taken before the GPU boosted its clocks?

        Observed on real hardware: four near-identical prompts produced mean
        power of 66.7 W, 141.4 W, 139.3 W and 138.7 W. The first run drew
        HALF the power of the others because the card had not ramped up yet.
        Since the baseline is the MINIMUM, it locked onto that warm-up run
        and reported Q_aux = +1.106 J/token for a generation that was in
        fact identical to the ones before it.

        Taking the minimum is correct. Taking it from an un-warmed GPU is
        not — the cheapest run has to be the cheapest STEADY-STATE run, or
        the baseline measures the card's idle clocks rather than the floor
        cost of producing output.
        """
        if len(self._observations) < 3:
            return None
        first, rest = self._observations[0], self._observations[1:]
        med = sorted(rest)[len(rest) // 2]
        if med > 0 and first < med * 0.7:
            return (f"First run measured {first:.5f} J/token against a median of "
                    f"{med:.5f} for the rest — under 70% of it. That is a cold-clock "
                    f"artefact, not a cheaper generation. Discard it with "
                    f"drop_first_observation() before setting the baseline, or the "
                    f"floor will be the card's un-boosted state and every subsequent "
                    f"Q_aux will be inflated by the difference.")
        return None

    def drop_first_observation(self) -> bool:
        """Discard the warm-up run. Returns whether anything was dropped."""
        if self._observations:
            self._observations.pop(0)
            return True
        return False

    def set_baseline_from_observations(self, min_runs: int = 3,
                                        allow_warmup: bool = False):
        """
        MINIMUM observed, not mean. The cheapest generation is the closest
        estimate of "output and nothing else"; using the mean would fold
        whatever auxiliary work happened during calibration into the
        baseline, and Q_aux would then measure deviation from typical
        overhead rather than overhead itself.
        """
        if len(self._observations) < min_runs:
            return None
        # Refuse silently-wrong baselines. A cold first run makes the
        # minimum meaningless, and the caller should decide rather than
        # discover it later in an inflated Q_aux.
        warn = self.warmup_warning()
        if warn and not allow_warmup:
            raise ValueError(warn + " Pass allow_warmup=True to override.")
        self.baseline_jpt = min(self._observations)
        return self.baseline_jpt


def check_valve_correlation(pairs: list) -> dict:
    """
    Compare whimsy_kinetic's predicted valve opening against measured Q_aux.
    pairs: (predicted_valve_opening, measured_q_aux). This can refute the
    whimsy model, which is the point of having it.
    """
    if len(pairs) < 4:
        return {"error": f"need at least 4 paired observations, have {len(pairs)}"}
    xs = [p for p, _ in pairs]
    ys = [q for _, q in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return {"error": "no variance in one series — correlation undefined"}
    r = cov / (sx * sy)
    return {"n": n, "correlation": round(r, 4),
            "interpretation": ("Predicted valve opening tracks measured auxiliary energy. "
                               "The whimsy model has external support." if r > 0.5 else
                               "Predicted valve opening does NOT track measured auxiliary "
                               "energy. The whimsy constraint model is not predicting real "
                               "dissipation and should be treated as unvalidated." if r < 0.2
                               else "Weak association. Neither supported nor refuted at this "
                               "sample size.")}


if __name__ == "__main__":
    m = RadiantHeatMonitor()
    print(f"Power telemetry available: {m.available}")
    if not m.available:
        print(f"  {m._reason}\n")
        print("s₁₄ cannot be measured without GPU power telemetry. Nothing is")
        print("substituted — a number computed from something other than energy")
        print("would not be Q_aux.\n")
        print("On a machine with an NVIDIA GPU:")
        print("  m = RadiantHeatMonitor()")
        print("  m.calibrate_idle()                 # nothing generating")
        print("  for each of 3+ clean generation runs:")
        print("      p = m.sample_during(elapsed)")
        print("      r = m.observe(tokens, elapsed, mean_power_w=p)")
        print("      m.record_for_baseline(r)")
        print("  m.set_baseline_from_observations()")
        print("  # from here observe() returns a meaningful q_aux")
    else:
        print(f"Idle power: {m.calibrate_idle(samples=10)} W")
