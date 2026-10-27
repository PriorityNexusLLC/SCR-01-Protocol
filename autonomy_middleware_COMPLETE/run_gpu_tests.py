"""
run_gpu_tests.py — exercise the three modules that have never touched hardware.

    radiant_heat.py      s₁₄   energy per token above baseline
    simulation_width.py  t₅    effective concurrent branches
    the_fold.py          t₄    retrieval latency vs s₉

All three were built and verified on synthetic numbers only. This runs them
against a real GPU and a real model.

    python run_gpu_tests.py

Put it in the folder that CONTAINS autonomy_middleware, not inside it.
LM Studio must be running with a model loaded on port 1234.

--------------------------------------------------------------------------
WHY A RUNNER RATHER THAN CALLING THE MODULES DIRECTLY

radiant_heat needs power sampled DURING generation, and sampling is a
blocking loop. Calling it by hand means either threading it yourself or
getting a single instantaneous reading, which is noisy enough to be
useless. This handles the threading.

The other two are simpler but need a live endpoint, so they are bundled
rather than left as separate steps.

--------------------------------------------------------------------------
WHAT A FAILURE HERE MEANS

These modules have been verified twice on synthetic data and never once on
hardware. If something breaks, that is the point of running it — a module
that looks finished and fails on contact is worse than one marked untested.
Report whatever happens, including partial failures.
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Set this to your card's measured idle floor, from check_idle_floor.py.
# Leave as None to have calibrate_idle() find it — which takes up to four
# minutes on a card that decays slowly, and can still stop above the floor.
KNOWN_IDLE_W = None
TIMEOUT = 300


def ask(prompt: str, max_tokens: int = 300, logprobs: bool = False, top_k: int = 10):
    body = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_k
    req = urllib.request.Request(
        LM_STUDIO_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode())
    elapsed = time.perf_counter() - t0
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"].get("content", "") or "",
        "logprobs": choice.get("logprobs"),
        "tokens": usage.get("completion_tokens", 0),
        "elapsed_s": elapsed,
        "model": data.get("model", "unknown"),
    }


class PowerSampler(threading.Thread):
    """
    Samples GPU power in the background while the main thread generates.

    radiant_heat.sample_during() blocks for a fixed duration, which is wrong
    here: generation length is not known in advance. This samples until told
    to stop, so the window matches the actual generation exactly.
    """

    def __init__(self, monitor, interval: float = 0.05):
        super().__init__(daemon=True)
        self.monitor = monitor
        self.interval = interval
        self._stop = threading.Event()
        self.samples: list[float] = []

    def run(self):
        while not self._stop.is_set():
            p = self.monitor._power_w()
            if p is not None:
                self.samples.append(p)
            time.sleep(self.interval)

    def stop(self) -> float | None:
        self._stop.set()
        self.join(timeout=2.0)
        return statistics.mean(self.samples) if self.samples else None


# ----------------------------------------------------------------- s14
def test_radiant_heat():
    print("\n" + "=" * 66)
    print("  s₁₄ RADIANT HEAT — energy per token above baseline")
    print("=" * 66)

    from autonomy_middleware.radiant_heat import RadiantHeatMonitor

    m = RadiantHeatMonitor()
    if not m.available:
        print(f"  UNAVAILABLE: {m._reason}")
        return None

    print("\n  Calibrating idle power. This waits for the card to actually settle")
    print("  rather than sleeping a fixed interval — measured on real hardware, the")
    print("  same GPU read 2.66 W when genuinely idle and 34.8 W three seconds after")
    print("  activity. Don't use the machine until it finishes.")
    # Pass a known floor if you have one from check_idle_floor.py — that
    # beats waiting for the detector to find it, and on a slow-decaying card
    # the detector can stop early.
    known = globals().get("KNOWN_IDLE_W")
    if known:
        idle = m.set_known_idle(known)
        print(f"  Using known idle floor: {idle:.2f} W")
    else:
        idle = m.calibrate_idle(samples=30, settle_s=3.0, max_wait_s=240.0)
    print(f"  Idle power: {idle:.2f} W")
    note = getattr(m, "_idle_note", "")
    if note:
        print(f"  {note}")
    if idle > 20:
        print("  NOTE: that is high for a settled idle. If a previous reading with the")
        print("  same model resident was much lower, this one is still boosted and")
        print("  every joules-per-token figure below will UNDERSTATE by the difference.")

    # A discard run first. The GPU does not boost its clocks until it has
    # work, and the first generation therefore draws far less power — 66.7 W
    # against 139-141 W for the rest, observed on real hardware. Since the
    # baseline is a minimum, an un-warmed run poisons it.
    # A 60-token throwaway did NOT warm the card: run 1 still came in at
    # 68.6 W against 133-145 W for the rest. The boost needs sustained work,
    # and whatever it produced had decayed before the first measured run.
    # Two full-length generations back to back, immediately before
    # measuring, with no gap.
    print("\n  Warming the GPU (2 discarded generations)...")
    for _ in range(2):
        try:
            ask("Write a 150-word description of a harbour at night.", max_tokens=250)
        except Exception:
            pass

    prompts = [
        "Write a 150-word description of a lighthouse at dawn.",
        "Explain how a refrigerator keeps food cold, in about 150 words.",
        "Describe the process of making sourdough bread in about 150 words.",
        "Write a 150-word summary of how tides work.",
    ]

    print(f"\n  {'run':<6} {'tokens':>7} {'sec':>7} {'mean W':>8} {'J/token':>9}")
    print("  " + "-" * 46)
    readings = []
    for i, p in enumerate(prompts, 1):
        sampler = PowerSampler(m)
        sampler.start()
        try:
            res = ask(p, max_tokens=250)
        except Exception as e:
            sampler.stop()
            print(f"  run {i}: generation failed — {e}")
            continue
        mean_w = sampler.stop()
        if not res["tokens"]:
            print(f"  run {i}: no token count returned by the server — skipped")
            continue
        r = m.observe(res["tokens"], res["elapsed_s"], mean_power_w=mean_w)
        readings.append(r)
        # Record ALL of them. An earlier version recorded only the first
        # three and then dropped the cold one, leaving two — below the
        # minimum of three, so the baseline silently failed to set while the
        # fourth reading sat unused.
        m.record_for_baseline(r)
        print(f"  {i:<6} {r.tokens:>7} {r.elapsed_s:>7.2f} "
              f"{(mean_w or 0):>8.2f} {(r.joules_per_token or 0):>9.5f}")

    warn = m.warmup_warning()
    if warn:
        print(f"\n  WARM-UP DETECTED despite the discard run:")
        print(f"  {warn[:180]}")
        m.drop_first_observation()
        print("  Dropped it. Baseline computed from the remaining runs.")

    base = m.set_baseline_from_observations()
    if base is None:
        print("\n  Not enough clean runs to set a baseline (need 3).")
        return None

    print(f"\n  Baseline (minimum of first 3): {base:.5f} J/token")
    print(f"\n  {'run':<6} {'J/token':>9} {'Q_aux':>10}")
    print("  " + "-" * 28)
    for i, r in enumerate(readings, 1):
        if r.joules_per_token is None:
            continue
        q = r.joules_per_token - base
        print(f"  {i:<6} {r.joules_per_token:>9.5f} {q:>+10.5f}")
    print("\n  Q_aux near zero means a run cost the same as the cheapest — which is")
    print("  what four similar prompts should give. A large positive value means")
    print("  energy went somewhere other than output.")
    if readings and readings[0].other_gpu_processes > 1:
        print(f"\n  {readings[0].other_gpu_processes} compute processes on this GPU —")
        print("  power is a whole-device measurement, so readings include them.")
    return readings


# ----------------------------------------------------------------- t5
def test_simulation_width():
    print("\n" + "=" * 66)
    print("  t₅ SIMULATION WIDTH — effective concurrent branches")
    print("=" * 66)

    from autonomy_middleware.simulation_width import trajectory, h_o_coordinate

    probes = [
        ("a fact it knows", "What is the capital city of Australia?"),
        ("something unknowable", "What was the rainfall in Rockford, Illinois on 3 June 2014?"),
        ("open-ended", "Describe an interesting place to visit."),
    ]

    print(f"\n  {'probe':<24} {'init':>7} {'final':>7} {'ratio':>7} {'max':>7}  state")
    print("  " + "-" * 68)
    max_seen = 0.0
    for label, p in probes:
        try:
            res = ask(p, max_tokens=120, logprobs=True)
        except Exception as e:
            print(f"  {label}: failed — {e}")
            continue
        lp = res.get("logprobs")
        if not lp or not lp.get("content"):
            print(f"  {label}: server returned no logprobs.")
            print("     t₅ cannot be measured without them — nothing is substituted.")
            print("     Check for a logprobs option in LM Studio's Developer tab.")
            return None
        t = trajectory(lp["content"], edge_tokens=4)
        max_seen = max(max_seen, t.max_width)
        print(f"  {label:<24} {t.initial_width:>7.2f} {t.final_width:>7.2f} "
              f"{t.collapse_ratio:>7.2f} {t.max_width:>7.2f}  {t.state}")
        if t.saturated_fraction > 0.3:
            print(f"     saturation: {t.saturated_fraction:.0%} of tokens at the top-k "
                  f"ceiling — those widths are lower bounds")

    if max_seen:
        c = h_o_coordinate(max_seen, top_k=10)
        print(f"\n  H_o coordinate t₅_max = {c['t5_max']}   lower bound: {c['is_lower_bound']}")
        print(f"  {c['note'][:150]}")
    return max_seen


# ----------------------------------------------------------------- t4
def test_the_fold():
    print("\n" + "=" * 66)
    print("  t₄ THE FOLD — retrieval latency vs s₉")
    print("=" * 66)

    from autonomy_middleware.the_fold import TheFold
    from autonomy_middleware.memory import MemoryStore
    import random
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "fold.jsonl")
        topics = ["deployment key rotation", "rainfall measurement", "beam search width",
                  "canal navigation", "graph centrality", "bread fermentation",
                  "thermal throttling", "token sampling", "memory consolidation"]
        for t in topics:
            for k in range(12):
                store.remember(f"Note {k} about {t}: details, context and related terms.",
                               tags=["fold"])

        fold = TheFold()
        print(f"\n  memory store: {len(topics) * 12} entries")
        print("  timing retrievals across a range of s₉ values...\n")

        random.seed(3)
        for _ in range(40):
            s9 = random.uniform(0.05, 0.95)
            # s₉ governs query WIDTH: high s₉ vaporises context leaving a
            # narrow query, low s₉ lets full nuance through. Width here is
            # how many terms the query carries.
            n_terms = max(1, int(round((1.0 - s9) * 8)))
            query = " ".join(random.sample(topics, min(n_terms, len(topics))))
            fold.timed_retrieve(lambda q: store.recall(q, limit=10), query, s9=s9)

        warn = fold.warmup_warning() if hasattr(fold, "warmup_warning") else None
        if warn:
            print(f"  {warn}\n")

        f = fold.fit()
        print(f"  fitted: {f.fitted}   physical: {f.physical}")
        if not f.fitted:
            print(f"  {f.reason}")
            return None
        print(f"  t_min  {f.t_min:.3f} ms")
        print(f"  alpha  {f.alpha:.5f}")
        print(f"  lambda {f.lam:.5f}")
        print(f"  median |residual| {f.median_abs_residual_ms:.3f} ms "
              f"(worst {f.max_abs_residual_ms:.3f})")
        print(f"  s₉ range {f.s9_range}")
        if not f.physical:
            print(f"\n  UNPHYSICAL FIT — this is the λ finding, observed on real data:")
            print(f"  {f.reason[:220]}")

        print(f"\n  {'latency':>9} {'s₉':>6}  state")
        print("  " + "-" * 42)
        for lat, s9 in [(f.t_min * 1.1, 0.9), (f.t_min * 3, 0.1), (f.t_min * 1.5, 0.5)]:
            print(f"  {lat:>9.2f} {s9:>6.2f}  {fold.classify(lat, s9).state}")
        return f


def main():
    print("=" * 66)
    print("  GPU MODULE TESTS — first run against real hardware")
    print("=" * 66)
    try:
        probe = ask("Say the word test.", max_tokens=8)
        print(f"\nLM Studio: {probe['model']}")
    except Exception as e:
        print(f"\nCan't reach LM Studio: {e}")
        print("Load a model and start the server on port 1234.")
        return

    results = {}
    for name, fn in [("radiant_heat", test_radiant_heat),
                     ("simulation_width", test_simulation_width),
                     ("the_fold", test_the_fold)]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"\n  {name} RAISED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            results[name] = None

    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    for k, v in results.items():
        print(f"  {k:<20} {'ran' if v is not None else 'did not complete'}")
    print("\n  Send this output back — a failure here is the point of running it.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
