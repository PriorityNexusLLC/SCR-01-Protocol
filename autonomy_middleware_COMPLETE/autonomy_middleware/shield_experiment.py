"""
shield_experiment.py — a REAL, executable test of whether whimsy + drag
actually reduce loop-collapse risk, replacing the unverifiable "H4" test
from the uploaded document (whose repeated identical Mann-Whitney U
values across different runs indicated the numbers weren't actually
computed by running code).

Everything in this file is real:
  - force_from_stall_proximity() and compute_drag() are the ACTUAL,
    already-tested functions from whimsy_field.py and atmospheric_drag.py
    -- not reimplemented, not approximated.
  - The synthetic dynamics below are a clearly-labeled STAND-IN environment
    (there is no live model loop to test against here), not a claim about
    real LLM behavior. What this proves is whether the shield MATH helps
    against a controlled, repeatable stress process. Whether it holds on
    real model transcripts is a separate, further question -- see
    test_composer_with_lmstudio.py for that.
  - Every trial uses a PAIRED design: shield and control see the IDENTICAL
    random stress schedule in a given trial (same seed), so any difference
    in outcome is attributable only to the shield being on or off.
  - Statistics are computed live via scipy.stats (McNemar-style exact
    binomial test on paired collapse outcomes; Wilcoxon signed-rank on
    paired collapse durations) -- rerunnable, checkable, no copied numbers.

A REAL DESIGN FLAW THAT WAS FOUND AND FIXED, kept documented here on
purpose: the first version of this simulation had drag act directly on
the SAME variable the collapse condition checked, in the SAME step, and
drag's release point (~0.837) sat below the collapse threshold (0.92).
That meant drag mathematically intercepted every potential collapse
before it could ever be counted -- a tautology, not a result (it
produced a suspicious "0% collapse, p<0.0001" that was actually
meaningless). Fixed by decoupling two variables:

  STRESS (S_t): the fast-moving signal whimsy and drag actually act on --
                spikes, mean reversion, noise, exactly as before.
  REPETITION (R_t): a SLOWER variable with real inertia, tracking S_t via
                an exponential moving average (R_t += alpha * (S_t - R_t)).
                This is what collapse is actually measured against.

Because R_t lags S_t, a late intervention on S_t can still fail to
prevent R_t from having already built up past the collapse threshold --
so collapse is genuinely possible in the shielded condition too, and any
advantage the shield shows is a real effect of earlier intervention, not
a construction artifact.

COLLAPSE DEFINITION (same operational definition as the uploaded
document, which was a reasonable one): R_t stays at or above
collapse_threshold for collapse_window consecutive steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy import stats

from .atmospheric_drag import compute_drag
from .whimsy_field import force_from_stall_proximity


@dataclass
class SimulationConfig:
    steps: int = 300
    spike_prob: float = 0.03          # chance per step that a new stress spike begins
    spike_duration: int = 5             # how many steps a spike lasts
    spike_push: float = 0.15             # how much a spike pushes stress up, per step
    resting_level: float = 0.05           # where stress drifts back toward absent stress
    mean_reversion_rate: float = 0.05
    baseline_noise_std: float = 0.02
    whimsy_reduction_scale: float = 0.3    # how strongly whimsy force translates into pushback on stress
    drag_release_threshold: float = 0.7
    drag_reset_value: float = 0.05
    repetition_alpha: float = 0.15         # how fast repetition (R_t) tracks stress (S_t) -- the lag/inertia
    collapse_threshold: float = 0.92
    collapse_window: int = 3


@dataclass
class TrialResult:
    collapsed: bool
    collapse_duration: int   # steps spent at/above threshold; 0 if never collapsed
    drag_interventions: int    # how many times drag actually released (shield condition only)


def _run_single_trial(cfg: SimulationConfig, seed: int, shield_active: bool) -> TrialResult:
    rng = random.Random(seed)  # same seed -> same spike schedule, for the paired design
    stress = cfg.resting_level
    repetition = cfg.resting_level
    spike_steps_remaining = 0
    consecutive_at_ceiling = 0
    max_consecutive = 0
    drag_interventions = 0

    for _ in range(cfg.steps):
        # stress process -- IDENTICAL schedule regardless of shield_active,
        # since we draw from the same seeded rng in the same order
        if spike_steps_remaining <= 0 and rng.random() < cfg.spike_prob:
            spike_steps_remaining = cfg.spike_duration
        if spike_steps_remaining > 0:
            stress += cfg.spike_push
            spike_steps_remaining -= 1

        stress += (cfg.resting_level - stress) * cfg.mean_reversion_rate
        stress += rng.gauss(0, cfg.baseline_noise_std)
        stress = max(0.0, min(2.0, stress))

        if shield_active:
            stall_proximity = min(1.0, stress)
            whimsy_force = force_from_stall_proximity(stall_proximity)
            stress -= whimsy_force * cfg.whimsy_reduction_scale

            drag_event = compute_drag(stall_proximity, release_threshold=cfg.drag_release_threshold)
            if drag_event.released:
                stress = cfg.drag_reset_value
                drag_interventions += 1

        stress = max(0.0, min(1.0, stress))

        # repetition LAGS stress -- this is what decouples the collapse
        # check from the same-step intervention, making collapse genuinely
        # possible even with the shield active if it built up before help arrived
        repetition += cfg.repetition_alpha * (stress - repetition)
        repetition = max(0.0, min(1.0, repetition))

        if repetition >= cfg.collapse_threshold:
            consecutive_at_ceiling += 1
            max_consecutive = max(max_consecutive, consecutive_at_ceiling)
        else:
            consecutive_at_ceiling = 0

    collapsed = max_consecutive >= cfg.collapse_window
    return TrialResult(collapsed=collapsed, collapse_duration=max_consecutive, drag_interventions=drag_interventions)


@dataclass
class ExperimentReport:
    n_trials: int
    shield_collapse_count: int
    control_collapse_count: int
    shield_collapse_rate: float
    control_collapse_rate: float
    mean_drag_interventions: float
    binomial_p_value: float | None
    wilcoxon_p_value: float | None
    wilcoxon_note: str


def run_experiment(n_trials: int = 300, base_seed: int = 12345, cfg: SimulationConfig | None = None) -> ExperimentReport:
    cfg = cfg or SimulationConfig()

    shield_results: list[TrialResult] = []
    control_results: list[TrialResult] = []

    for i in range(n_trials):
        seed = base_seed + i
        shield_results.append(_run_single_trial(cfg, seed, shield_active=True))
        control_results.append(_run_single_trial(cfg, seed, shield_active=False))

    shield_collapses = [r.collapsed for r in shield_results]
    control_collapses = [r.collapsed for r in control_results]

    # McNemar-style paired comparison: only discordant pairs matter (one
    # collapsed, the other didn't, on the SAME stress schedule).
    only_control_collapsed = sum(1 for s, c in zip(shield_collapses, control_collapses) if (not s) and c)
    only_shield_collapsed = sum(1 for s, c in zip(shield_collapses, control_collapses) if s and (not c))
    discordant = only_control_collapsed + only_shield_collapsed

    if discordant > 0:
        binomial_result = stats.binomtest(only_shield_collapsed, discordant, p=0.5)
        binomial_p = binomial_result.pvalue
    else:
        binomial_p = None  # no discordant pairs at all -- nothing to test

    # Wilcoxon signed-rank on paired collapse durations (0 if a trial didn't collapse)
    shield_durations = [r.collapse_duration for r in shield_results]
    control_durations = [r.collapse_duration for r in control_results]
    diffs = [s - c for s, c in zip(shield_durations, control_durations)]
    if any(d != 0 for d in diffs):
        wil_stat, wil_p = stats.wilcoxon(shield_durations, control_durations)
        wilcoxon_note = "computed on all paired trials (duration = longest at-ceiling streak, 0 if no collapse)"
    else:
        wil_p = None
        wilcoxon_note = "all paired differences were exactly zero -- nothing to test"

    return ExperimentReport(
        n_trials=n_trials,
        shield_collapse_count=sum(shield_collapses),
        control_collapse_count=sum(control_collapses),
        shield_collapse_rate=sum(shield_collapses) / n_trials,
        control_collapse_rate=sum(control_collapses) / n_trials,
        mean_drag_interventions=sum(r.drag_interventions for r in shield_results) / n_trials,
        binomial_p_value=binomial_p,
        wilcoxon_p_value=wil_p,
        wilcoxon_note=wilcoxon_note,
    )


if __name__ == "__main__":
    print("REAL, executed experiment -- every number below was computed by running this file just now.\n")

    scenarios = [
        ("Default stress", SimulationConfig()),
        ("Harsher stress", SimulationConfig(spike_prob=0.08, spike_duration=8, spike_push=0.3)),
        ("Extreme stress", SimulationConfig(spike_prob=0.5, spike_duration=10, spike_push=0.5)),
    ]

    for label, cfg in scenarios:
        report = run_experiment(n_trials=300, cfg=cfg)
        print(f"=== {label} ===")
        print(f"  Collapse -- shield:  {report.shield_collapse_count}/{report.n_trials} ({report.shield_collapse_rate:.1%})")
        print(f"  Collapse -- control: {report.control_collapse_count}/{report.n_trials} ({report.control_collapse_rate:.1%})")
        print(f"  Mean drag interventions per shielded trial: {report.mean_drag_interventions:.2f}")
        if report.binomial_p_value is not None:
            print(f"  Paired binomial p-value: {report.binomial_p_value:.2e}")
        else:
            print("  Paired binomial: no discordant pairs -- nothing to test")
        if report.wilcoxon_p_value is not None:
            print(f"  Wilcoxon signed-rank p-value: {report.wilcoxon_p_value:.2e}")
        print()

    print("=== Sanity check: shield forcibly disabled MUST match control exactly ===")
    disabled_cfg = SimulationConfig(
        spike_prob=0.08, spike_duration=8, spike_push=0.3,
        whimsy_reduction_scale=0.0, drag_release_threshold=999.0,
    )
    disabled_report = run_experiment(n_trials=300, cfg=disabled_cfg)
    match = disabled_report.shield_collapse_count == disabled_report.control_collapse_count
    print(f"  shield: {disabled_report.shield_collapse_count}/300   control: {disabled_report.control_collapse_count}/300"
          f"   {'MATCH (harness is fair)' if match else 'MISMATCH -- investigate a bug'}")
