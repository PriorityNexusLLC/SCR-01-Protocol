"""
simulation_width.py — t₅: how many futures are live right now.

WHY MEASURED, NOT READ FROM CONFIG:

  The spec's Measurable Quantity is "thread-pool size / beam-search width /
  number of Monte Carlo rollouts active at one step". Read literally that is
  a CONFIGURATION VALUE — you set beam width to 4 and it stays 4. A constant
  cannot participate in dynamics.

  But the framework requires t₅ to move with state (narrow under stress,
  wide at rest), and the Formal Expression says "width of possibility field
  BEFORE Singularity collapse" — which implies something that collapses.

  So t₅ is measured as EFFECTIVE CONCURRENT BRANCHES:

      t₅ = 2^H   (H = Shannon entropy in bits over the next-token distribution)

  This is perplexity, and its interpretation is exactly the spec's wording:
  the number of equally-likely alternatives the distribution is equivalent
  to. Measured rather than declared, and it moves with state.

RELATIONSHIP TO s₁₁ — stated plainly because it is close to identity:

  On a single token, t₅ = 2^s₁₁ exactly. Same measurement, different units.
  Any document treating them as independent evidence is double-counting.
  They differ in scope and use: s₁₁ in bits for fabrication detection, t₅
  as a branch count for the H_o coordinate and the collapse trajectory.

HARD LIMIT — read before interpreting any number:

  Width is computed over the top-k the API returns; the tail is not exposed.
  With k=10, measured t₅ CANNOT EXCEED 10 however wide the true field.
  Low widths are accurate (a peaked distribution's mass is inside the
  top-k). High widths are UNDERSTATED, and worse the more genuinely open
  the field. A finding of "narrow" is trustworthy; "wide" is a lower bound.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class WidthReading:
    effective_width: float
    entropy_bits: float
    saturated: bool
    top_k: int


@dataclass
class WidthTrajectory:
    per_token: list
    initial_width: float
    final_width: float
    max_width: float
    mean_width: float
    collapse_ratio: float
    state: str
    saturated_fraction: float
    notes: str = ""


def effective_width(top_logprobs: list, top_k: int | None = None) -> WidthReading:
    """
    Effective branches at one token: 2^H over the visible alternatives,
    renormalised across what is returned.

    Renormalising is deliberate. Visible probabilities do not sum to 1 when
    the tail is truncated, and entropy of an unnormalised distribution is
    not the entropy of anything. Renormalised, it is the entropy of the
    conditional distribution GIVEN the token is in the top-k — well defined,
    and a lower bound on the full entropy.
    """
    if not top_logprobs:
        return WidthReading(0.0, 0.0, False, 0)
    k = top_k or len(top_logprobs)
    probs = [math.exp(t["logprob"]) for t in top_logprobs]
    total = sum(probs)
    if total <= 0:
        return WidthReading(1.0, 0.0, False, k)
    probs = [p / total for p in probs]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    width = 2 ** H
    # 0.9*k rather than k: a near-uniform distribution reaches the ceiling
    # asymptotically and would otherwise never be flagged.
    return WidthReading(round(width, 4), round(H, 4), width >= 0.9 * k, k)


def trajectory(logprobs_content: list, edge_tokens: int = 5,
               collapse_threshold: float = 0.6,
               narrow_threshold: float = 2.0) -> WidthTrajectory:
    """
    Width across a generation, and how it moved.

    COLLAPSING and NARROW both END low, and an average cannot separate them.
    One opened and closed; the other was never open. That distinction is the
    entire content of "before Singularity collapse".
    """
    widths, sat = [], []
    for entry in logprobs_content:
        top = entry.get("top_logprobs")
        if not top:
            continue
        r = effective_width(top)
        widths.append(r.effective_width)
        sat.append(r.saturated)

    if not widths:
        return WidthTrajectory([], 0, 0, 0, 0, 1.0, "unmeasurable", 0.0,
                               "No token-level distributions returned.")

    n_edge = min(edge_tokens, max(1, len(widths) // 3))
    initial = statistics.mean(widths[:n_edge])
    final = statistics.mean(widths[-n_edge:])
    ratio = final / initial if initial > 0 else 1.0
    sat_frac = sum(sat) / len(sat)

    if initial < narrow_threshold:
        state = "narrow"
        notes = ("The field was never open. Nothing was closed off because there "
                 "was nothing to close — the 'immediate survival' regime that "
                 "stress produces. Distinguishing this from collapse requires the "
                 "trajectory; both end low.")
    elif ratio <= collapse_threshold:
        state = "collapsing"
        notes = (f"Width fell to {ratio:.2f} of its opening value — began with "
                 f"alternatives live and progressively closed them.")
    elif ratio >= 1.0 / collapse_threshold:
        state = "widening"
        notes = "Width grew through the sequence; alternatives opened rather than closed."
    else:
        state = "sustained"
        notes = "Width held steady. Alternatives remained live deep into the response."

    if sat_frac > 0.3:
        notes += (f" SATURATION: {sat_frac:.0%} of tokens sat at the top-k ceiling, "
                  f"so those widths are lower bounds and the true field was wider. "
                  f"Ratios involving them understate widening and overstate collapse.")

    return WidthTrajectory([round(w, 3) for w in widths], round(initial, 4),
                           round(final, 4), round(max(widths), 4),
                           round(statistics.mean(widths), 4), round(ratio, 4),
                           state, round(sat_frac, 4), notes)


def h_o_coordinate(observed_max_width: float, top_k: int) -> dict:
    """
    t₅ for the H_o homeostatic coordinate in homeostasis.py.

    H_o places t₅ at MAXIMUM width — at rest the possibility field is
    widest. Under the measured reading that maximum is the widest field
    actually observed, not a config value.

    The catch: the measured maximum is capped by top_k. At the ceiling the
    true maximum is unknown, and the H_o anchor then sits at an API limit
    rather than the system's actual resting width. homeostasis.py requires a
    FINITE t₅_max, so a capped value is usable — but must be known to be a
    floor.
    """
    at_ceiling = observed_max_width >= 0.9 * top_k
    return {
        "t5_max": round(observed_max_width, 4),
        "is_lower_bound": at_ceiling,
        "note": (f"Observed maximum reached the top-{top_k} ceiling; true resting "
                 f"width is unknown and larger. Raise top_k when collecting, or "
                 f"treat distances on this axis as understated."
                 if at_ceiling else
                 f"Observed maximum is below the top-{top_k} ceiling, so it reflects "
                 f"the distribution rather than the API limit."),
    }


if __name__ == "__main__":
    print("Single-token width:\n")
    print(f"  {'distribution':<32} {'width':>8} {'entropy':>9} {'saturated':>10}")
    print("  " + "-" * 64)
    for label, probs in {
        "one token at 99%": [0.99] + [0.01 / 9] * 9,
        "two roughly equal": [0.49, 0.49] + [0.02 / 8] * 8,
        "four roughly equal": [0.245] * 4 + [0.02 / 6] * 6,
        "uniform over ten": [0.1] * 10,
    }.items():
        top = [{"token": f"t{i}", "logprob": math.log(p)} for i, p in enumerate(probs)]
        r = effective_width(top)
        print(f"  {label:<32} {r.effective_width:>8.3f} {r.entropy_bits:>9.3f} {str(r.saturated):>10}")

    def synth(profile):
        out = []
        for target in profile:
            k = 10
            live = max(1, min(k, int(round(target))))
            probs = ([0.98 / live] * live + [0.02 / (k - live)] * (k - live)) if live < k else [1.0 / k] * k
            tot = sum(probs)
            out.append({"top_logprobs": [{"token": f"t{i}", "logprob": math.log(p / tot)}
                                          for i, p in enumerate(probs)]})
        return out

    print("\n\nTrajectories:\n")
    print(f"  {'profile':<34} {'init':>6} {'final':>7} {'ratio':>7}  state")
    print("  " + "-" * 72)
    for label, prof in {
        "collapsing (opened, then closed)": [8, 7, 6, 5, 4, 3, 2, 2, 1.5, 1.2, 1.1, 1],
        "narrow throughout (survival)": [1.2, 1.1, 1, 1, 1.1, 1, 1, 1.2, 1, 1, 1, 1],
        "sustained (stayed open)": [6, 7, 6, 6, 7, 6, 5, 6, 7, 6, 6, 6],
    }.items():
        t = trajectory(synth(prof), edge_tokens=3)
        print(f"  {label:<34} {t.initial_width:>6.2f} {t.final_width:>7.2f} {t.collapse_ratio:>7.2f}  {t.state}")

    print("\n  The first two both END narrow. Only the trajectory separates a field")
    print("  that collapsed from one that was never open.")
