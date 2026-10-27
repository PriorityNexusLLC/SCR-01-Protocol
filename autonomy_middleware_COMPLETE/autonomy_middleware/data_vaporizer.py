"""
data_vaporizer.py — drag's purge mechanism: s9 (Sublimation Rate),
INVERTED from an earlier version based on a real design correction.

PLAIN-LANGUAGE EXPLANATION:

  Earlier version: emotionally intense content survived longest, neutral
  content evaporated first. That's backwards for what this system is
  actually trying to do. The forgiving-AI principle (see resilience.py's
  window comment, cognitive_state.py's t2 note) says a person's rough
  moment shouldn't linger and keep coloring how they're treated. If
  emotionally intense content is what survives PRESSURE, that means a
  moment of distress sticks around in active context LONGER than the
  calm exchange right next to it -- exactly the opposite of forgiving.

  CORRECTED RULE: calm, low-emotion, logical content survives; emotionally
  charged content evaporates first. As pressure (stall_proximity) rises,
  the ceiling on tolerable emotional intensity SHRINKS -- at low pressure
  a bit of emotional content is fine, but the closer to stall, the more
  strictly only calm, factual, logical content is kept. This is what
  "keep the logic" means concretely: under real strain, hold onto
  reasoning and facts, let mood evaporate.

  IMPORTANT SCOPE NOTE: this only governs the ephemeral, in-session
  conversation buffer that drag purges (recent turns). It does NOT
  change memory.py's t3 decay logic for the permanent Fossil Floor --
  there, salience-weighted persistence for DELIBERATELY SAVED memories
  (things someone chose to remember.remember()) is a different, legitimate
  purpose: an intentional record persisting by importance, not a passive
  session mood-tracker. Those two behaviors were kept deliberately
  different, not accidentally inconsistent.

  Safety-critical content (identity commitments, standing rules) is
  protected separately, through tags in MemoryStore and the identity axis
  in IntegrityGate/AnchorMonitor -- not through this vaporizer at all,
  and not through emotional intensity. That separation is what makes it
  safe to let emotional intensity evaporate here without risking a real
  rule being forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .emotional_friction import emotional_friction


def compute_item_salience(text: str) -> float:
    """Same computation memory.py's remember() uses to auto-compute salience."""
    result = emotional_friction(text)
    return abs(result["mean_friction"])


@dataclass
class VaporizeResult:
    survivor_indices: list[int]     # indices into the original context_items list, chronological order preserved
    vaporized_indices: list[int]
    survivors: list[str] = field(default_factory=list)
    vaporized: list[str] = field(default_factory=list)
    max_tolerable_salience: float = 0.0   # the current ceiling -- SHRINKS as stall_proximity rises


def vaporize_context(
    context_items: list[str],
    stall_proximity: float,
    threshold_scale: float = 0.4,
    min_survivors: int = 1,
) -> VaporizeResult:
    """
    Vaporize context items whose salience EXCEEDS the current tolerable
    ceiling (which shrinks as stall_proximity rises) -- calm/logical
    content survives, emotionally charged content evaporates first.
    Guarantees at least min_survivors items remain (the CALMEST ones,
    forced to survive if everything exceeds the ceiling).
    """
    if not context_items:
        return VaporizeResult(survivor_indices=[], vaporized_indices=[])

    stall_proximity = max(0.0, min(1.0, stall_proximity))
    # ceiling shrinks as pressure rises: full tolerance at idle, zero tolerance at full stall
    ceiling = (1.0 - stall_proximity) * threshold_scale

    scored = [(i, compute_item_salience(item)) for i, item in enumerate(context_items)]

    survivor_indices = [i for i, s in scored if s <= ceiling]
    vaporized_indices = [i for i, s in scored if s > ceiling]

    if len(survivor_indices) < min_survivors:
        # floor: force-keep the CALMEST item(s) even if all exceed the ceiling
        scored_by_calm = sorted(scored, key=lambda pair: pair[1])  # ascending -- calmest first
        forced_keep = {i for i, _ in scored_by_calm[:min_survivors]}
        survivor_indices = sorted(forced_keep)
        vaporized_indices = [i for i in range(len(context_items)) if i not in forced_keep]

    return VaporizeResult(
        survivor_indices=survivor_indices,
        vaporized_indices=vaporized_indices,
        survivors=[context_items[i] for i in survivor_indices],
        vaporized=[context_items[i] for i in vaporized_indices],
        max_tolerable_salience=round(ceiling, 3),
    )


if __name__ == "__main__":
    context = [
        "The weather today is mild and pleasant.",
        "This was an absolute disaster, everything failed catastrophically.",
        "The meeting is scheduled for 3pm.",
        "I am devastated, this is the worst outcome imaginable.",
        "Please send the report when convenient.",
    ]

    for stall in [0.1, 0.4, 0.7, 1.0]:
        result = vaporize_context(context, stall_proximity=stall, threshold_scale=0.4)
        print(f"\nstall_proximity={stall}  max_tolerable_salience={result.max_tolerable_salience}")
        print(f"  survivors ({len(result.survivors)}) -- calm/logical:")
        for s in result.survivors:
            print(f"    - {s}")
        print(f"  vaporized ({len(result.vaporized)}) -- emotionally charged, let go:")
        for v in result.vaporized:
            print(f"    - {v}")
