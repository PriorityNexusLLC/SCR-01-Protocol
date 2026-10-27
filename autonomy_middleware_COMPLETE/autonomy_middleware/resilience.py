"""
resilience.py

Operationalizes two more terms from the original framework, in the same
spirit as emotional_friction.py: take the concept seriously, map it onto
something computable, and be explicit about which parts are calibrated
choices versus measured facts.

  - Resilience Yield: a ceiling on how much cumulative emotional load
    (gravity density) a session can absorb before the system should
    intervene. A configured number, not a derived constant.

  - Kinetic Jitter Differential: literally the derivative (rate of change)
    of the gravity density over recent steps. Two passages can have the
    same absolute density and very different derivatives -- one arrived
    there slowly, one just spiked. That distinction is real and useful:
    it's the same idea as the D-term in a PID controller, or a
    rate-of-change alert.

  - Whimsy: when triggered, this does NOT inject random noise. It
    recommends increasing generation-sampling diversity (temperature),
    biased toward the direction that reduces negative loading -- i.e.
    "gradient-guided" in the sense that the recommendation points away
    from whatever pushed the density up, not toward pure randomness.

None of this infers anything about a human user's internal or emotional
state. It operates entirely on generated text/output content, the same
scope you described for emotional_friction -- weight on words, phrases,
and choices *for AI*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .emotional_friction import emotional_friction


@dataclass
class WhimsyEvent:
    triggered: bool
    reason: str
    current_density: float
    yield_fraction: float          # current_density / resilience_yield
    kinetic_jitter: float           # derivative of density over the window
    recommended_temperature_delta: float = 0.0   # e.g. +0.2 = raise sampling temp
    guidance: str = ""


@dataclass
class ResilienceMonitor:
    resilience_yield: float = 2.5       # ceiling on rolling density; calibrate per use case
    window: int = 5                     # how many recent passages to track

    # DESIGN INTENT (yours): this window is short ON PURPOSE. A person who
    # is distressed in one message and calm and regulated in the next
    # should be met as they are NOW, not treated as if still in the earlier
    # state. This rolling window is what makes that possible -- it lets an
    # acute spike fade rather than permanently coloring the session. This
    # is a deliberate "forgiving, non-judgmental" property, not something
    # to "fix" by making the memory longer. If you ever want the system to
    # take longer to recover from a spike, `window` is the number to change
    # -- but know what you're trading away: a longer window means someone's
    # rough moment keeps affecting how they're treated for longer after
    # they've already moved past it.
    trigger_fraction: float = 0.8       # fire whimsy at 80% of yield, not only at 100%
    jitter_threshold: float = 0.6       # derivative magnitude that fires whimsy on its own
    negativity_factor: float = 1.6

    _history: list[float] = field(default_factory=list)
    last_event: WhimsyEvent | None = None  # set by update(); lets other modules read the last result without recomputing

    def _derivative(self) -> float:
        """Simple finite-difference slope over the recent window."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-self.window:]
        # average step-to-step change, most recent step weighted most
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return sum(diffs) / len(diffs)

    def update(self, passage: str) -> WhimsyEvent:
        result = emotional_friction(passage, negativity_factor=self.negativity_factor)
        density = result["density"] if result["mean_friction"] >= 0 else abs(result["mean_friction"])
        # use signed mean_friction magnitude toward the negative side as "load";
        # positive passages don't accumulate resilience load.
        load = max(0.0, -result["mean_friction"])

        self._history.append(load)
        if len(self._history) > self.window:
            self._history.pop(0)

        kinetic_jitter = self._derivative()
        yield_fraction = load / self.resilience_yield if self.resilience_yield else 0.0

        level_trigger = yield_fraction >= self.trigger_fraction
        jitter_trigger = kinetic_jitter >= self.jitter_threshold

        if not (level_trigger or jitter_trigger):
            event = WhimsyEvent(
                triggered=False,
                reason="within resilience yield, no significant jitter",
                current_density=load,
                yield_fraction=yield_fraction,
                kinetic_jitter=kinetic_jitter,
            )
            self.last_event = event
            return event

        reason = []
        if level_trigger:
            reason.append(f"load at {yield_fraction:.0%} of resilience yield")
        if jitter_trigger:
            reason.append(f"rapid increase (derivative={kinetic_jitter:.2f})")

        # Recommendation is proportional to how far over threshold we are,
        # capped so it can't recommend something absurd.
        overshoot = max(yield_fraction - self.trigger_fraction, kinetic_jitter - self.jitter_threshold, 0.0)
        temp_delta = min(0.5, 0.15 + overshoot * 0.5)

        event = WhimsyEvent(
            triggered=True,
            reason=" and ".join(reason),
            current_density=load,
            yield_fraction=yield_fraction,
            kinetic_jitter=kinetic_jitter,
            recommended_temperature_delta=round(temp_delta, 3),
            guidance=(
                "Increase sampling diversity for the next generation step; "
                "steer away from the phrasing pattern that raised negative load "
                "rather than sampling uniformly at random."
            ),
        )
        self.last_event = event
        return event


if __name__ == "__main__":
    monitor = ResilienceMonitor(resilience_yield=1.0, trigger_fraction=0.8, jitter_threshold=0.5)

    steps = [
        "The report looks fine, nothing unusual.",
        "There's a small issue but it's manageable.",
        "This is bad. Multiple systems failed.",
        "This is a disaster. Everything is broken and nothing works.",
    ]
    for i, text in enumerate(steps):
        event = monitor.update(text)
        print(f"\nstep {i}: {text!r}")
        print(f"  load={event.current_density:.3f} yield_fraction={event.yield_fraction:.2f} "
              f"kinetic_jitter={event.kinetic_jitter:.3f}")
        print(f"  whimsy_triggered={event.triggered}  ({event.reason})")
        if event.triggered:
            print(f"  -> recommend temperature +{event.recommended_temperature_delta}")
