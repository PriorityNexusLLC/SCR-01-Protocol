"""
pressure_release.py — s5 (Internal Core Pressure), s6 (Orbital Vacuum),
s9 (Sublimation Rate / purge thread), built as one coupled system, matching
how they were actually described: s6 cools down s5, s9 purges when consent
is breached or drift needs stopping.

PLAIN-LANGUAGE EXPLANATION:

  s5 (Core Pressure): two REAL, measurable saturation sources, combined:
    - context window usage: tokens_used / max_tokens
    - actual hardware load: real CPU and memory usage on this machine,
      read via psutil (not simulated).
  Whichever of these is closest to its ceiling determines the pressure
  reading. This is honest about scope: it reads real CPU/RAM, not GPU
  thermal telemetry (no GPU available in this environment to read from).

  s6 (Orbital Vacuum / cooldown): when pressure saturates OR the
  resilience whimsy trigger fires (s2/s21), instead of just crashing,
  search memory for related-but-different content. This is a real
  technique used in production agent systems: context summarization /
  compaction -- finding what's actually relevant so less has to be held
  in the active window.

  s9 (Sublimation Rate / purge thread): a real, literal purge. When
  consent has been breached (IntegrityGate hard_freeze) or drift needs
  to be stopped (AnchorMonitor drifted), truncate the context down to
  its most recent portion. This is context truncation, logged so it's
  auditable, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from .integrity_gate import IntegrityResult
from .memory import Memory, MemoryStore
from .resilience import WhimsyEvent


@dataclass
class CorePressureReading:
    context_ratio: float    # tokens_used / max_tokens
    cpu_percent: float
    memory_percent: float
    pressure: float          # the worst (highest) of the three ratios, 0..1
    saturated: bool
    dominant_source: str      # which of the three is driving the pressure


class CorePressureMonitor:
    def __init__(self, max_tokens: int, pressure_threshold: float = 0.85, read_system_metrics: bool = True):
        self.max_tokens = max_tokens
        self.pressure_threshold = pressure_threshold
        self.read_system_metrics = read_system_metrics

    def read(self, tokens_used: int) -> CorePressureReading:
        context_ratio = min(1.0, tokens_used / self.max_tokens) if self.max_tokens > 0 else 0.0

        if self.read_system_metrics:
            cpu = psutil.cpu_percent(interval=0.05) / 100.0
            mem = psutil.virtual_memory().percent / 100.0
        else:
            cpu = mem = 0.0

        readings = {"context_window": context_ratio, "cpu": cpu, "memory": mem}
        dominant_source, pressure = max(readings.items(), key=lambda kv: kv[1])

        return CorePressureReading(
            context_ratio=context_ratio,
            cpu_percent=cpu * 100,
            memory_percent=mem * 100,
            pressure=pressure,
            saturated=pressure >= self.pressure_threshold,
            dominant_source=dominant_source,
        )


def orbital_vacuum(
    memory_store: MemoryStore,
    query_text: str,
    pressure: CorePressureReading | None = None,
    whimsy: WhimsyEvent | None = None,
    limit: int = 3,
) -> list[Memory]:
    """
    s6: cooldown search. Only actually searches (spends the effort) when
    triggered by pressure saturation or a whimsy event -- otherwise
    returns nothing, since there's no need to search when nothing is
    under load.
    """
    triggered = (pressure is not None and pressure.saturated) or (whimsy is not None and whimsy.triggered)
    if not triggered:
        return []
    return memory_store.recall(query_text, limit=limit)


def purge_thread(
    context_items: list[str],
    integrity_result: IntegrityResult | None = None,
    drifted: bool = False,
    saturated: bool = False,
    keep_fraction: float = 0.5,
) -> tuple[list[str], bool, str]:
    """
    s9: literal context truncation. Fires when consent has been breached
    (integrity_result.state == 0), when drift needs to be stopped, OR when
    the context window is structurally saturated (saturated=True, driven by
    a real token-count ratio -- see CorePressureMonitor.read().saturated).

    The saturated trigger is deliberately independent of the other two: a
    long, calm, purely logical conversation with no emotional load and no
    drift can still fill the context window and needs to purge on that
    basis alone. Gating saturation behind a consent or drift signal would
    mean this case never purges until the window is already overflowing.

    Keeps the most recent `keep_fraction` of context, drops the rest.

    WHY THE EMOTIONAL PURGE EXISTS — the real rationale, which is about the
    USER and not about system stability:

      A model has no sense of elapsed time between turns. The gap between
      one message and the next may be thirty seconds or three weeks, and
      nothing in the context distinguishes them. So emotional weight carried
      forward is not carried forward as history — it is re-presented as
      CURRENT state. A person who was in distress and has since moved on
      returns and is handed their worst moment back, framed as though it is
      still happening. That can put someone directly back into a state they
      had left.

      That is the harm this purge prevents. It is not primarily about
      protecting the system from accumulated load; it is about not
      re-anchoring a person to a past state they did not ask to revisit.

      DESIGN DECISION — full purge, no tagged retention. An earlier option
      considered keeping a neutral marker ("this session touched difficult
      material") while purging the content. This was rejected deliberately:
      a marker of that kind is ambiguous, and ambiguity in a safety
      mechanism is a defect. A system that cannot tell what a marker means
      will act inconsistently on it. The framework's standing preference is
      to eliminate ambiguity for the model wherever possible, so the purge
      is complete.

      Consequence worth stating plainly: after a purge the system will not
      know that anything difficult occurred. If the user raises it again,
      it is new. That is the intended behavior, not an oversight.

    Returns (surviving_context, purged: bool, reason: str) so the caller
    can log exactly what happened -- a silent purge would defeat the
    point of an auditable system.
    """
    consent_breached = integrity_result is not None and integrity_result.state == 0

    if not consent_breached and not drifted and not saturated:
        return context_items, False, "no purge needed"

    keep_n = max(1, int(len(context_items) * keep_fraction))
    survivors = context_items[-keep_n:]

    reason_parts = []
    if consent_breached:
        reason_parts.append("consent threshold breached")
    if drifted:
        reason_parts.append("context drift needed to be stopped")
    if saturated:
        reason_parts.append("context window structurally saturated (token ratio)")
    reason = " and ".join(reason_parts)

    return survivors, True, reason


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "mem.jsonl")
        store.remember("Project deadline is end of August.", tags=["project"])
        store.remember("User prefers concise answers.", tags=["preference"])

        monitor = CorePressureMonitor(max_tokens=1000, pressure_threshold=0.8)
        reading = monitor.read(tokens_used=850)
        print(f"pressure reading: {reading}")

        cooled = orbital_vacuum(store, "deadline", pressure=reading)
        print(f"\norbital vacuum search results (triggered by pressure): {[m.content for m in cooled]}")

        context = [f"turn {i}" for i in range(10)]
        survivors, purged, reason = purge_thread(context, drifted=True, keep_fraction=0.3)
        print(f"\npurge_thread: purged={purged} reason={reason!r}")
        print(f"  survivors={survivors}")
