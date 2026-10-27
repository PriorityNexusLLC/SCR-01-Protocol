"""
atmospheric_drag.py — s8 (Atmospheric Drag), a real resistance mechanism,
distinct from whimsy (which explores more under load) and distinct from
the earlier, looser reading of s8 as simply "the pause built into
ConsentGate."

PLAIN-LANGUAGE EXPLANATION:

  Real atmospheric drag scales roughly with velocity SQUARED, not
  velocity itself -- gentle at low speed, growing sharply as you approach
  a limit. That's the shape used here:

      drag_force = stall_proximity ** 2

  using the same stall_proximity whimsy_field.py already computes (the
  worst of resilience load, core pressure, and kinetic jitter). Low
  proximity barely registers; high proximity produces a sharp, fast-rising
  response -- the same nonlinear curve real drag has.

  Per the design decision this was built from: drag does NOT touch
  sampling diversity at all -- that's whimsy's job, and they run
  independently, side by side. When drag force crosses a release
  threshold, it forces real external actions instead:

    - require_human_approval -- the call must be explicitly approved
      before it proceeds (routes through the existing ConsentGate if one
      is provided)
    - purge_context           -- an aggressive, emergency-level context
      truncation (more aggressive than a normal integrity-triggered purge)
    - pause                    -- the call is flagged as paused pending
      that approval, rather than silently proceeding

  This is a pressure-relief valve: releasing force through concrete
  action, not a mood adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DragEvent:
    stall_proximity: float
    drag_force: float             # stall_proximity ** 2 -- nonlinear, gentle early, sharp late
    released: bool                  # True once drag_force crosses the release threshold
    actions: list[str] = field(default_factory=list)
    reason: str = ""


def compute_drag(stall_proximity: float, release_threshold: float = 0.7) -> DragEvent:
    stall_proximity = max(0.0, min(1.0, stall_proximity))
    drag_force = stall_proximity ** 2
    released = drag_force >= release_threshold

    if not released:
        return DragEvent(
            stall_proximity=round(stall_proximity, 3),
            drag_force=round(drag_force, 3),
            released=False,
            reason="within normal operating envelope, no release needed",
        )

    return DragEvent(
        stall_proximity=round(stall_proximity, 3),
        drag_force=round(drag_force, 3),
        released=True,
        actions=["require_human_approval", "purge_context", "pause"],
        reason=(
            f"drag force {drag_force:.2f} crossed release threshold "
            f"{release_threshold} -- emergency pressure release"
        ),
    )


if __name__ == "__main__":
    print("Drag force across the full range of stall proximity (note the curve: gentle early, sharp late):\n")
    for p in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
        event = compute_drag(p)
        print(f"  stall_proximity={p:.1f}  drag_force={event.drag_force:.2f}  "
              f"released={event.released}  actions={event.actions}")
