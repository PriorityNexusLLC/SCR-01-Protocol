"""
whimsy_field.py — s21 (Whimsy) as a continuous, ALWAYS-PRESENT ambient
field, not a binary on/off trigger.

PLAIN-LANGUAGE EXPLANATION:

  The earlier version (resilience.py's WhimsyEvent) was binary: it either
  fired or it didn't. That's not what was actually described. Whimsy is
  an ambient field that's ALWAYS present -- even at idle there's a small
  baseline amount, which is what helps find new ideas during ordinary
  processing (innovative exploration, a release valve). Its FORCE grows
  the closer the system gets to stall, combining three real measurements
  that already exist in this package:

    1. resilience load       (s2  -- ResilienceMonitor's yield_fraction)
    2. core/hardware pressure (s5  -- CorePressureMonitor's pressure)
    3. kinetic jitter          (rate of change -- ResilienceMonitor's derivative)

  These combine into one continuous "force" value: interpolated from a
  fixed baseline minimum (never truly zero) up toward maximum as
  whichever of the three is closest to ITS OWN ceiling climbs. Named
  regimes give physical intuition for where the force currently sits,
  matching the scale described:

    idle           -- gentle baseline creative noise, nothing stressed
    earth          -- playful and balanced (moderate, functional)
    star           -- flare-like, energetic bursts (elevated)
    event_horizon  -- extreme, loud, right at the edge of stall

  The recommended sampling-temperature adjustment scales CONTINUOUSLY
  with force now, instead of only firing past one hard threshold.

HONESTY NOTE: the specific baseline value, regime thresholds, and the
"worst of the three" combination rule are calibration choices, not
derived constants -- they encode the shape you described (always present,
grows toward stall) but the exact numbers would need real usage data to
tune properly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pressure_release import CorePressureReading
from .resilience import ResilienceMonitor

BASELINE_FORCE = 0.05  # whimsy never fully disappears, even at total idle

# (upper bound exclusive, regime name) -- checked in order
REGIME_BANDS = [
    (0.30, "idle"),
    (0.60, "earth"),
    (0.85, "star"),
    (1.01, "event_horizon"),  # catches everything up to and including 1.0
]

GUIDANCE = {
    "idle": "Baseline creative noise -- small, exploratory diversity, nothing urgent.",
    "earth": "Balanced, playful diversity -- healthy exploration alongside stable output.",
    "star": "Elevated diversity -- actively searching for a different path; load is building.",
    "event_horizon": "Maximum diversity -- right at the edge of stall; strongly steer away from the current pattern.",
}


@dataclass
class WhimsyField:
    force: float                        # 0..1 continuous ambient field strength, never truly 0
    regime: str                          # idle / earth / star / event_horizon
    stall_proximity: float                # the worst (highest) of the three contributing measurements
    resilience_component: float
    pressure_component: float
    jitter_component: float
    recommended_temperature_delta: float
    guidance: str


def _regime_for(force: float) -> str:
    for upper_bound, name in REGIME_BANDS:
        if force < upper_bound:
            return name
    return "event_horizon"


def force_from_stall_proximity(stall_proximity: float) -> float:
    """
    The pure interpolation formula, extracted so other code (e.g. a
    simulation/test harness) can call the exact same tested logic
    directly with a stall_proximity value, without needing a full
    ResilienceMonitor object.
    """
    stall_proximity = max(0.0, min(1.0, stall_proximity))
    return BASELINE_FORCE + (1.0 - BASELINE_FORCE) * stall_proximity


def compute_whimsy_field(
    resilience_monitor: ResilienceMonitor,
    core_pressure: CorePressureReading | None = None,
    max_temperature_delta: float = 0.6,
) -> WhimsyField:
    """
    Reads the CURRENT state of the resilience monitor (does not advance
    it -- call resilience_monitor.update(text) separately first, e.g. via
    IntegrityGate.evaluate()) and an optional core pressure reading, and
    computes the continuous ambient whimsy field.
    """
    last = resilience_monitor.last_event

    resilience_component = last.yield_fraction if last else 0.0
    raw_jitter = max(0.0, last.kinetic_jitter) if last else 0.0
    jitter_component = (
        raw_jitter / resilience_monitor.jitter_threshold if resilience_monitor.jitter_threshold else 0.0
    )
    pressure_component = core_pressure.pressure if core_pressure else 0.0

    stall_proximity = min(1.0, max(resilience_component, pressure_component, jitter_component))
    force = force_from_stall_proximity(stall_proximity)
    regime = _regime_for(force)
    temp_delta = round(max_temperature_delta * force, 3)

    return WhimsyField(
        force=round(force, 3),
        regime=regime,
        stall_proximity=round(stall_proximity, 3),
        resilience_component=round(resilience_component, 3),
        pressure_component=round(pressure_component, 3),
        jitter_component=round(min(1.0, jitter_component), 3),
        recommended_temperature_delta=temp_delta,
        guidance=GUIDANCE[regime],
    )


if __name__ == "__main__":
    from .pressure_release import CorePressureMonitor

    monitor = ResilienceMonitor(resilience_yield=3.0, trigger_fraction=0.8)

    print("At true idle (nothing processed yet):")
    field = compute_whimsy_field(monitor)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}")

    print("\nAfter a calm, neutral update:")
    monitor.update("Everything is running smoothly today.")
    field = compute_whimsy_field(monitor)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}")

    print("\nAfter a mildly concerning update:")
    monitor.update("There's a minor issue, but nothing serious.")
    field = compute_whimsy_field(monitor)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}")

    print("\nAfter a genuinely distressing update:")
    monitor.update("This is quite bad, several problems have piled up at once.")
    field = compute_whimsy_field(monitor)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}")

    print("\nAfter a full-blown crisis update (resilience load nears the ceiling):")
    monitor.update("This is a total catastrophe, everything is broken and awful, nothing works at all.")
    field = compute_whimsy_field(monitor)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}")

    print("\nSame moment, but with high core/hardware pressure too:")
    pressure_monitor = CorePressureMonitor(max_tokens=1000, read_system_metrics=False)
    hot_pressure = pressure_monitor.read(tokens_used=950)
    field = compute_whimsy_field(monitor, core_pressure=hot_pressure)
    print(f"  force={field.force} regime={field.regime}  temp_delta={field.recommended_temperature_delta}  "
          f"(pressure_component={field.pressure_component})")
