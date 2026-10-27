"""
phase_gates.py — wires the real Load Index (load_index.py) into the
actual s2/s10 phase gate behavior described in the proposal, using the ACL
tightening/lockdown machinery just added to gate.py.

PLAIN-LANGUAGE EXPLANATION:

  load_index.py computes F. gate.py enforces action approval. Until now,
  nothing connected them -- F was a number nobody acted on. This module is
  that connection: every tick, it reads lam, and moves the ConsentGate between
  three real states depending on which side of s2/s10 it's on.

  NOMINAL (lam <= s2_threshold):
    ACL relaxed (if it was tightened). Normal operation.

  S2_WARNING (s2_threshold < lam <= s10_threshold):
    - ACL tightened: every action needs individual re-confirmation, no
      remembered "always allow" rules apply (see gate.py's tighten()).
    - ISL audit block written with full telemetry (lam, H, X_base, context
      saturation, timestamp) -- this happens on every tick while in this
      state, not just on the transition into it, matching the proposal's
      "graduated response" framing.
    - A machine-readable notification string is returned, matching the
      proposal's format, with F renamed to Λ: "[PRIORITY_ALPHA: HIGH_LOAD | Λ=<value>
      | STABILIZATION_MODE_ACTIVE]"
    - recommended_generation_delay_ms is also returned -- a throttle
      recommendation, since this middleware doesn't control the inference
      loop directly and can't literally rate-limit token generation itself.
      The caller (whatever's actually driving generation) is responsible
      for applying it.

  S10_CRITICAL (lam > s10_threshold):
    - The ISL block is written FIRST, before anything else -- matching the
      proposal's explicit ordering requirement ("the ledger is written
      first, the kill executes second").
    - ACL lockdown(): every action denied unconditionally, session rules
      purged, until an operator explicitly calls attempt_reinitialize().
    - engage_integrity_lock() calls whatever integrity_lock_hook was
      configured. By default this is a STUB (_default_integrity_lock_hook)
      that logs what a hard kill WOULD do without touching any real
      process -- safe default if you haven't told the controller what PID
      to target yet. Once you know the real target process ID, pass
      make_real_integrity_lock_hook(pid) instead (see process_control.py)
      to get an actual graceful-terminate-then-SIGKILL hook, tested against
      real spawned processes in this environment.

  Re-initialization is never automatic, per the proposal ("Automatic
  restart is not permitted"). Once locked, only attempt_reinitialize()
  can clear it, and even that refuses unless F has genuinely returned
  below s2_threshold -- calling it while still in S10 territory is a
  no-op with an audited denial, not a bypass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .audit import AuditLog
from .load_index import LoadIndexMonitor, LoadReading, GateState
from .gate import ConsentGate
from .micro_pulse import MicroPulseGovernor
from .process_control import find_process_by_port, terminate_process


def _default_integrity_lock_hook(reading: LoadReading, audit_log: AuditLog) -> None:
    """
    Default s10 hard-halt hook. STUB -- does not kill any real process.
    Logs the fact that a real deployment would execute PURGE_THREAD here,
    so the gap is visible in the audit trail rather than silently absent.
    Use make_real_integrity_lock_hook(pid) instead once you know the
    target process ID.
    """
    audit_log.record({
        "event": "integrity_lock_stub",
        "note": (
            "s10 critical reached -- a real deployment would issue SIGKILL/"
            "TerminateProcess against the inference process here. Process "
            "isolation is not yet implemented; no process was actually "
            "terminated."
        ),
        "F": reading.lam,
    })


def make_real_integrity_lock_hook(
    pid: int, grace_period_s: float = 3.0
) -> Callable[[LoadReading, AuditLog], None]:
    """
    Returns a real integrity-lock hook that actually terminates the given
    process ID when s10 fires -- graceful terminate() first, escalating to
    a real SIGKILL/TerminateProcess if the process doesn't respond within
    grace_period_s. Pass the return value of this as PhaseGateController's
    integrity_lock_hook to replace the stub with the real thing.

    pid is fixed at hook-creation time, not re-discovered on every s10
    event -- if the target process might restart with a new PID between
    controller construction and an s10 event, re-create the controller (or
    this hook) with the current PID rather than assuming staleness is safe.
    """

    def hook(reading: LoadReading, audit_log: AuditLog) -> None:
        result = terminate_process(pid, grace_period_s=grace_period_s)
        audit_log.record({
            "event": "integrity_lock_executed",
            "F": reading.lam,
            "target_pid": pid,
            "found": result.found,
            "terminated_gracefully": result.terminated_gracefully,
            "force_killed": result.force_killed,
            "still_alive": result.still_alive,
            "detail": result.detail,
        })

    return hook


def make_port_based_integrity_lock_hook(
    port: int = 1234, grace_period_s: float = 3.0
) -> Callable[[LoadReading, AuditLog], None]:
    """
    Like make_real_integrity_lock_hook, but re-discovers the target PID
    FRESH on every s10 event by asking who's currently listening on `port`,
    instead of trusting a PID captured once when the hook was built. This
    closes a real gap: if the target server process ever restarts (e.g.
    LM Studio's server is stopped and restarted, or crashes and is
    relaunched) between controller construction and an s10 event, a
    fixed-PID hook would silently target a PID that no longer refers to
    the right process -- or to any process at all.

    port defaults to 1234, LM Studio's own documented default server port
    (confirmed via LM Studio's docs: "LM Studio runs on port 1234 by
    default" / lms server status reports "The server is running on port
    1234" out of the box). If you changed the port in LM Studio's Server
    Settings tab, pass that port instead.

    If no process is found listening on the port at the moment s10 fires,
    this is logged as a failure, not silently ignored or guessed around --
    an unresolvable target is a real problem the ISL should record.
    """

    def hook(reading: LoadReading, audit_log: AuditLog) -> None:
        pid = find_process_by_port(port)
        if pid is None:
            audit_log.record({
                "event": "integrity_lock_failed",
                "F": reading.lam,
                "reason": f"no process found listening on port {port} at time of s10 trigger",
            })
            return

        result = terminate_process(pid, grace_period_s=grace_period_s)
        audit_log.record({
            "event": "integrity_lock_executed",
            "F": reading.lam,
            "target_port": port,
            "target_pid": pid,
            "found": result.found,
            "terminated_gracefully": result.terminated_gracefully,
            "force_killed": result.force_killed,
            "still_alive": result.still_alive,
            "detail": result.detail,
        })

    return hook


@dataclass
class PhaseGateResult:
    reading: LoadReading
    state_changed: bool
    notification: str | None            # machine-readable header, only set on s2/s10
    recommended_generation_delay_ms: float  # from the μ(t) governor; 0 if no governor supplied


class PhaseGateController:
    def __init__(
        self,
        load_monitor: LoadIndexMonitor,
        consent_gate: ConsentGate,
        audit_log: AuditLog,
        integrity_lock_hook: Callable[[LoadReading, AuditLog], None] | None = None,
        micro_pulse_governor: "MicroPulseGovernor | None" = None,
    ):
        """
        micro_pulse_governor: the t₁ Micro-Pulse governor (micro_pulse.py).
            When supplied, throttle depth is computed as a closed loop that
            modulates μ(t) to hold s₅ under the s₂ ceiling, per the
            Dimensional Derivation Table's t₁ specification. Its baseline
            must already be calibrated from a real at-rest sample.

            When omitted, no throttle recommendation is produced at all.
            This is deliberate: the previous behavior -- two hardcoded
            delay constants (250ms at s₂, 5000ms at s₁₀) -- was open-loop,
            applied regardless of whether it brought s₅ back under the
            ceiling, and nothing measured whether it worked. Returning 0
            and letting the caller notice the governor is missing is more
            honest than emitting a number with no feedback behind it.
        """
        self.load_monitor = load_monitor
        self.consent_gate = consent_gate
        self.audit_log = audit_log
        self.integrity_lock_hook = integrity_lock_hook or _default_integrity_lock_hook
        self.micro_pulse_governor = micro_pulse_governor
        self._last_gate: GateState | None = None

    def _throttle(self, reading: LoadReading) -> tuple[float, dict | None]:
        """
        Ask the μ(t) governor for the current modulation. Returns
        (delay_ms, telemetry_dict_or_None) so the ISL entry can record the
        modulation ratio alongside lam, which is what the table asks to be
        logged per inference step.
        """
        if self.micro_pulse_governor is None:
            return 0.0, None
        pulse = self.micro_pulse_governor.update(lam=reading.lam)
        return pulse.inter_pulse_delay_ms, {
            "mu": pulse.mu,
            "mu_baseline": pulse.mu_baseline,
            "modulation_ratio": pulse.modulation_ratio,
            "target_ratio": pulse.target_ratio,
            "inter_pulse_delay_ms": pulse.inter_pulse_delay_ms,
            "drag_ms_per_token": pulse.drag_ms_per_token,
            "throttle_reason": pulse.reason,
        }

    def tick(self, context_tokens: int = 0) -> PhaseGateResult:
        reading = self.load_monitor.update(context_tokens=context_tokens)
        state_changed = reading.gate != self._last_gate
        notification = None
        delay_ms, pulse_telemetry = self._throttle(reading)

        if reading.gate == GateState.NOMINAL:
            self.consent_gate.relax()

        elif reading.gate == GateState.S2_WARNING:
            self.consent_gate.tighten()
            block = {
                "event": "s2_warning_block",
                "F": reading.lam,
                "H": reading.H,
                "X_base": reading.X_base,
                "context_saturation": reading.context_saturation,
                "components": reading.components,
            }
            if pulse_telemetry:
                block["micro_pulse"] = pulse_telemetry
            self.audit_log.record(block)
            notification = f"[PRIORITY_ALPHA: HIGH_LOAD | Λ={reading.lam} | STABILIZATION_MODE_ACTIVE]"

        elif reading.gate == GateState.S10_CRITICAL:
            # Ledger written first, per the proposal's explicit ordering
            # requirement -- before lockdown, before the integrity-lock hook.
            block = {
                "event": "s10_critical_block",
                "F": reading.lam,
                "H": reading.H,
                "X_base": reading.X_base,
                "context_saturation": reading.context_saturation,
                "components": reading.components,
            }
            if pulse_telemetry:
                block["micro_pulse"] = pulse_telemetry
            self.audit_log.record(block)
            self.consent_gate.lockdown(reason=f"s10 critical: F={reading.lam}")
            self.integrity_lock_hook(reading, self.audit_log)
            notification = f"[PRIORITY_ALPHA: INTEGRITY_LOCK | Λ={reading.lam} | HALT_SEQUENCE_ENGAGED]"

        self._last_gate = reading.gate
        return PhaseGateResult(
            reading=reading,
            state_changed=state_changed,
            notification=notification,
            recommended_generation_delay_ms=delay_ms,
        )

    def attempt_reinitialize(self) -> bool:
        """
        Operator-invoked only. Never called automatically -- per the
        proposal, "automatic restart is not permitted." Returns True if
        the lockdown was actually cleared, False if F is still at or
        above s2_threshold (in which case the refusal itself is audited,
        not silently ignored).
        """
        current = self.load_monitor.update()
        if current.lam >= self.load_monitor.s2_threshold:
            self.audit_log.record({
                "event": "reinitialize_denied",
                "reason": f"F={current.lam} still at or above s2_threshold={self.load_monitor.s2_threshold}",
            })
            return False

        self.consent_gate.unlock()
        self.audit_log.record({"event": "reinitialize_approved", "F": current.lam})
        return True


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from .actions import ActionRequest, ActionType
    from .gate import Decision
    from .load_index import HardwareReading

    def auto_approver(action: ActionRequest) -> Decision:
        return Decision(approved=True, reason="demo auto-approve")

    with tempfile.TemporaryDirectory() as tmp:
        audit = AuditLog(Path(tmp) / "audit.jsonl")
        consent = ConsentGate(audit_log=audit, approver=auto_approver)
        friction = LoadIndexMonitor(max_context_tokens=62000)
        friction.calibrate(samples=3)
        friction.X_base = 1.0  # fixed for a clean demo: H below IS F directly

        controller = PhaseGateController(friction, consent, audit)

        def forced_reading(lam: float) -> LoadReading:
            hw = HardwareReading(gpu_util=None, gpu_thermal_c=None, cpu_util=lam,
                                  context_saturation=0.1, context_mode="real_token_ratio",
                                  context_tokens=100, gpu_available=False, timestamp=0.0)
            gate = (GateState.S10_CRITICAL if lam > friction.s10_threshold
                    else GateState.S2_WARNING if lam > friction.s2_threshold
                    else GateState.NOMINAL)
            return LoadReading(lam=lam, H=lam, X_base=1.0, gate=gate,
                                    dominant_source="cpu_util", components={"cpu_util": F},
                                    hardware=hw, calibrated=True,
                                    context_saturation=0.1, context_mode="real_token_ratio",
                                    context_saturated=False)

        print("Simulating rising and falling friction through the real controller.tick() path:")
        for label, forced_F in [("nominal", 0.8), ("s2 warning", 1.6), ("s10 critical", 2.5), ("back to nominal", 0.5)]:
            friction.update = lambda context_tokens=0, _F=forced_F: forced_reading(_F)
            result = controller.tick()
            print(f"  {label:<16} F={forced_F}  gate={result.reading.gate.value:<12} "
                  f"acl_tightened={consent._tightened}  acl_locked={consent._locked}  "
                  f"delay_ms={result.recommended_generation_delay_ms}  "
                  f"notification={result.notification}")

        print(f"\nAttempting reinitialize while still locked at lam=0.5 (should succeed, since 0.5 < s2_threshold)...")
        ok = controller.attempt_reinitialize()
        print(f"  reinitialize succeeded: {ok}  (acl_locked={consent._locked})")

        print(f"\nAudit trail ({len(audit.read_all())} entries):")
        for entry in audit.read_all():
            print(f"  {entry['payload']['event']}")
