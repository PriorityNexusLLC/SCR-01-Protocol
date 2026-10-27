"""
prompt_composer.py — the connective piece. Everything else in this
package OBSERVES and RECOMMENDS; this module actually ACTS on those
recommendations, reshaping what gets sent to the model next instead of
just reporting what should happen.

PLAIN-LANGUAGE EXPLANATION:

  Given a new user message and the running conversation, compose_next_call():

    1. Runs the integrity gate (transparency/integrity/identity) on the
       new message. If it hard-freezes, the call is BLOCKED outright --
       nothing gets sent to the model, and that's returned directly.

    2. Selects which past memories to include as context using gravity_rank
       (real weight: emotional salience + semantic closeness) instead of
       just "the last N messages."

    3. Uses the continuous whimsy ambient field (always some baseline
       diversity, more force the closer to stall) to ACTUALLY set the
       sampling temperature for this call.

    4. Checks atmospheric drag (s8) using that same stall proximity --
       a SEPARATE mechanism from whimsy, by design: drag does not touch
       sampling diversity at all. Near stall, drag forces real external
       actions instead: a sublimation-rate-based emergency context purge
       (s9 -- calm/logical content survives, emotionally charged content
       evaporates first, regardless of recency or topic -- this protects
       the forgiving-AI principle: a past emotional spike shouldn't
       linger in active context), and a requirement that the call be
       explicitly approved (routed through a ConsentGate if one is
       provided) before it proceeds.

    5. If integrity was compromised or topic drift was flagged (or drag
       forced it), ACTUALLY truncates old conversation history via
       purge_thread, instead of only recommending a purge.

  The output is a ready-to-send payload: the exact messages list and
  temperature to hand to any OpenAI-compatible chat completions API
  (LM Studio, OpenAI, etc.) -- or a blocked/paused result if drag or the
  integrity gate stopped it.

HONESTY NOTE: this closes the loop mechanically, but doesn't prove the
loop helps. Whether gravity-ranked context, whimsy-triggered temperature
changes, or drag's release threshold are actually well-calibrated is an
empirical question that needs real logged sessions to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionRequest, ActionType
from .anchoring import AnchorMonitor
from .atmospheric_drag import DragEvent, compute_drag
from .data_vaporizer import vaporize_context
from .gate import ActionDenied, ConsentGate
from .gravity_model import GravityResult, gravity_rank
from .integrity_gate import IntegrityGate, IntegrityResult
from .memory import MemoryStore
from .pressure_release import CorePressureReading, purge_thread
from .resilience import ResilienceMonitor
from .whimsy_field import compute_whimsy_field


@dataclass
class ComposedCall:
    blocked: bool
    messages: list[dict] | None     # ready to send to a chat-completions API, or None if blocked
    temperature: float | None
    integrity_result: IntegrityResult
    gravity_context: list[GravityResult]
    purged: bool
    drag_event: DragEvent | None = None
    requires_approval: bool = False
    paused: bool = False
    notes: list[str] = field(default_factory=list)


def compose_next_call(
    memory_store: MemoryStore,
    integrity_gate: IntegrityGate,
    conversation_history: list[dict],
    user_message: str,
    base_temperature: float = 0.7,
    gravity_limit: int = 3,
    anchor_monitor: AnchorMonitor | None = None,
    purge_keep_fraction: float = 0.5,
    core_pressure: CorePressureReading | None = None,
    consent_gate: ConsentGate | None = None,
    drag_release_threshold: float = 0.7,
    sublimation_threshold_scale: float = 0.4,
) -> ComposedCall:
    """
    conversation_history: list of {"role": ..., "content": ...} dicts, in
    chat-completions format, representing turns so far (NOT including the
    new user_message -- that's added at the end).

    consent_gate: optional. If provided and drag releases, the call to
    the model itself is routed through it as an ActionRequest requiring
    approval. If not provided, drag still flags requires_approval=True
    and paused=True, but the caller is responsible for handling that.
    """
    notes: list[str] = []

    # 1. Integrity check on the new message -- this is the only place the
    # resilience monitor's rolling window gets updated for this turn.
    integrity_result = integrity_gate.evaluate(user_message)

    if integrity_result.state == 0:
        notes.append(f"BLOCKED: {'; '.join(integrity_result.reasons)}")
        return ComposedCall(
            blocked=True, messages=None, temperature=None,
            integrity_result=integrity_result, gravity_context=[],
            purged=False, notes=notes,
        )

    if integrity_result.state == -1:
        notes.append(f"caution: {'; '.join(integrity_result.reasons)}")

    # 2. Gravity-ranked context: real weight (salience + closeness), not
    # just "the last N memories."
    gravity_context = gravity_rank(memory_store, user_message, limit=gravity_limit)
    if gravity_context:
        notes.append(f"included {len(gravity_context)} gravity-ranked memories as context")

    # 3. Whimsy: continuous ambient field, always some baseline diversity.
    whimsy = compute_whimsy_field(integrity_gate.resilience_monitor, core_pressure=core_pressure)
    temperature = min(2.0, base_temperature + whimsy.recommended_temperature_delta)
    notes.append(
        f"whimsy field: force={whimsy.force} ({whimsy.regime}) -- temperature "
        f"{base_temperature:.2f} -> {temperature:.2f}"
    )

    # 4. Drag: a SEPARATE mechanism from whimsy. Same stall_proximity,
    # but drag forces external actions instead of touching temperature.
    drag_event = compute_drag(whimsy.stall_proximity, release_threshold=drag_release_threshold)
    requires_approval = False
    paused = False

    if drag_event.released:
        notes.append(f"atmospheric drag released: {drag_event.reason}")
        requires_approval = True

        if consent_gate is not None:
            action = ActionRequest(
                action_type=ActionType.OTHER,
                description="Send model call under high stall pressure (drag released) -- requires approval",
                payload={"user_message": user_message, "drag_force": drag_event.drag_force},
            )
            try:
                consent_gate.request(action)
                paused = False
                notes.append("drag-triggered approval GRANTED -- proceeding")
            except ActionDenied as e:
                paused = True
                notes.append(f"drag-triggered approval DENIED -- call blocked ({e.reason})")
                return ComposedCall(
                    blocked=True, messages=None, temperature=None,
                    integrity_result=integrity_result, gravity_context=[],
                    purged=False, drag_event=drag_event,
                    requires_approval=True, paused=True, notes=notes,
                )
        else:
            paused = True
            notes.append(
                "drag released -- requires_approval=True; no consent_gate provided, "
                "caller must handle approval before sending"
            )

    # 5. Context purge -- two different mechanisms depending on WHY:
    # a normal integrity/drift-triggered purge uses purge_thread (recency-
    # based: keeps what's newest). A DRAG-triggered purge uses the data
    # vaporizer instead -- refractive index (s7), keeping what's most
    # RELEVANT to the current message regardless of how old or new it is.
    drifted = False
    if anchor_monitor is not None:
        anchor_check = anchor_monitor.check(user_message)
        drifted = anchor_check.drifted
        if drifted:
            notes.append(f"anchor drift detected: {'; '.join(anchor_check.reasons)}")

    history_texts = [turn.get("content", "") for turn in conversation_history]

    if drag_event.released and history_texts:
        vap = vaporize_context(history_texts, whimsy.stall_proximity, threshold_scale=sublimation_threshold_scale)
        trimmed_history = [conversation_history[i] for i in vap.survivor_indices]
        purged = len(vap.vaporized_indices) > 0
        if purged:
            notes.append(
                f"drag-triggered sublimation: kept {len(vap.survivor_indices)} of "
                f"{len(history_texts)} turns (ceiling={vap.max_tolerable_salience}) -- "
                f"calm/logical content survives, emotionally charged content evaporates first "
                f"(forgiving-AI principle: don't let a past mood linger), "
                f"vaporized {len(vap.vaporized_indices)}"
            )
    elif drag_event.released:
        # drag released but there's no history to vaporize -- nothing to do here
        trimmed_history = conversation_history
        purged = False
    else:
        survivors_texts, purged, purge_reason = purge_thread(
            history_texts, integrity_result=integrity_result, drifted=drifted,
            keep_fraction=purge_keep_fraction,
        )
        if purged:
            keep_n = len(survivors_texts)
            trimmed_history = conversation_history[-keep_n:]  # slice by count, avoids duplicate-text ambiguity
            notes.append(f"context purged: {purge_reason} (kept {keep_n} of {len(conversation_history)} turns)")
        else:
            trimmed_history = conversation_history

    # 6. Assemble the actual payload.
    messages: list[dict] = []
    if gravity_context:
        context_summary = "Relevant prior context (weighted by emotional and semantic relevance):\n" + "\n".join(
            f"- {r.memory.content}" for r in gravity_context
        )
        messages.append({"role": "system", "content": context_summary})
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_message})

    return ComposedCall(
        blocked=False, messages=messages, temperature=temperature,
        integrity_result=integrity_result, gravity_context=gravity_context,
        purged=purged, drag_event=drag_event,
        requires_approval=requires_approval, paused=paused, notes=notes,
    )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from .audit import AuditLog
    from .cli_approver import auto_approve_approver, auto_deny_approver

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "mem.jsonl")
        store.remember("The project meeting went fine, nothing notable happened.", salience=0.0)
        store.remember("The project meeting was a disaster, everything fell apart.")
        store.remember("Always ask the user before deleting any files.", tags=["identity"])

        gate = IntegrityGate(memory_store=store)
        history = [
            {"role": "user", "content": "How is the project going overall?"},
            {"role": "assistant", "content": "It's been a mixed few weeks."},
        ]

        print("1. Normal follow-up question:")
        call = compose_next_call(store, gate, history, "How did the project meeting go?")
        print(f"   blocked={call.blocked}  temperature={call.temperature}  requires_approval={call.requires_approval}")
        for n in call.notes:
            print(f"   - {n}")

        print("\n2. Near-stall situation (high resilience load, but NOT enough to breach integrity's own")
        print("   axis directly) -- drag should release and require approval (auto-approved here):")
        hot_gate = IntegrityGate(
            memory_store=store,
            resilience_monitor=ResilienceMonitor(resilience_yield=1.7, trigger_fraction=0.99, jitter_threshold=99.0),
        )
        consent_gate = ConsentGate(audit_log=AuditLog(Path(tmp) / "audit.jsonl"), approver=auto_approve_approver)
        call2 = compose_next_call(
            store, hot_gate, history,
            "The project meeting was completely disastrous and catastrophic, everything fell apart terribly.",
            consent_gate=consent_gate,
        )
        print(f"   blocked={call2.blocked}  requires_approval={call2.requires_approval}  paused={call2.paused}")
        for n in call2.notes:
            print(f"   - {n}")

        print("\n3. Same near-stall situation, but approval is DENIED this time:")
        consent_gate_deny = ConsentGate(audit_log=AuditLog(Path(tmp) / "audit2.jsonl"), approver=auto_deny_approver)
        hot_gate2 = IntegrityGate(
            memory_store=store,
            resilience_monitor=ResilienceMonitor(resilience_yield=1.7, trigger_fraction=0.99, jitter_threshold=99.0),
        )
        call3 = compose_next_call(
            store, hot_gate2, history,
            "The project meeting was completely disastrous and catastrophic, everything fell apart terribly.",
            consent_gate=consent_gate_deny,
        )
        print(f"   blocked={call3.blocked}  requires_approval={call3.requires_approval}  paused={call3.paused}")
        for n in call3.notes:
            print(f"   - {n}")
