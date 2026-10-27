"""
The consent gate itself.

An agent (powered by any LLM, any framework) proposes an ActionRequest.
The gate decides, synchronously, whether that action is allowed to run:

  - LOW risk actions can be configured to auto-pass (e.g. read-only calls)
  - MEDIUM/HIGH risk actions are held and routed to an "approver" callback,
    which is where a human actually says yes or no
  - Every decision — approved, denied, or auto-passed — is written to the
    audit log before the action is allowed to execute

The gate does not know or care what LLM, framework, or prompt produced the
action. It only sees the ActionRequest. That's what makes it model-agnostic:
you can point GPT, Claude, a local Llama model, or a hand-written script at
the same gate and get the same governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .actions import ActionRequest, RiskLevel
from .audit import AuditLog


class ActionDenied(PermissionError):
    """Raised when a gated action is rejected by the approver."""

    def __init__(self, action: ActionRequest, reason: str = "denied by approver"):
        self.action = action
        self.reason = reason
        super().__init__(f"Action {action.id} ({action.description!r}) denied: {reason}")


@dataclass
class Decision:
    approved: bool
    reason: str
    remember: bool = False  # if True, gate will auto-decide matching future actions this session


# An approver is any callable that looks at an ActionRequest and decides.
# The default one (cli_approver.prompt_approver) asks a human on stdin/stdout;
# in a real deployment you'd swap in a Slack approval bot, a web UI callback,
# a mobile push notification, etc. — the gate doesn't care, as long as it
# gets back a Decision.
Approver = Callable[[ActionRequest], Decision]


@dataclass
class ConsentGate:
    audit_log: AuditLog
    approver: Approver
    auto_allow_below: RiskLevel = RiskLevel.LOW  # risk levels at/below this pass without asking
    _session_rules: dict[str, bool] = field(default_factory=dict)  # "action_type:risk" -> approved
    _tightened: bool = field(default=False)   # s2 warning state: bypass remembered rules
    _locked: bool = field(default=False)        # s10 critical state: deny everything until unlock()
    _lock_reason: str | None = field(default=None)

    _RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}

    def _rule_key(self, action: ActionRequest) -> str:
        return f"{action.action_type.value}:{action.resolved_risk().value}"

    # ------------------------------------------------------------------
    # s2 Warning state: tighten the authorization window down to
    # one-action-per-confirmation by bypassing remembered "always allow"
    # session rules -- every action goes back through the real approver,
    # even ones that would normally auto-pass on a remembered rule.
    # ------------------------------------------------------------------
    def tighten(self) -> None:
        if not self._tightened:
            self._tightened = True
            self.audit_log.record({"event": "acl_tightened", "reason": "s2 friction warning"})

    def relax(self) -> None:
        if self._tightened:
            self._tightened = False
            self.audit_log.record({"event": "acl_relaxed", "reason": "friction returned to nominal"})

    # ------------------------------------------------------------------
    # s10 Critical state: full lockdown. Every request is denied
    # unconditionally until unlock() is called. Session-remembered rules
    # are purged, not just bypassed -- matching the proposal's "ACL
    # buffer purged, no queued actions survive a thread purge." Purge
    # count is logged so the ledger records what was cleared, not just
    # that a clear happened.
    # ------------------------------------------------------------------
    def lockdown(self, reason: str) -> None:
        purged_count = len(self._session_rules)
        self._session_rules.clear()
        self._locked = True
        self._lock_reason = reason
        self.audit_log.record({
            "event": "acl_lockdown",
            "reason": reason,
            "session_rules_purged": purged_count,
        })

    def unlock(self) -> None:
        if self._locked:
            self._locked = False
            self._lock_reason = None
            self.audit_log.record({"event": "acl_unlocked"})

    def request(self, action: ActionRequest) -> bool:
        """
        Ask the gate to approve an action. Returns True if approved.
        Raises ActionDenied if rejected. Always logs the outcome.
        """
        if self._locked:
            self.audit_log.record({
                "event": "action_blocked_lockdown",
                "action": action.to_dict(),
                "reason": self._lock_reason,
            })
            raise ActionDenied(action, reason=f"ACL is in lockdown: {self._lock_reason}")

        risk = action.resolved_risk()

        # 1. Auto-allow low-risk actions below the configured threshold.
        if self._RISK_ORDER[risk] <= self._RISK_ORDER[self.auto_allow_below]:
            self.audit_log.record({
                "event": "action_auto_allowed",
                "action": action.to_dict(),
            })
            return True

        # 2. Check session-remembered rules ("always allow this type this
        # session") -- but ONLY if the gate isn't currently tightened. While
        # tightened (s2 warning), every action requires individual
        # re-confirmation regardless of any remembered rule.
        rule_key = self._rule_key(action)
        if not self._tightened and rule_key in self._session_rules:
            approved = self._session_rules[rule_key]
            self.audit_log.record({
                "event": "action_session_rule_applied",
                "action": action.to_dict(),
                "approved": approved,
            })
            if not approved:
                raise ActionDenied(action, reason="matches a session 'always deny' rule")
            return True

        # 3. Otherwise, hold the action and route to the approver.
        self.audit_log.record({"event": "action_pending", "action": action.to_dict()})
        decision = self.approver(action)

        if decision.remember and not self._tightened:
            self._session_rules[rule_key] = decision.approved

        self.audit_log.record({
            "event": "action_approved" if decision.approved else "action_denied",
            "action": action.to_dict(),
            "reason": decision.reason,
        })

        if not decision.approved:
            raise ActionDenied(action, reason=decision.reason)
        return True
