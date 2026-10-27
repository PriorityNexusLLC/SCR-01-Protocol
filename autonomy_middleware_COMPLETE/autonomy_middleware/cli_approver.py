"""
Default human approver: asks on stdin/stdout.

Swap this out for a Slack bot, web hook, mobile push notification, etc. in
a real deployment — anything that eventually returns a Decision is valid.
"""

from __future__ import annotations

from .actions import ActionRequest
from .gate import Decision


def prompt_approver(action: ActionRequest) -> Decision:
    print("\n" + "=" * 60)
    print(f"ACTION PENDING APPROVAL  [{action.resolved_risk().value.upper()} risk]")
    print(f"  type:        {action.action_type.value}")
    print(f"  description: {action.description}")
    if action.payload:
        for k, v in action.payload.items():
            print(f"  {k}: {v}")
    print("=" * 60)

    answer = input("Approve? [y]es / [n]o / [a]lways this type / [d]eny always: ").strip().lower()

    if answer in ("y", "yes"):
        return Decision(approved=True, reason="approved by user")
    if answer in ("a", "always"):
        return Decision(approved=True, reason="approved by user (session rule: always allow)", remember=True)
    if answer in ("d", "deny always"):
        return Decision(approved=False, reason="denied by user (session rule: always deny)", remember=True)
    return Decision(approved=False, reason="denied by user")


def auto_deny_approver(action: ActionRequest) -> Decision:
    """Useful for tests / CI: denies everything that reaches the approver."""
    return Decision(approved=False, reason="auto-deny (non-interactive mode)")


def auto_approve_approver(action: ActionRequest) -> Decision:
    """Useful for tests / trusted sandboxes only — do not use in production."""
    return Decision(approved=True, reason="auto-approve (non-interactive mode)")
