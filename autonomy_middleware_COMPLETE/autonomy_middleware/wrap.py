"""
Decorator for wrapping any "tool" function (the kind an LLM agent calls)
so it goes through the consent gate before it actually runs.

This is the model-agnostic seam: it doesn't matter whether the call to
`write_file(...)` was decided on by GPT-4, Claude, a local model, or a
rule-based script — if the function is wrapped, the gate sees it first.
"""

from __future__ import annotations

import functools
from typing import Callable

from .actions import ActionRequest, ActionType
from .gate import ConsentGate


def require_consent(
    gate: ConsentGate,
    action_type: ActionType,
    describe: Callable[..., str],
    payload_fn: Callable[..., dict] | None = None,
):
    """
    Wrap a tool function so it requires gate approval before executing.

    Args:
        gate: the ConsentGate instance to check against.
        action_type: what kind of action this function performs.
        describe: fn(*args, **kwargs) -> str, human-readable summary for the approver.
        payload_fn: optional fn(*args, **kwargs) -> dict, structured detail for the log.
    """

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            action = ActionRequest(
                action_type=action_type,
                description=describe(*args, **kwargs),
                payload=payload_fn(*args, **kwargs) if payload_fn else {},
            )
            gate.request(action)  # raises ActionDenied if rejected
            return fn(*args, **kwargs)

        return wrapper

    return decorator
