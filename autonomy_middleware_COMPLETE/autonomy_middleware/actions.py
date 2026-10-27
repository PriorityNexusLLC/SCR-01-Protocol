"""
Action definitions and risk classification.

This module has no opinion about which LLM produced an action request —
it only cares about the *effect* an action would have on the system
(filesystem, shell, network), which is what actually needs governing.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL_EXEC = "shell_exec"
    NETWORK_CALL = "network_call"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"        # read-only, no side effects
    MEDIUM = "medium"   # reversible side effects (writes, most network calls)
    HIGH = "high"        # hard-to-reverse or destructive (deletes, shell exec, payments)


# Default risk classification per action type. Callers can override per-action
# via ActionRequest(risk=...) if they know more about a specific call site.
DEFAULT_RISK: dict[ActionType, RiskLevel] = {
    ActionType.FILE_READ: RiskLevel.LOW,
    ActionType.FILE_WRITE: RiskLevel.MEDIUM,
    ActionType.FILE_DELETE: RiskLevel.HIGH,
    ActionType.SHELL_EXEC: RiskLevel.HIGH,
    ActionType.NETWORK_CALL: RiskLevel.MEDIUM,
    ActionType.OTHER: RiskLevel.MEDIUM,
}


@dataclass
class ActionRequest:
    """A single proposed action, about to be executed on behalf of an agent."""

    action_type: ActionType
    description: str                      # human-readable summary, e.g. "write config.json (312 bytes)"
    payload: dict = field(default_factory=dict)   # structured detail: path, command, url, args, etc.
    risk: RiskLevel | None = None         # override; falls back to DEFAULT_RISK[action_type]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def resolved_risk(self) -> RiskLevel:
        return self.risk or DEFAULT_RISK[self.action_type]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "description": self.description,
            "payload": self.payload,
            "risk": self.resolved_risk().value,
            "created_at": self.created_at,
        }
