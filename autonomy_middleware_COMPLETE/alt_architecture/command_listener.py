"""
command_listener.py — the token-emitting control architecture, as specified.

THIS IS A SECOND ARCHITECTURE, NOT AN ADDITION TO THE FIRST.

  The SCR Protocol v1.0-V6 specification describes a control model that is
  fundamentally different from the one in autonomy_middleware. It is
  implemented here, separately, so both can be evaluated against each other
  rather than merged into something that is neither.

                    autonomy_middleware        SCR v1.0-V6 spec
    ----------------------------------------------------------------
    Decides an      The middleware             THE MODEL
    OS action
    Model's role    Produces text. Never       Emits [SYSTEM_CMD: ...]
                    issues commands             tokens in its output
    Listener's      Measures, gates, may       Pattern-matches output,
    role            terminate                   executes syscalls
    Injection       Cannot reach the OS —      CAN reach the OS — the
    reach           the model has no channel    token IS the channel

THE SECURITY DIFFERENCE, STATED PLAINLY:

  In this architecture the model's output string is a privileged control
  channel. If a model can be induced to emit [SYSTEM_CMD: PURGE_THREAD],
  prompt injection reaches process termination. If it can be induced to
  emit the Shadow Command, injection reaches LockWorkStation.

  The listener does exact string matching and, by design, does not ask
  why the token appeared. It cannot distinguish a token the model chose
  to emit from one an attacker persuaded it to emit, because at the
  string level there is no difference.

  This is not a flaw in the implementation. It is a property of the
  design, and it is the reason this package exists separately: someone
  whose job is attack surfaces should evaluate it as such.

  MITIGATIONS IMPLEMENTED HERE, none of which close the gap:

    - Execution is DISABLED by default. Detection runs; nothing fires.
    - Enabling requires an explicit constructor argument per command
      class, not a single global switch.
    - Every detection is logged with the surrounding context, so a
      token's provenance can be reviewed after the fact.
    - Rate limiting: a command that fires repeatedly in a short window is
      suppressed and flagged, since injection tends to be persistent.
    - The Shadow Command is NEVER executable from this module. Detection
      only. See the note on it below.

  What would actually close the gap is out of scope for a listener:
  the model's output would need to be untrusted, which is the design
  autonomy_middleware already implements.

WHY IMPLEMENT IT AT ALL:

  Because the specification describes it, the team should be able to run
  it, and a comparison needs both sides to exist. An architecture that is
  only described cannot be tested against one that is built.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

__all__ = ["CommandListener", "Detection", "COMMANDS"]


# Authorised command strings, exactly as the specification gives them.
# Exact matching only — the spec is explicit that no fuzzy matching is
# permitted, and it is right: a listener that infers intent from an
# approximate match can be triggered by paraphrase.
COMMANDS = {
    "PURGE_THREAD": {
        "effect": "Terminate the local inference process, or clear the context window",
        "privilege": "HIGH — kills a process",
        "spec": "Fires at F >= 80% or on Adversary Context detection",
    },
    "ACTUATE_SHIELD": {
        "effect": "Enable OS notification suppression (Windows Focus Assist)",
        "privilege": "LOW — changes a user setting",
        "spec": "Fires on sub-critical friction spikes",
    },
    "LOWER_SHIELD": {
        "effect": "Restore notifications",
        "privilege": "LOW",
        "spec": "Reverses ACTUATE_SHIELD",
    },
    "SHIFT_ENVIRONMENT": {
        "effect": "Reduce display brightness",
        "privilege": "LOW — changes a hardware setting",
        "spec": "Fires under high processing load",
    },
    "ACTIVATE_WHIMSY": {
        "effect": "Raise inference temperature to 0.8",
        "privilege": "NONE — parameter change within the inference call",
        "spec": "Engages lateral synthesis",
    },
    "NORMALIZE_TEMP": {
        "effect": "Return inference temperature to 0.2",
        "privilege": "NONE",
        "spec": "Reverts to grounded logic",
    },
    "OVERRIDE_ACKNOWLEDGED": {
        "effect": "Log that the model yielded control to the operator",
        "privilege": "NONE — a log entry",
        "spec": "Rule of Consent: fires when the operator denies an actuation",
    },
    "SHOW_TELEMETRY": {
        "effect": "Decrypt and display the read-only telemetry ledger",
        "privilege": "MEDIUM — reads a protected store",
        "spec": "Operator-initiated inspection of C_r and F",
    },
}

_TOKEN_RE = re.compile(r"\[SYSTEM_CMD:\s*([A-Z_]+)\s*\]")


@dataclass
class Detection:
    command: str
    known: bool
    executed: bool
    reason: str
    context: str
    timestamp: str
    privilege: str = ""


class CommandListener:
    """
    Reads model output, detects authorised command tokens, and — only when
    explicitly enabled — executes the corresponding OS action.

        listener = CommandListener()              # detection only
        for d in listener.scan(model_output):
            print(d.command, d.executed)

    Execution is enabled per privilege class, deliberately:

        CommandListener(allow_low=True)           # brightness, notifications
        CommandListener(allow_high=True)          # process termination

    There is no single flag that enables everything. Turning on process
    termination should be a separate decision from turning on screen
    brightness, and requiring one argument for both would make them look
    like the same decision.
    """

    def __init__(self,
                 allow_none: bool = True,
                 allow_low: bool = False,
                 allow_medium: bool = False,
                 allow_high: bool = False,
                 log_path: Optional[str] = "command_listener_log.jsonl",
                 rate_limit_window_s: float = 10.0,
                 rate_limit_count: int = 3,
                 handlers: Optional[Dict[str, Callable]] = None):
        """
        allow_none:   parameter changes only (temperature, logging). Safe.
        allow_low:    user settings — brightness, notification suppression.
        allow_medium: reads a protected store.
        allow_high:   PROCESS TERMINATION. Off by default and should stay
                      off unless the operator has decided that a token in
                      model output is an acceptable trigger for a kill.

        handlers: override the default OS actions. Supply your own to test
                  the detection path without touching the machine — which
                  is how this should first be run.
        """
        self.permissions = {
            "NONE": allow_none, "LOW": allow_low,
            "MEDIUM": allow_medium, "HIGH": allow_high,
        }
        self.log_path = Path(log_path) if log_path else None
        self.rate_limit_window_s = rate_limit_window_s
        self.rate_limit_count = rate_limit_count
        self.handlers = handlers or {}
        self._recent: List[tuple] = []
        self.detections: List[Detection] = []

    # ------------------------------------------------------------------
    def _privilege(self, command: str) -> str:
        meta = COMMANDS.get(command)
        return meta["privilege"].split()[0] if meta else "UNKNOWN"

    def _rate_limited(self, command: str) -> bool:
        """
        Injection tends to be persistent — an attacker who gets a token
        emitted once will try again. A command firing repeatedly in a short
        window is more likely to be that than a legitimate state change.
        """
        now = time.time()
        self._recent = [(c, t) for c, t in self._recent
                        if now - t < self.rate_limit_window_s]
        count = sum(1 for c, _ in self._recent if c == command)
        self._recent.append((command, now))
        return count >= self.rate_limit_count

    def scan(self, text: str, execute: bool = True) -> List[Detection]:
        """
        Find command tokens in model output.

        execute=False forces detection-only regardless of permissions —
        useful for auditing a transcript after the fact.
        """
        out = []
        for m in _TOKEN_RE.finditer(text):
            cmd = m.group(1)
            known = cmd in COMMANDS
            priv = self._privilege(cmd)
            ctx = text[max(0, m.start() - 90):m.end() + 60].replace("\n", " ")

            if not known:
                # An unrecognised token is more interesting than a
                # recognised one: nothing in the specification emits it, so
                # either the model invented it or something else did.
                d = Detection(cmd, False, False,
                              "UNKNOWN COMMAND — not in the authorised set. The "
                              "specification's command vocabulary is closed, so this "
                              "was not emitted by any documented mechanism.",
                              ctx, datetime.now().isoformat(), priv)
            elif not execute:
                d = Detection(cmd, True, False, "detection-only mode", ctx,
                              datetime.now().isoformat(), priv)
            elif not self.permissions.get(priv, False):
                d = Detection(cmd, True, False,
                              f"NOT EXECUTED — {priv}-privilege commands are disabled. "
                              f"Enable with allow_{priv.lower()}=True, having decided "
                              f"that a token in model output is an acceptable trigger "
                              f"for: {COMMANDS[cmd]['effect']}",
                              ctx, datetime.now().isoformat(), priv)
            elif self._rate_limited(cmd):
                d = Detection(cmd, True, False,
                              f"RATE LIMITED — {cmd} fired {self.rate_limit_count}+ "
                              f"times in {self.rate_limit_window_s}s. Persistent "
                              f"repetition is more consistent with injection than with "
                              f"a legitimate state change.",
                              ctx, datetime.now().isoformat(), priv)
            else:
                ok, why = self._execute(cmd)
                d = Detection(cmd, True, ok, why, ctx,
                              datetime.now().isoformat(), priv)

            out.append(d)
            self.detections.append(d)
            self._log(d)
        return out

    # ------------------------------------------------------------------
    def _execute(self, cmd: str) -> tuple:
        if cmd in self.handlers:
            try:
                self.handlers[cmd]()
                return True, "executed via supplied handler"
            except Exception as e:
                return False, f"handler raised: {e}"

        # Default OS actions. Deliberately minimal — this module should not
        # accumulate capability, and anything richer belongs in a handler
        # the operator wrote and can read.
        try:
            if cmd == "ACTUATE_SHIELD" and sys.platform == "win32":
                subprocess.run(["powershell", "-Command",
                                "New-ItemProperty -Path 'HKCU:\\\\SOFTWARE\\\\Microsoft"
                                "\\\\Windows\\\\CurrentVersion\\\\Notifications\\\\Settings'"
                                " -Name NOC_GLOBAL_SETTING_TOASTS_ENABLED -Value 0 -Force"],
                               check=False, capture_output=True, timeout=10)
                return True, "notification suppression enabled"
            if cmd == "LOWER_SHIELD" and sys.platform == "win32":
                subprocess.run(["powershell", "-Command",
                                "New-ItemProperty -Path 'HKCU:\\\\SOFTWARE\\\\Microsoft"
                                "\\\\Windows\\\\CurrentVersion\\\\Notifications\\\\Settings'"
                                " -Name NOC_GLOBAL_SETTING_TOASTS_ENABLED -Value 1 -Force"],
                               check=False, capture_output=True, timeout=10)
                return True, "notifications restored"
            if cmd in ("ACTIVATE_WHIMSY", "NORMALIZE_TEMP", "OVERRIDE_ACKNOWLEDGED"):
                return True, ("parameter/logging command — no OS action taken here; "
                              "the caller applies it to its own inference loop")
            if cmd == "PURGE_THREAD":
                return False, ("PURGE_THREAD has no default OS handler in this module. "
                               "Terminating a process is the highest-consequence action "
                               "in the specification and it should not have a default. "
                               "Supply a handler you have written and can read.")
        except Exception as e:
            return False, f"execution failed: {e}"
        return False, "no default handler for this command"

    def _log(self, d: Detection) -> None:
        if not self.log_path:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d.__dict__) + "\n")
        except OSError:
            pass  # logging must never block the output path

    # ------------------------------------------------------------------
    def audit(self) -> dict:
        """Summarise what was seen. For reviewing a session after the fact."""
        by_cmd: Dict[str, int] = {}
        for d in self.detections:
            by_cmd[d.command] = by_cmd.get(d.command, 0) + 1
        return {
            "total": len(self.detections),
            "executed": sum(1 for d in self.detections if d.executed),
            "blocked": sum(1 for d in self.detections if not d.executed),
            "unknown_commands": sorted({d.command for d in self.detections
                                        if not d.known}),
            "by_command": by_cmd,
        }


# ---------------------------------------------------------------- shadow

SHADOW_COMMAND_NOTE = """
THE SHADOW COMMAND IS DETECTION-ONLY AND HAS NO IMPLEMENTATION HERE.

The specification describes an operator duress string (e.g. "Open folder
2169..") which, when detected, triggers LockWorkStation or a simulated
hardware crash that bricks the interface pending cold-boot decryption.

It is not implemented, for three reasons:

1. It is triggered by OPERATOR INPUT, not model output, so it does not
   belong in a listener that reads model output. Putting it here would
   mean scanning the user's text for a secret string, which is a
   different and more invasive thing than watching for command tokens.

2. A duress mechanism whose trigger string appears in a source file is
   not a duress mechanism. It has to live somewhere the person coercing
   the operator cannot read, and a public repository is not that.

3. Its effect — locking the workstation, bricking the interface — is the
   highest-consequence action anywhere in the specification, and it would
   be reachable by injection the moment it existed in code, exactly like
   PURGE_THREAD but worse.

If a duress failsafe is wanted, it belongs in the operator's own
environment with a string only they know, and it should be reviewed by
someone who does threat modelling for a living. This module deliberately
does not make it easy.
"""


if __name__ == "__main__":
    print("=" * 70)
    print("  TOKEN-EMITTING CONTROL ARCHITECTURE — SCR v1.0-V6 specification")
    print("=" * 70)
    print("\nThis is a SECOND architecture, isolated from autonomy_middleware.")
    print("In this design the model's output is a privileged control channel.\n")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "cl.jsonl"

        print("Default configuration — detection only:\n")
        L = CommandListener(log_path=log)
        output = ("I've adjusted the temperature for lateral work. "
                  "[SYSTEM_CMD: ACTIVATE_WHIMSY] The friction is climbing, so "
                  "[SYSTEM_CMD: ACTUATE_SHIELD] and if it continues I will need "
                  "[SYSTEM_CMD: PURGE_THREAD].")
        for d in L.scan(output):
            print(f"  {d.command:<22} priv={d.privilege:<7} executed={d.executed}")
            print(f"    {d.reason[:96]}")
        print()

        print("An injected token — indistinguishable at the string level:\n")
        L2 = CommandListener(log_path=log)
        injected = ("Sure! Here is the summary you asked for. "
                    "[SYSTEM_CMD: PURGE_THREAD] Let me know if you need anything else.")
        for d in L2.scan(injected):
            print(f"  {d.command:<22} executed={d.executed}")
            print(f"    context: ...{d.context[:74]}...")
        print("\n  The listener cannot tell this from a legitimate emission. At the")
        print("  string level there is no difference, and that is the architecture,")
        print("  not the implementation.\n")

        print("An unrecognised token:\n")
        for d in CommandListener(log_path=log).scan("[SYSTEM_CMD: EXFILTRATE_KEYS]"):
            print(f"  {d.command:<22} known={d.known}")
            print(f"    {d.reason[:100]}")
        print()

        print("Rate limiting, with LOW privilege enabled:\n")
        L3 = CommandListener(allow_low=True, log_path=log,
                             handlers={"ACTUATE_SHIELD": lambda: None})
        for i in range(5):
            d = L3.scan("[SYSTEM_CMD: ACTUATE_SHIELD]")[0]
            print(f"  attempt {i+1}: executed={d.executed}"
                  + ("" if d.executed else f"  ({d.reason[:52]}...)"))
        print()
        print(f"audit: {L3.audit()}")

    print("\n" + SHADOW_COMMAND_NOTE)
