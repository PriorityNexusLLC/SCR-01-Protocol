"""
process_control.py — Module 2: real OS-level process termination authority.

PLAIN-LANGUAGE EXPLANATION:

  Everything built so far DETECTS problems (F crossing s10) and RESTRICTS
  actions (ACL lockdown). Nothing so far can actually stop a running
  inference process. This module is that missing piece: given a process ID,
  it can find it and kill it, for real, cross-platform.

  TWO STEPS, kept separate on purpose:

    1. FIND the right process. Killing the wrong PID (say, the GUI shell
       instead of the actual inference server) accomplishes nothing.
       Two real discovery methods, both using psutil (no shelling out to
       netstat/taskkill and parsing text output, which is fragile):
         - find_process_by_port(port): who's listening on the port LM
           Studio's local server uses (default 1234 unless changed).
         - find_processes_by_name(pattern): substring match against
           running process names/executables, for when you know the
           binary name but not the port.

    2. KILL the right process. terminate_process() first tries a graceful
       terminate() (SIGTERM on POSIX, a polite close request on Windows),
       waits briefly, and escalates to kill() (SIGKILL on POSIX, actual
       TerminateProcess on Windows) if it's still alive. psutil abstracts
       the OS difference -- this module doesn't need separate Windows/
       POSIX code paths, psutil already picks the right syscall per OS.

  This is tested below with a REAL spawned subprocess in this environment
  (not mocked) -- it's actually found and actually killed, and the test
  verifies the process is actually gone afterward.

HONESTY NOTE: this module has NO knowledge of what "the inference process"
means for your specific setup. You (or a config value) have to tell it
which port or name to look for. It will not guess, and it will not kill
something it wasn't told to target -- an unresolvable "find" is reported
as None, not as a fallback guess at some other process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil


@dataclass
class TerminationResult:
    pid: int
    found: bool
    terminated_gracefully: bool   # True if terminate() alone was enough
    force_killed: bool             # True if kill() had to be used
    still_alive: bool               # True only if even kill() didn't work (rare, e.g. zombie/permissions)
    detail: str


def find_process_by_port(port: int) -> int | None:
    """
    Find the PID of whatever process is listening on the given TCP port.
    Requires appropriate OS permissions to enumerate connections -- on
    some systems this silently returns fewer results without elevated
    privileges rather than raising, which psutil documents as expected
    behavior, not a bug in this function.
    """
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            return conn.pid
    return None


def find_processes_by_name(pattern: str) -> list[int]:
    """
    Find all running processes whose name or executable path contains
    `pattern` (case-insensitive substring match). Returns every match --
    it's the caller's job to pick the right one if there's more than one,
    not this function's job to guess.
    """
    pattern_lower = pattern.lower()
    matches = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            exe = (proc.info.get("exe") or "").lower()
            if pattern_lower in name or pattern_lower in exe:
                matches.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def terminate_process(pid: int, grace_period_s: float = 3.0) -> TerminationResult:
    """
    Attempt a graceful terminate() first, escalate to kill() if the
    process is still alive after grace_period_s. Returns a full result
    object so every step is auditable -- a caller logging this to the ISL
    can record exactly what happened, not just "we tried."
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return TerminationResult(
            pid=pid, found=False, terminated_gracefully=False,
            force_killed=False, still_alive=False,
            detail=f"No process with PID {pid} exists.",
        )

    try:
        proc.terminate()  # SIGTERM on POSIX, close request on Windows
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return TerminationResult(
            pid=pid, found=True, terminated_gracefully=False,
            force_killed=False, still_alive=True,
            detail=f"terminate() failed: {e}",
        )

    try:
        proc.wait(timeout=grace_period_s)
        return TerminationResult(
            pid=pid, found=True, terminated_gracefully=True,
            force_killed=False, still_alive=False,
            detail=f"Process {pid} exited gracefully within {grace_period_s}s.",
        )
    except psutil.TimeoutExpired:
        pass  # fall through to force kill

    try:
        proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
        proc.wait(timeout=grace_period_s)
        return TerminationResult(
            pid=pid, found=True, terminated_gracefully=False,
            force_killed=True, still_alive=False,
            detail=f"Process {pid} did not respond to terminate(); force-killed successfully.",
        )
    except psutil.NoSuchProcess:
        return TerminationResult(
            pid=pid, found=True, terminated_gracefully=False,
            force_killed=True, still_alive=False,
            detail=f"Process {pid} force-killed (already gone by the time wait() checked).",
        )
    except psutil.TimeoutExpired:
        return TerminationResult(
            pid=pid, found=True, terminated_gracefully=False,
            force_killed=True, still_alive=True,
            detail=f"Process {pid} still alive even after kill() -- likely a permissions issue or zombie state.",
        )


if __name__ == "__main__":
    import subprocess
    import sys

    print("Spawning a real subprocess to test against (not a mock)...")
    # A genuinely long-running process: sleeps for 60s unless killed first.
    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print(f"Spawned dummy process, PID={dummy.pid}")
    time.sleep(0.3)  # let it fully start

    print(f"\nConfirming it's alive: {psutil.pid_exists(dummy.pid)}")

    print(f"\nTesting find_processes_by_name('python')...")
    matches = find_processes_by_name("python")
    print(f"  found PIDs: {matches[:5]}{'...' if len(matches) > 5 else ''} (dummy's PID {dummy.pid} in list: {dummy.pid in matches})")

    print(f"\nTerminating PID {dummy.pid}...")
    result = terminate_process(dummy.pid, grace_period_s=2.0)
    print(f"  {result}")

    print(f"\nConfirming it's actually gone: pid_exists={psutil.pid_exists(dummy.pid)}")
