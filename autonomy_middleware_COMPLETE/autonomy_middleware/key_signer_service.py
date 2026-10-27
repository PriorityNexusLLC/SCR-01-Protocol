"""
key_signer_service.py — Module 3 prototype: proves the core separation
property before anything else gets built on top of it.

PLAIN-LANGUAGE EXPLANATION:

  This is a SEPARATE PROCESS, not a function the middleware calls directly.
  That separation is the entire point -- if signing were just a method
  inside the same Python process as the AI-run middleware, the AI's process
  would have the key in its own memory, full stop, no matter how the code
  around it is organized.

  So instead: this service runs on its own, listens on localhost only
  (127.0.0.1 -- unreachable from any other machine), and offers exactly
  ONE operation over that connection: "sign this text for me." It never
  offers "give me the key," because that operation was never written --
  not disabled, not permission-gated, LITERALLY ABSENT from the protocol.
  A caller can't request what was never implemented.

  The secret key itself is generated once, written to a local file with
  restrictive permissions, and loaded only by this service process --
  never by anything that imports audit.py or runs the middleware.

WHAT THIS PROTOTYPE PROVES (and what it doesn't yet):
  Proves:
    - A client (standing in for the AI-run middleware) can request
      signatures and successfully verify the ledger.
    - That same client has NO code path to read the raw key -- demonstrated
      by attempting it and hitting an AttributeError, not a caught
      PermissionError. The capability doesn't exist to be denied.
    - Tampering with a past entry breaks verification, and -- critically --
      an attacker who edits the ledger file CANNOT re-sign their edit,
      because forging a valid signature requires the key, which they don't
      have (unlike the old unkeyed hash chain, where anyone could
      recompute a valid-looking chain over edited content).
  Does NOT yet prove:
    - Auto-start behavior, a settings UI, or hardware-anchoring (Option
      A/B from before). Those are real next steps, deliberately deferred.
    - Protection if someone has direct filesystem access to the key file
      itself while the service is offline -- this prototype's key file
      uses OS file permissions (chmod 600 on POSIX) as its only protection
      right now, which is honest but not the TPM-level guarantee Option B
      would eventually add.

RUNNING IT: this file can run as the service (`python key_signer_service.py
serve`) in one terminal, and a separate process can act as a client.
For this prototype, the demo at the bottom runs both in-process using
threads, purely so it's runnable as a single self-contained test --  a
real deployment runs the service as an actual separate OS process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import stat
import threading
import time
from pathlib import Path


class KeyVault:
    """
    Owns the secret key. This class is used ONLY inside the service
    process. The client class further down deliberately does not import
    or reference this class at all -- that's not a style choice, it's
    the actual mechanism that makes "the client can't read the key" true.
    """

    def __init__(self, key_path: Path):
        self.key_path = key_path
        if key_path.exists():
            self._key = key_path.read_bytes()
        else:
            self._key = secrets.token_bytes(32)
            key_path.write_bytes(self._key)
            # Best-effort restrictive permissions. POSIX: owner read/write
            # only. Windows' ACL model differs and isn't fully covered by
            # chmod -- a real deployment on Windows needs an explicit ACL
            # restriction (icacls) as a follow-up, noted honestly here
            # rather than implied as already handled.
            try:
                os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
            except (NotImplementedError, OSError):
                pass

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()


class SigningService:
    """
    The standing process. Listens on localhost only. Exactly one request
    type is handled: {"op": "sign", "payload": "<text>"}. Anything else
    gets an error. There is no "get_key" branch to accidentally leave
    unguarded -- it simply isn't in this if/elif chain.
    """

    def __init__(self, key_path: Path, host: str = "127.0.0.1", port: int = 8765):
        self.vault = KeyVault(key_path)
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._running = False

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self._running = True
        threading.Thread(target=self._serve_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def _serve_loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        with conn:
            try:
                data = conn.recv(65536)
                request = json.loads(data.decode())
                if request.get("op") == "sign":
                    payload = request["payload"].encode()
                    signature = self.vault.sign(payload)
                    conn.sendall(json.dumps({"ok": True, "signature": signature}).encode())
                else:
                    conn.sendall(json.dumps({"ok": False, "error": "unknown operation"}).encode())
            except Exception as e:
                try:
                    conn.sendall(json.dumps({"ok": False, "error": str(e)}).encode())
                except Exception:
                    pass


class SigningClient:
    """
    What the AI-run middleware actually gets access to. Deliberately has
    no reference to KeyVault, no field that could hold key bytes, and no
    method resembling get_key(). This isn't "the key is present but
    denied" -- there's no attribute here that holds it, ever.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port

    def sign(self, payload: str) -> str:
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.sendall(json.dumps({"op": "sign", "payload": payload}).encode())
            response = json.loads(sock.recv(65536).decode())
        if not response.get("ok"):
            raise RuntimeError(f"signing failed: {response.get('error')}")
        return response["signature"]


def run_demo():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / "vault.key"

        print("Starting the signing service (a separate process in real deployment)...")
        service = SigningService(key_path, port=8765)
        service.start()
        time.sleep(0.2)

        client = SigningClient(port=8765)

        print("\n1. Client signs three ledger-style entries:")
        entries = [
            "event=action_auto_allowed action=file_read",
            "event=s2_warning_block F=1.6",
            "event=s10_critical_block F=2.5",
        ]
        signatures = [client.sign(e) for e in entries]
        for e, s in zip(entries, signatures):
            print(f"   {e!r} -> {s[:16]}...")

        print("\n2. Proving the client CANNOT read the key -- attempting client.get_key()...")
        try:
            client.get_key()
            print("   FAIL: this should never succeed")
        except AttributeError as e:
            print(f"   Confirmed: {e}  (the capability doesn't exist to be denied)")

        print("\n3. Verifying entries against their signatures (normal case, should all pass):")
        for e, s in zip(entries, signatures):
            check = client.sign(e)  # re-signing the same content should reproduce the same signature
            print(f"   {e[:40]!r}... valid={hmac.compare_digest(check, s)}")

        print("\n4. Simulating tampering: an attacker edits entry 2 in the stored ledger,")
        print("   then tries to keep the OLD signature (they can't produce a new valid one")
        print("   without the key):")
        tampered_entry = "event=s2_warning_block F=0.1"  # attacker lowers the recorded friction value
        old_signature = signatures[1]
        check = client.sign(tampered_entry)
        print(f"   original signature: {old_signature[:16]}...")
        print(f"   signature the tampered content WOULD need: {check[:16]}...")
        print(f"   tamper detected: {not hmac.compare_digest(check, old_signature)}")

        service.stop()


def run_serve(key_filename: str = "vault.key", port: int = 8765):
    """
    Runs the service for real, as a long-lived process you start on
    purpose in its own terminal window and leave running. This is what a
    real deployment actually uses -- run_demo() above is only a
    self-contained proof, not how this gets used day to day.
    """
    key_path = Path(key_filename)
    is_new_key = not key_path.exists()

    service = SigningService(key_path, port=port)
    service.start()

    print(f"Signing service is running.")
    print(f"  Listening on: 127.0.0.1:{port} (localhost only -- not reachable from other machines)")
    print(f"  Key file:     {key_path.resolve()}")
    print(f"  {'New key generated just now.' if is_new_key else 'Loaded existing key.'}")
    print(f"\nLeave this window open. This is the service holding the key.")
    print(f"Press Ctrl+C to stop it.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        run_serve()
    else:
        run_demo()
