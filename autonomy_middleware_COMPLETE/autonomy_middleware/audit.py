"""
Append-only, hash-chained audit log.

Every entry includes the hash of the previous entry, so any edit or deletion
of a past line breaks the chain and is detectable by verify_chain(). This is
a real, standard tamper-evidence technique (same idea as a blockchain's
"prev hash" field, minus the distributed-consensus part, which you don't
need for a local audit trail).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

GENESIS_HASH = "0" * 64


def _hash_entry(prev_hash: str, index: int, timestamp: float, payload: dict) -> str:
    body = json.dumps(
        {"prev_hash": prev_hash, "index": index, "timestamp": timestamp, "payload": payload},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    entry_count: int
    first_bad_index: int | None = None
    reason: str | None = None
    signatures_checked: bool = False  # True only if a signer was supplied to verify_chain()


class AuditLog:
    """
    Append-only JSONL audit log with hash chaining, and optional HMAC
    signing via a separate signing service (see key_signer_service.py).

    Without a signer: this is the original hash-chain-only behavior --
    real tamper-EVIDENCE (edits are detectable), but not tamper-PREVENTION
    against someone who can rewrite the whole file, since the hash chain
    uses no secret and anyone can recompute it over edited content.

    With a signer (a SigningClient instance, or anything with a matching
    .sign(str) -> str method): every entry also gets an HMAC signature
    from the separate signing service, which holds a key this process
    never has access to. That closes the gap above -- forging a valid
    signature over edited content requires the key, which only the
    separate service holds. See key_signer_service.py's module docstring
    for the full explanation of why the separation matters.

    Usage:
        from key_signer_service import SigningClient
        signer = SigningClient(port=8765)  # signing service must be running
        log = AuditLog("audit.jsonl", signer=signer)
        log.record({"event": "action_approved", "action_id": "abc123"})
        result = log.verify_chain(signer=signer)

    HONEST NOTE ON FAILURE BEHAVIOR: if a signer is configured but the
    signing service isn't reachable (not running, wrong port), record()
    will raise rather than silently falling back to writing an unsigned
    entry. An audit log that's SOMETIMES signed and sometimes silently
    isn't is worse than one that's consistently either -- a silent
    fallback would let a real gap hide behind entries that look normal.
    If signing is failing, that's something the operator needs to see
    immediately, not something the ledger should paper over.
    """

    def __init__(self, path: str | Path, signer=None):
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self.signer = signer

    def _last_hash(self) -> tuple[str, int]:
        prev_hash = GENESIS_HASH
        index = -1
        if self.path.stat().st_size == 0:
            return prev_hash, index
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prev_hash = entry["hash"]
                index = entry["index"]
        return prev_hash, index

    def record(self, payload: dict) -> dict:
        """Append a new entry to the chain and return it."""
        prev_hash, last_index = self._last_hash()
        index = last_index + 1
        timestamp = time.time()
        entry_hash = _hash_entry(prev_hash, index, timestamp, payload)

        # Signing the hash (not the raw payload) is enough -- the hash
        # already commits to prev_hash + index + timestamp + payload, so a
        # signature over the hash transitively covers all of it. Signing
        # both would be redundant, not more secure.
        signature = self.signer.sign(entry_hash) if self.signer else None

        entry = {
            "index": index,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
            "payload": payload,
            "hash": entry_hash,
            "signature": signature,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def verify_chain(self, signer=None, allow_unsigned: bool = False) -> VerifyResult:
        """
        Recompute every hash and confirm the chain is unbroken and unedited.

        A signer is REQUIRED by default. Calling this without one raises
        ValueError rather than quietly performing the weaker hash-only
        check. That's deliberate: the hash-only check can be defeated by
        anyone who edits an entry and recomputes the chain over their edit
        (demonstrated in this module's test suite), so silently falling
        back to it would let someone believe they'd verified the ledger
        when they'd actually run the check that doesn't hold up. If the
        signing service is down, that's something to fix before trusting
        the result, not something to route around.

        allow_unsigned=True explicitly opts into the weaker check, for the
        legitimate case of a log that was written without a signer in the
        first place. It has to be typed out on purpose -- it can't happen
        by accident or by forgetting an argument.

        The `signer` passed here can come from a completely separate
        verifier script or process -- that's the point. It doesn't have
        the key either; it just asks the same signing service the same
        question the original writer did.
        """
        if signer is None and not allow_unsigned:
            raise ValueError(
                "verify_chain() requires a signer. Without one, only the hash chain is "
                "checked, which does NOT detect an attacker who edits an entry and "
                "recomputes the chain over it. Pass signer=<SigningClient>, or pass "
                "allow_unsigned=True to explicitly accept the weaker check (only "
                "appropriate for logs written without a signer)."
            )

        entries = self.read_all()
        prev_hash = GENESIS_HASH
        for entry in entries:
            expected = _hash_entry(prev_hash, entry["index"], entry["timestamp"], entry["payload"])
            if entry["hash"] != expected:
                return VerifyResult(
                    ok=False,
                    entry_count=len(entries),
                    first_bad_index=entry["index"],
                    reason="hash mismatch — entry was edited or chain was tampered with",
                    signatures_checked=signer is not None,
                )
            if entry["prev_hash"] != prev_hash:
                return VerifyResult(
                    ok=False,
                    entry_count=len(entries),
                    first_bad_index=entry["index"],
                    reason="prev_hash mismatch — an entry was deleted or reordered",
                    signatures_checked=signer is not None,
                )
            if signer is not None:
                stored_signature = entry.get("signature")
                if stored_signature is None:
                    return VerifyResult(
                        ok=False,
                        entry_count=len(entries),
                        first_bad_index=entry["index"],
                        reason="entry has no signature, but a signer was supplied for verification",
                        signatures_checked=True,
                    )
                expected_signature = signer.sign(entry["hash"])
                if stored_signature != expected_signature:
                    return VerifyResult(
                        ok=False,
                        entry_count=len(entries),
                        first_bad_index=entry["index"],
                        reason="signature mismatch — entry's hash was forged without the real key",
                        signatures_checked=True,
                    )
            prev_hash = entry["hash"]
        return VerifyResult(ok=True, entry_count=len(entries), signatures_checked=signer is not None)
