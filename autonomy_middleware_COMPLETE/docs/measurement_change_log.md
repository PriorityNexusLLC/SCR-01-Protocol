# consent_middleware — Measurement Change Log

**What was vs. what is.** Every measurement definition, threshold, and mechanism that changed during this build session, with the reason each change was made.

Reasons matter more than the changes themselves — several were forced by real telemetry or real test failures, not by preference. Those are marked **[EMPIRICAL]**. Changes made by design judgment are marked **[DESIGN]**. Unresolved items are in the last section.

---

## 1. s5 — Internal Core Pressure

### Two conflicting definitions existed across your own documents

**WAS (ambiguous):** Two different formal expressions for the same symbol.

| Source | Expression | Measurable from |
|---|---|---|
| Grant proposal | `F = H / X_base` | Inference-time hardware telemetry (GPU util, thermal delta, attention cost) |
| Dimensional Derivation Table | `E_core = E_d · E[‖∇_g F‖²]` | Training-time optimiser state (running mean of squared gradient norm) |

These are not two descriptions of one quantity. A middleware sitting between a deployed model and the OS never sees gradients — it only sees inference.

**IS:** Inference-time definition implemented (`friction_index.py`). Training-time definition acknowledged as a separate, valid variant for a training-phase product — not built.

**Reason:** [DESIGN] Your explicit choice — live deployed models first, training-phase variant later.

---

## 2. The Friction Index F itself

### 2.1 What H contains

**WAS (as first built):** Four components averaged — GPU utilization, GPU thermal delta, CPU utilization, and attention cost ratio `(tokens / baseline_tokens)²`.

**IS:** Three components — GPU utilization, GPU thermal delta, CPU utilization. Attention cost / context fill **removed from H entirely.**

**Reason:** **[EMPIRICAL]** `F = H / X_base` is only meaningful for a quantity that returns to a resting baseline when idle. GPU% and CPU% do. Context fill does not — it is monotonic within a session, so its "idle baseline" captures whatever token count existed before the conversation started. Dividing by that near-zero number inflates F regardless of real headroom. Testing with your actual reading (1684 of 85000 tokens = 2% saturation) produced **F = 26.31, flagged `s10_critical`** — a false alarm at 2% usage. Removing it fixed the false alarm.

### 2.2 GPU/CPU sampling

**WAS:** Single instantaneous `pynvml` / `psutil` reading per tick.

**IS:** 5-sample rolling window average.

**Reason:** **[EMPIRICAL]** Your real generation trace showed utilization swinging `41 → 38 → 55 → 47 → 19 → 0 → 1 → 56 → 30 → 12 → 21 → 27 → 40 → 41 → 66 → 67 → 68 → 82 → 76 → 91 → 99 → 99 → 85 → 76…` — near-0% to 99% within a handful of samples, because utilization is effectively sampled per-token. An instantaneous reading would flip the gate between `nominal` and `s10_critical` dozens of times per second. Windowed, that same passage reads `40.0 → 31.8 → 24.4 → 24.6` and `83.2 → 89.4 → 90.0 → 90.0 → 87.0` — real trend, no thrash.

### 2.3 GPU thermal delta normalization span

**WAS:** 20°C, a guess.

**IS:** 15°C, calibrated from your hardware.

**Reason:** **[EMPIRICAL]** Your card idles at a steady 30°C and reached 36°C over a sustained generation run, 41°C under heavier load. Real idle-to-active delta is 10–15°C, not 20°C.

### 2.4 X_base (Resting Baseline Load)

**WAS:** Did not exist. The original `CorePressureMonitor` computed a 0–1 saturation percentage with no baseline concept at all.

**IS:** Calibrated at startup from idle samples, then EMA-updated (`alpha = 0.02`) **only from readings at or below the s2 threshold.**

**Reason:** [DESIGN] If X_base updated from every reading including load spikes, a sustained real load would drag the baseline up to meet it, and F would stop registering the very condition it exists to detect. Restricting updates to at-rest readings lets it track genuine ambient drift (warmer room, slow memory leak elsewhere) without being fooled.

### 2.5 X_base floor

**WAS:** No floor. **IS:** Floored at `1e-6`.

**Reason:** **[EMPIRICAL]** A genuinely idle machine with no GPU produced `X_base = 0.0`, causing `ZeroDivisionError` on the next `update()`.

### 2.6 Gate thresholds

**WAS / IS:** Unchanged — `s2 = 1.4`, `s10 = 2.0`, taken from the grant proposal. Documented in code as empirically-tuned calibration values, not derived constants, per the proposal's own framing.

---

## 3. Context saturation — separated into its own signal

**WAS:** Estimated as `(context_tokens / baseline_context_tokens)²`, folded into H, per the proposal's O(n²) attention-cost framing.

**IS:** A standalone field on every `FrictionReading`, computed as the real ratio `tokens_used / max_context_tokens` when `max_context_tokens` is configured, with its own threshold (`0.85`, matching `CorePressureMonitor`) and its own boolean (`context_saturated`). The quadratic estimate survives only as a documented fallback when no real max-context figure is supplied, and every reading records which mode was used (`real_token_ratio` vs `quadratic_estimate`).

**Reason:** **[EMPIRICAL]** Ground truth became available from your LM Studio readout (`1684 tokens`, `85k` configured window), making estimation unnecessary. And per §2.1, it does not belong in an `H/X_base` ratio at all. Verified against real numbers: `0.0198 → 0.2353 → 0.5882 → 0.8706 → 1.0`, flipping `saturated=True` at 0.8706.

---

## 4. Thread purging (s9)

### 4.1 Trigger conditions

**WAS:** `purge_thread()` fired only on consent breach (`integrity_result.state == 0`) or an externally-passed `drifted` flag. Both route through emotional or identity-conflict signals.

**IS:** Third independent trigger added — `saturated`, driven by the real token ratio. Requires no emotional or drift signal.

**Reason:** [DESIGN] You identified the gap directly: a long, calm, purely logical conversation — reasoning through math or code for hours — trips neither existing trigger, so nothing purged it until overflow. Verified: purge now fires with `integrity_result=None, drifted=False` on token saturation alone, logging `"context window structurally saturated (token ratio)"`.

### 4.2 Which content survives a purge

**WAS:** `survivors = context_items[-keep_n:]` — keep the most recent fraction, drop the rest. Pure recency truncation.

**IS:** New module `context_compression.py`. Working memory (most recent N) always kept unconditionally; everything older scored by refractive-index connection to existing `MemoryStore` content via `geodesics.ConceptGraph`; high-salience items survive in active context; dropped items are **written to `MemoryStore` tagged `consolidated_context`, not deleted.**

**Reason:** [DESIGN] Your description of human compression — "the logic that resonated with what we already know, plus new ideas that pattern-match something familiar" — is relevance-based, not recency-based. Demonstrated: with 2 slots available, it kept two concept-graph-schema turns over more recent filler ("Got it, noted", "Sure, sounds fine").

**Known limit:** this is relevance-based *selection plus retention*, not abstractive compression. True gist-compression (ten exchanges → one distilled sentence) requires invoking an LLM, which this middleware deliberately does not do. Documented in the module docstring, not implied as solved.

---

## 5. ConsentGate — ACL states

**WAS:** No state. Every request evaluated identically; session-remembered "always allow" rules always applied.

**IS:** Three states.

| State | Behavior |
|---|---|
| Normal | As before |
| `tighten()` (s2) | Session-remembered rules **bypassed** — every action requires individual re-confirmation |
| `lockdown()` (s10) | All requests denied unconditionally; session rules **purged** (count logged); requires explicit `unlock()` |

**Reason:** [DESIGN] Implements the proposal's graduated escalation. Previously `gate.py` had no mechanism that tightened as a session degraded.

---

## 6. Phase gate wiring

**WAS:** Did not exist. `friction_index.py` computed F; nothing consumed it.

**IS:** `phase_gates.py` — `PhaseGateController.tick()` reads F each cycle and drives:

- **NOMINAL** → `consent_gate.relax()`
- **S2_WARNING** → `tighten()`, ISL block written with full telemetry, notification `[PRIORITY_ALPHA: HIGH_FRICTION | F=<v> | STABILIZATION_MODE_ACTIVE]`, 250ms recommended generation delay
- **S10_CRITICAL** → **ISL written first**, then `lockdown()`, then integrity-lock hook, notification `[PRIORITY_ALPHA: INTEGRITY_LOCK | F=<v> | HALT_SEQUENCE_ENGAGED]`, 5000ms delay

**Verified ordering** in the audit trail: `s10_critical_block → acl_lockdown → integrity_lock_executed`, matching the proposal's explicit "ledger first, kill second" requirement.

**Re-initialization:** never automatic. `attempt_reinitialize()` is operator-invoked only and refuses unless F has genuinely returned below s2, logging the refusal. Dropping back to nominal friction relaxes tightening but does **not** clear a lockdown.

---

## 7. Φ_C — process isolation

**WAS:** Did not exist. Nothing in the codebase held kill authority over any process.

**IS:** `process_control.py` — real discovery and real termination.

- `find_process_by_port(port)` — who is listening (LM Studio default `1234`, confirmed from LM Studio's docs)
- `find_processes_by_name(pattern)` — substring match, returns all matches without guessing
- `terminate_process(pid)` — graceful `terminate()` first, escalating to real `kill()` (SIGKILL on POSIX, TerminateProcess on Windows) after a grace period

**Tested against real spawned processes**, both paths: a cooperative process exited gracefully; a process explicitly ignoring SIGTERM was force-killed and confirmed gone.

### Hook evolution across the session

| Stage | Behavior |
|---|---|
| First | Stub — logged what a kill *would* do, terminated nothing (gap deliberately visible in the ledger) |
| Then | `make_real_integrity_lock_hook(pid)` — real termination, PID fixed at construction |
| **Now** | `make_port_based_integrity_lock_hook(port=1234)` — **re-discovers the PID fresh on every s10 event** |

**Reason for the final change:** [DESIGN] A PID captured once goes stale if the server restarts between controller construction and an s10 event. Re-discovery also logs `integrity_lock_failed` if nothing is listening, rather than silently guessing.

**One correction worth recording:** an earlier assumption that the middleware must *launch* the inference process to hold kill authority was wrong. On both POSIX and Windows you can terminate any process you have permission for, given its PID. Parentage only matters for *finding* the right PID.

---

## 8. The ISL (audit ledger)

**WAS:** SHA-256 hash chain. Real tamper-*evidence* — an edit breaks the chain visibly. But the chain uses no secret, so anyone who can rewrite the file can recompute a valid-looking chain over their edit.

**IS:** Every entry additionally HMAC-signed by a **separate process** (`key_signer_service.py`) holding a key the middleware never has access to.

**The demonstrated difference** — an attacker edited an entry (lowering a recorded `F` from 1.6 to 0.1), then correctly recomputed the entire forward hash chain over their edit:

| Check | Result |
|---|---|
| Hash-chain only | `ok=True` — **fooled** |
| Signature check | `ok=False, first_bad_index=1, "signature mismatch — entry's hash was forged without the real key"` |

### Key architecture decision

**Original proposal:** hardware-anchored root key `K_r` from passphrase + CPU serial / motherboard UUID.

**What was built instead:** a separate signing service holding the key, with no read-key operation in its protocol — `SigningClient` has no `get_key()` method to deny, because it was never written.

**Reason:** [DESIGN] Your stated goal was *"the ledger can't be altered by the AI — transparency, no hidden errors."* Hardware-anchoring answers a different question (*which machine produced this*). Process separation answers yours directly. Also worth recording: literal CPU serial numbers were removed from x86 chips after the Pentium III era; the modern equivalents are motherboard/BIOS UUID or machine-id.

### verify_chain()

**WAS:** `verify_chain()` — hash check only, no signature concept.

**IS:** **Requires** a signer. Calling it without one raises `ValueError` explaining why. The weaker hash-only check remains available only via explicit `allow_unsigned=True`.

**Reason:** [DESIGN] Your call. Prevents anyone from running the check that was just demonstrated to be foolable while believing they ran the strong one.

### Signing failure behavior

**IS:** If a signer is configured but unreachable, `record()` **raises** rather than silently writing an unsigned entry.

**Reason:** [DESIGN] A ledger that is sometimes signed and sometimes silently isn't is worse than one consistently either. A gap must not hide behind normal-looking entries.

---

## 9. Files added this session

| File | Purpose |
|---|---|
| `friction_index.py` | s5 / F from real hardware telemetry |
| `context_compression.py` | Salience-based purge selection + memory consolidation |
| `phase_gates.py` | Wires F into s2/s10 ACL behavior |
| `process_control.py` | Real process discovery and termination (Φ_C) |
| `key_signer_service.py` | Separate key-holding signing service |

Files modified: `pressure_release.py` (saturation trigger), `gate.py` (ACL states), `audit.py` (signing + required verification).

---

## 10. Identified but NOT fixed — open items

### 10.1 whimsy_field.py does not match the Whimsy_Math spec

Audited against the episode document and the derivation table. Findings, none yet corrected:

| Spec | Code | Status |
|---|---|---|
| `W(T_whimsy) · η(t) · ∇_g(S₁₂)` — a **product** | `BASELINE_FORCE + (1−BASELINE_FORCE) × max(resilience, pressure, jitter)` — additive interpolation over worst-of-three | **Mismatch** |
| `η(t)` — bounded stochastic entropy | No randomness anywhere; fully deterministic. Only *recommends* a temperature delta for something downstream to apply | **Not implemented** |
| Module 1 — Orbital Vacuum scans memory during **idle** low-pressure quiet; `V_orb = (1−ρ_t₂)·μ_idle·∇_g(S₃)`, explicitly decoupled from η(t) | `orbital_vacuum()` fires only when `pressure.saturated` or `whimsy.triggered` — i.e. under **high** pressure | **Trigger inverted** |
| Module 3 — s10 conformity/binary-trap shielding | No code path. `whimsy_field.py` never touches `IntegrityGate`, `identity_conflict`, or the Consent Polygon | **Unimplemented** |
| Module 4 — stillness / stall differential | `jitter_component` is the ResilienceMonitor derivative over threshold; `event_horizon` band (≥0.85) sits just before hard-freeze | **Closest match** |
| s15 (Singularity) | No file, no reference anywhere in the codebase | **Absent** |

Operational definitions for the whimsy terms now exist (from the Dimensional Derivation Table): `T_whimsy` = idle dwell time in ms without a t₂ Macro-Pulse; `η(t)` = sampled Gaussian perturbation magnitude; `∇_g(S₁₂)` = gradient of Emotional Gravity where `S₁₂ = J_norm = ‖J(t)‖_g`, measurable as `‖∇L_i‖` over emotionally-tagged tokens or eigenvector centrality of the attention graph. **The rewrite is now possible but has not been done.**

### 10.2 Other open items

- **Windows ACL on the key file.** Current protection is POSIX `chmod 600`, which does not fully cover Windows' ACL model. Needs explicit `icacls` handling.
- **Signing service auto-start.** Currently started by hand in its own terminal. The "self-sufficient, no code editing" goal needs it running as a background service.
- **Settings interface.** Hardware name, thresholds, port — currently code-level. Should be user-editable; the key itself must remain neither user-editable nor user-visible.
- **TPM-backed key (Option B).** Deliberately deferred. Current model protects against the AI reading the key; it does not protect against someone with filesystem access while the service is offline.
- **`context_compression.py` not wired into a live loop.** Tested standalone only.
- **`real_high` telemetry arm** in the experiment harness fails to import `friction_index` from the experiment folder.
- **GPU support is NVIDIA-only.** `pynvml` finds nothing on AMD or Apple Silicon; falls back to CPU-only and records that it did.

---

## 11. Cross-cutting principle applied throughout

Every calibration choice is labeled as such in code rather than presented as derived. Component weights, threshold values, the 15°C thermal span, the 5-sample window, `keep_fraction`, `working_memory_size` — all are documented as tuned, not proven. Where a capability is absent (no GPU, no process isolation at the stub stage, no abstractive compression), the code records the absence in the audit trail or docstring rather than degrading silently.

This follows the standard the original README set: *every claim is either tested and shown, or explicitly marked as untested.*

---

## CORRECTION (supersedes the s₅ "contradiction" finding above)

The earlier conclusion that **s₅ had two incompatible definitions was wrong**, and the correction matters because it changes what needs editing.

There is only one F in the framework: **F(X) = free energy / prediction error**, the quantity whose gradient drives the master equation via `−∇_g F(X)`. This is confirmed by the UCE framework figure, which defines it explicitly, and it is consistent across the Unified Matrix drafts and Friston's formulation.

Given that, **s₅ = `E_core = E_d · E[‖∇_g F‖²]` is the magnitude of the free-energy gradient** — how hard the system is being pushed to reduce prediction error. The derivation table's protocol ("running average of ‖∇L‖² from optimiser state") is the computable stand-in, since loss gradient proxies free-energy gradient. That entry was never in conflict with anything.

**What actually happened:** the grant proposal reused the symbol F for an unrelated engineering quantity — a hardware load ratio, `H / X_base`. That is a naming collision in one document, not a competing definition of s₅. Searching all four documents confirms `X_base` in that scalar sense appears **only** in the grant proposal; it is absent from the derivation table, Unified Matrix v5, and the first-draft Unified Matrix.

**Resolution adopted:** the hardware ratio is renamed **Λ (Load Index)**. It is a real, useful, implemented instrument — but it is telemetry, not part of the formal dynamics, and it is not s₅. F is now reserved for free energy throughout.

**Code updated accordingly:** `friction_index.py` → `load_index.py`; `FrictionIndexMonitor` → `LoadIndexMonitor`; `FrictionReading` → `LoadReading`; the `F` field → `lam`; notification strings now read `[PRIORITY_ALPHA: HIGH_LOAD | Λ=<value> | ...]`.

**Second collision worth noting:** `X_base` also means two different things. In the Core Architecture it is a homeostatic reference **state vector** obeying `Ẋ_base = α(X − X_base)`. In the Load Index it is a scalar resting hardware load. The exponential moving average used in code is the discrete-time solution of that same ODE form, so the structures agree — but the objects do not, and the documents should distinguish them.

**Remaining document edits implied:**
- Grant proposal: rename F → Λ throughout the Friction Index sections; state that Λ is telemetry, not a term in the dynamics.
- Grant proposal: note that the Λ component weighting is calibrated, not derived (no formulation for it exists in any document).
- All documents: reserve F for free energy exclusively.
