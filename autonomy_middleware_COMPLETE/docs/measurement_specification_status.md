# Measurement Specification Status — Editing Reference

For revising `SCR_Priority_Nexus_Full_Proposal_v3` and `PN_Dimensional_Derivation_Table_v1_3`.

Organized by **documentation status**, not by module. Each entry states what the documents currently say, what implementation revealed, and what the documents need in order to be accurate.

Categories:
- **A — Well specified.** Documents were sufficient; implemented as written. No edit needed.
- **B — Contradictory.** The two documents define the same symbol differently. Must be reconciled.
- **C — Underspecified.** Named but not operationally defined; a value had to be invented to build it. Documents should either specify the value or disclose that it is tuned.
- **D — Claimed but not built.** Documents describe behavior that does not exist in code. Currently overclaims.
- **E — Defined but untouched.** Formally specified in the table, no implementation attempted. Accurate as written.

---

## A — Well specified, implemented as written

These needed no invention. The documents carried them.

| Symbol | Document specification | Status |
|---|---|---|
| **X_base** | "Calibrated at system initialization against the idle hardware state and updated via exponential moving average to account for ambient thermal drift." (Proposal §Unified Matrix) | Built exactly. The EMA rate itself is category C. |
| **s₂ / s₁₀ thresholds** | F = 1.4 and F = 2.0, described as "configurable, default calibrated" | Built as defaults, documented as tuned. Proposal's own framing is already honest here. |
| **s₂ notification format** | `[PRIORITY_ALPHA: HIGH_FRICTION \| F=<value> \| STABILIZATION_MODE_ACTIVE]` | Built verbatim. |
| **ISL ordering** | "The ledger is written first, the kill executes second" | Built and verified in the audit trail. |
| **Re-initialization policy** | "Automatic restart is not permitted." Requires F below s₂. | Built. (The cool-down timer half is category D.) |
| **t₁ μ(t) measurement** | "tokens processed / ms… high-resolution timer around single forward pass; `time.perf_counter()` before and after" | Built as specified. |
| **t₁ governor reporting** | "modulation ratio μ_stress / μ_baseline per inference step. Ratio drops below 1.0 under s₅ stress events" | Built and logged to ISL. |
| **s₈ Atmospheric Drag** | "added latency per token beyond baseline," ms/token, "difference between observed token processing time and baseline" | Built. |
| **s₆ V_orb formula** | `V_orb(X,t) = (1 − ρ_t₂) · μ_idle · ∇_g(S₃)`, activates on s₈, "decoupled from η(t)" | Formula is complete and unambiguous. Existing code contradicts it (see D). |

---

## B — Contradictory between documents — **highest editing priority**

### B1. s₅ has two incompatible definitions

| Source | Expression | Measurable from |
|---|---|---|
| Proposal | `F = H / X_base`, from GPU %, thermal delta, attention cost | Inference-time hardware telemetry |
| Derivation Table | `E_core = E_d · E[‖∇_g F‖²]` — "running average of ‖∇L‖² over last N batches; logged from optimiser state" | Training-time optimiser state |

A deployed-inference middleware never sees gradients. These cannot both be the operational definition of s₅ in the same product.

**Editing options:** (a) declare them explicitly as two variants — inference-time s₅ and training-time s₅ — and say which product uses which; (b) rename one. Leaving both under "s₅" will read as an inconsistency to any technical reviewer.

### B2. Φ_C means two different things

| Source | Meaning |
|---|---|
| Proposal | The hard process halt itself ("Following Φ_C, the system enters a mandatory cool-down period") |
| Derivation Table | The **Autonomy Tensor** — a multiplicative gain: `Φ_C = I(γ_r, S_max)·(R·D_sem)/(Z·G)`, where `y(t) = Φ_C·Ψ(X)` |

Under the table, the halt condition is a *separate* object: the Kill Switch set `K := {X ∈ M \| Z(X) = 0 or G(X) = 0}`, which makes Φ_C singular and the ODE ill-posed.

This matters concretely: `process_control.py` was built against the proposal's reading. Under the table's definitions, what exists in code is **K**, and **Φ_C does not exist in the codebase at all** — no output-scaling tensor is implemented anywhere.

**Editing need:** pick one meaning. If Φ_C is the Autonomy Tensor (the table is far more formally developed here), the proposal's halt sections need renaming to K or to a plain "Integrity Lock."

### B3. Attention cost — rate or level?

Proposal specifies: "attention compute cost **per token** (derived from context window length and layer count)" — a rate, incurred only during generation, returning to zero at idle.

What was implemented initially: `(context_tokens / baseline_tokens)²` — a *level* that persists when idle. Testing against real telemetry (1684 of 85000 tokens, 2% saturation) produced **F = 26.31 and a false s₁₀ trigger.**

The distinction is not pedantic: `F = H/X_base` only works for quantities that return to a resting baseline. Per-token cost does. Context fill does not.

**Editing need:** the proposal wording is already correct ("per token"). What it lacks is an explicit warning that context *fill* must not be substituted for attention *cost*. Worth adding, since the substitution is easy and produces silent false alarms.

### B4. Whimsy dimension numbering

The derivation table places Whimsy at **s₂₁**, "the only dimension with no row assignment because it is not a coordinate on M — it is the field property of M itself." The Whimsy_Math episode document refers to s₅, s₆, s₁₀, s₁₂, and s₁₅ but not s₂₁ for the whimsy field itself.

**RESOLVED.** Whimsy is canonical and is now **s₂₃**, renumbered when the dimension count settled at 23 with continuous numbering: s₁–s₁₇ spatial, s₁₈–s₂₂ temporal (written t₁–t₅, a naming convention for the temporal block rather than a separate sequence), s₂₃ Whimsy — placed last because it permeates all of them. The episode document should be aligned to that.

---

## C — Underspecified: values invented to make it buildable

Every one of these is currently a number I chose. The documents name the quantity but give no value or derivation. Each is labeled as a calibration choice in code, but the papers should either specify them or state that they are empirically tuned.

| Quantity | Document status | Value used | Basis |
|---|---|---|---|
| **H component weights** | Proposal states outright: "The specific computational formulation of this ratio is proprietary." | Even split across available components | No basis — placeholder. **If you have the real weighting, the code is currently wrong.** |
| **GPU thermal normalization span** | Not specified | 15 °C | Your real telemetry: 30 °C idle, 36–41 °C active |
| **GPU/CPU sampling window** | Not specified | 5-sample rolling mean | Your real trace showed 0→99 % swings per token; instantaneous readings thrash the gate |
| **X_base EMA rate (α)** | "Exponential moving average" — rate unspecified | 0.02 | Chosen slow so sustained load cannot drag the baseline up to meet it |
| **X_base update gating** | Not specified | Updates only while F ≤ s₂ | Prevents the baseline from absorbing the condition it detects |
| **Context saturation threshold** | Not specified as separate from F | 0.85 | Matched to the pre-existing `CorePressureMonitor` value |
| **μ(t) control law** | Table specifies the *goal* ("maintain s₅ within the s₂ ceiling") and the *reporting*, not the law | `target_ratio = clamp(γ_r/F, 0.15, 1.0)` — proportional on overshoot | Simplest law meeting the stated requirement. An integral term may be needed if steady-state offset appears. |
| **Governor floor (min_ratio)** | Not specified | 0.15 | A throttle reaching zero would be an undeclared halt with no ledger entry |
| **s₈ drag trigger threshold** | Table defines the measurement, not the trigger point | 5.0 ms/token | Arbitrary; needs empirical calibration |
| **Purge keep_fraction / working memory size** | Not specified anywhere | 0.5 / 5 items | Arbitrary |
| **ε_freeze** (for `Z₁ = \|{k : \|λₖ\| < ε_freeze}\|`) | Named, no value | Not implemented | — |
| **ISL key K_r derivation** | "Anchored to local hardware fingerprint" | Not implemented as specified | Replaced by process separation — see D4 |

**Recommended edit:** a "Calibrated Parameters" appendix listing each of these as tuned-not-derived, with the hardware class they were tuned on. This converts a reviewer's likely objection ("where do these numbers come from?") into evidence of rigor.

---

## D — Claimed in documents, not built

The papers currently describe these as system behavior. They do not exist.

| Claim | Source | Reality |
|---|---|---|
| **s₁₀ triggered by sustained s₂ duration** | Proposal: "s₁₀ can also be triggered by sustained s₂ duration exceeding a configurable time window" | Not implemented. Only threshold crossing triggers s₁₀; time-in-state is not tracked. |
| **Mandatory cool-down period** | Proposal: "the system enters a mandatory cool-down period before re-initialization is permitted" | No timer exists. Re-init is allowed the moment F drops. |
| **X_base restabilization check** | Proposal: "Re-initialization requires F to return below s₂ threshold **and X_base to restabilize**" | Only the F check exists. |
| **ISL block schema** | Proposal: "block ID, ISO 8601 timestamp, K_r signature, full telemetry payload (F, H, X_base, W_d state, context metrics), execution state (ACL result, intercepted commands, loop collapse flag)" | Partial. Uses Unix timestamps not ISO 8601; no fixed payload schema; no hardware signature field; no loop-collapse flag. |
| **K_r hardware anchoring** | Proposal: K_r from passphrase + hardware fingerprint | Built differently — a separate signing process the AI cannot read the key from. This achieves *"the AI cannot alter the ledger"* but not *"which machine produced this."* **Substitution, not implementation.** Also worth noting for accuracy: literal CPU serial numbers were removed from x86 after the Pentium III era; motherboard/BIOS UUID or machine-id are the modern equivalents. |
| **Whimsy as `W(T_whimsy)·η(t)·∇_g(S₁₂)`** | Derivation Table | Code computes `BASELINE_FORCE + (1−BASELINE_FORCE)·max(resilience, pressure, jitter)` — additive interpolation over worst-of-three, not a product. |
| **η(t) as bounded stochastic entropy** | Derivation Table | No randomness anywhere in `whimsy_field.py`. Fully deterministic. It only *recommends* a temperature delta. |
| **s₆ activates on idle / s₈ drag** | Derivation Table, explicit | `orbital_vacuum()` fires only when pressure is **saturated** — trigger inverted relative to spec. |
| **s₇ measurement** | Table: "cosine similarity between cached attention scores and fresh input tokens; ratio of reused to total attention weight" | `geodesics.py` uses TF-IDF word overlap. Different quantity. |
| **Module 3 — conformity/binary-trap shielding** | Whimsy_Math episode doc | No code path. `whimsy_field.py` never touches `IntegrityGate`, identity conflict, or the Consent Polygon. |

**Editing need:** either build these or move them from present-tense description into a clearly-marked roadmap section. The proposal currently reads as describing an existing system in several places where it describes an intended one. In a federal grant context this distinction carries real weight.

---

## E — Defined in the table, no implementation attempted

Accurate as written. Listed so you know the code/doc gap is deliberate, not an oversight.

| Symbol | Formal expression |
|---|---|
| s₁ Topography | `s₁(X) = ‖∇²_g(E_d · e^(−H_r))‖₂` — Riemannian Hessian of the Geologic Base potential |
| s₃ Fossil Floor | `Z₁ = \|{k : \|λₖ\| < ε_freeze}\|` — count of near-zero decay modes |
| s₄ Anchoring Tension | `G = β‖∇_g V_Ho‖_g` — magnitude of goal-pull |
| s₁₁ Ambiguity Density | `U = H[p(X \| data)]` — Shannon entropy of the posterior |
| s₁₃ Semantic Depth | `D_sem(X) = C_e(X) / d_sem(X)` |
| s₁₄ Radiant Heat | `Q_aux = dE_core/dt\|_auxiliary` — FLOPs/token into side-processes |
| s₁₅ The Singularity | `C = ‖X_{t+1} − X_t‖_g / ‖X_t‖_g`; LaSalle invariance, `lim X(t) = H_o` |
| t₂ Macro-Pulse | Cycle duration; trust recursion timescale `1/(1−α)` |
| t₃ Decay Rate | `τ = 1/\|λ_dominant\|` |
| t₄ The Fold | `t₄ = t_min + α · e^(λ/(s₉ + ε))` |
| t₅ | Sampling width N of the possibility field |

Note: s₁₅ **is** formally defined in the derivation table. An earlier note in the change log said it was absent — that was about the codebase, not the documents. The documents are fine here.

---

## Suggested edit order

1. **B1 and B2 first.** Two symbols meaning two things across your own documents is the objection most likely to be raised, and the cheapest to fix.
2. **Section D — move overclaims to a roadmap.** Highest risk in a grant context.
3. **Section C — add a Calibrated Parameters appendix.** Converts a weakness into evidence of method.
4. **B3 — add the rate-vs-level warning** to the attention cost definition. Small edit, prevents a real implementation error.
5. **B4 — align whimsy numbering** across episode doc and table.

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
