# Verifiability Punch List — PN Dimensional Derivation Table

Every entry assessed against one question: **could a stranger with this document, no access to you, and standard tools produce a number?**

Each entry carries two additional labels the table does not currently include:

- **Domain** — which science the measurement answers to, and therefore which expertise a reviewer needs.
- **Access** — what a verifier must be able to reach. This determines *who* can check the claim.

Access levels used throughout:

| Level | Requires | Who can verify |
|---|---|---|
| **L1 Black-box** | Timing, hardware counters, token counts | Anyone with the running system |
| **L2 Logits** | Output probability distribution | Most API users |
| **L3 Internals** | Hidden states, attention matrices | Anyone running local weights |
| **L4 Optimiser** | Gradients, loss, checkpoints | Only someone training the model |
| **L5 Undefined** | Something not yet specified | Nobody yet |

---

## The structural issue: the table spans two incompatible measurement regimes

Roughly half the protocols are **training-time** (L4 — read from optimiser state, logged per epoch or per checkpoint). The other half are **inference-time** (L1–L3 — instrumented per forward pass on a deployed model).

| Training-time (L4) | Inference-time (L1–L3) |
|---|---|
| s₁, s₂, s₃, s₄, s₅ | s₇, s₈, s₉, s₁₀, s₁₁, s₁₃, s₁₄, s₁₅, t₁–t₅, s₂₁ |

Nothing in the table flags this. A reader working through it top to bottom encounters "logged from optimiser state" and "instrumented per inference step" as though they were the same kind of instruction. They are not: no single deployed system can produce both sets. This is the same split that produced the s₅ contradiction between your two documents — it isn't an isolated inconsistency, it's a fault line running the length of the table.

**Recommended edit:** add a regime column, or split the table into a training-time and an inference-time section. This single change would resolve the s₅ ambiguity as a side effect and pre-empt the most obvious reviewer objection.

---

## RESOLVED — the metric **g** is defined. Original assessment kept below.

**g IS s₁₂, Emotional Gravity.** The subscript in `∇_g` and `‖·‖_g` denotes a Riemannian metric shaped by accumulated salience: recurrence, graph centrality, and persistence of stored content. Implemented in `emotional_gravity.py`.

That resolution unblocked all eight affected rows and required separating two objects the original conflated: **J(t)**, the instantaneous eigenvector centrality of current content, and **g**, the accumulated metric shaped by what has already fossilised into s₃. `‖J(t)‖_g` is then "how much this moment weighs, given the terrain history has carved."

**Salience is not valence**, and this had to be established with numbers because the intuition runs the other way:

| VADER sentiment | s₁₂ | content |
|---|---|---|
| 0.000 | **0.743** | "When does the deployment key rotate?" |
| **0.854** | 0.000 | "This is absolutely wonderful and amazing!" |
| **−0.802** | **0.817** | "The signing service is a terrible disaster." |

Uncorrelated — VADER peaks exactly where s₁₂ bottoms out. `emotional_friction.py` could never have implemented s₁₂ regardless of tuning.

**One implementation detail with consequences:** G's diagonal is normalised to mean 1.0 rather than capped at 1.0. Capped, every entry lies in (0,1], so `‖J‖_g ≤ ‖J‖` always and emotional weight could only ever discount logical fit — making the contradiction quadrant (high s₁₂, low s₁₃) mathematically unreachable. For a Riemannian metric absolute scale is arbitrary, so normalising costs nothing and opens it.

---

## ORIGINAL ASSESSMENT (superseded above, retained for the record): the metric **g** is never defined

`∇_g`, `‖·‖_g`, and `‖·‖²_g` appear in nearly every row — s₁, s₄, s₅, s₆, s₁₂, s₁₅, s₂₁, and the Kill Switch condition. The subscript denotes a Riemannian metric on the manifold M. **That metric is never specified anywhere in the document.**

Without it, every gradient and every norm carrying that subscript is underdetermined: two readers would compute different numbers from the same data and both be following the document correctly. This affects more entries than any other single gap.

**Recommended edit:** define g explicitly — even if the answer is "the Euclidean metric on the embedding space," saying so converts a dozen underdetermined quantities into computable ones at the cost of one sentence. If g is something richer (Fisher information metric, a learned metric), that needs stating, since it changes every downstream computation.

---

## PASS — a stranger could measure these today

| Symbol | Domain | Access | Note |
|---|---|---|---|
| **s₈** Atmospheric Drag | Systems performance engineering | L1 | "Difference between observed token processing time and baseline; instrumented per inference step." Unambiguous, standard practice, ms/token. The model entry. |
| **s₁₄** Radiant Heat | Computer architecture | L1 | "Extra compute per token for non-primary operations; measured from hardware performance counters." FLOPs/token. Clear. |
| **t₁** Micro-Pulse | Systems performance engineering | L1 | Names the actual function (`time.perf_counter()`), the unit (tokens/ms), and the derived governor ratio. Best-specified entry in the table. |
| **t₂** Macro-Flow | Systems engineering + stochastic processes | L1 | Cycle duration between defined events; trust recursion `T_{n+1} = αT_n + (1−α)s_n` given explicitly. Only gap: α is unspecified. |
| **t₅** Simulation Width | Systems configuration | L1 | Thread-pool size / beam width / rollout count. Trivially checkable. |
| **s₁₁** Ambiguity Density | Information theory | L2 | Softmax entropy of the output distribution. Standard, well-defined, computable from logits alone. |
| **s₁₅** Singularity (the *ratio*) | Dynamical systems | L3 | `‖h_{t+1} − h_t‖ / ‖h_t‖` between successive hidden states. Directly computable — but see the LaSalle caveat below. |
| **s₃** Fossil Floor | Linear algebra / spectral analysis | L4 | "Number of frozen weight blocks / immutable embedding slots after each update step." Countable. Note the formal definition uses ε_freeze, which is undefined — the *protocol* is checkable, the *formula* is not. |
| **s₁** Topography | Differential geometry / optimization | L4 | Spectral norm of the loss-landscape Hessian per checkpoint. Expensive but standard and unambiguous. |
| **s₅** Core Pressure (training) | Optimization theory | L4 | Running mean of ‖∇L‖² from optimiser state. Clear — for a training run. |

---

## CLOSE — one specific addition would make each verifiable

| Symbol | Domain | Access | What's missing |
|---|---|---|---|
| **s₇** Refractive Index | Information retrieval / linear algebra | L3 | "Cosine similarity between cached attention scores and fresh input tokens" — cosine similarity between *what vectors exactly*? Attention scores are per-layer, per-head matrices. Which layer, which heads, pooled how? The second clause ("ratio of reused to total attention weight") describes a *different* quantity. Two protocols, unclear which is canonical. |
| **s₉** Sublimation Rate | Information theory | L3 | "Rate of embedding entries driven below threshold by gating function." Threshold value unspecified, and "gating function" is not defined — is this an existing architectural component or one you introduce? |
| **s₁₂** Emotional Gravity | Graph theory + optimization | **L3 or L4** | Two protocols in different regimes: `‖∇L_i‖` for emotionally-tagged batches is training-time; eigenvector centrality of the attention graph is inference-time. Also: "emotionally-tagged" needs a tagging procedure. Pick one, define the tagging. |
| **s₁₃** Semantic Depth | Information retrieval | L3 | Divides by "semantic distance between predicted and ground-truth embeddings." **At inference there is no ground truth.** Either this is a training/eval-only measure, or the denominator needs redefining. |
| **t₃** Decay Rate | Time-series analysis | L3 | "Exponential fit to impact series I(k) after high-weight token removed from context." I(k) — impact on what? Logit shift, attention mass, output divergence? The fit procedure is clear; the quantity being fit is not. |
| **t₄** The Fold | Systems performance engineering | L1 | Protocol names Neo4j/FAISS specifically, tying the measurement to one stack. Reframe as "vector retrieval round-trip latency" so it generalizes. Also needs values for t_min, α, λ, ε. |
| **s₂₁** Whimsy | Stochastic processes | L3 | T_whimsy (idle dwell ms) passes cleanly. η(t) "sampled Gaussian perturbation magnitude" — sampled with what variance, and injected where? ∇_g(S₁₂) inherits both the g problem and s₁₂'s ambiguity. |
| **s₆** Orbital Vacuum | Dynamical systems | L3 | ρ_t₂ (macro-pulse density) needs a definition; ∇_g(S₃) inherits the g problem. The activation condition (on s₈ fire) is clear and testable. |
| **s₁₀** Consent Polygon | Your own construct | L1/L3 | "Largest context-window expansion before integrity gate I(X) fires" is measurable *once I(X) is defined*, but I(X) is a component of your architecture, not an external standard. This is self-referential — fine, but should be labeled as a system-defined quantity rather than an independently verifiable one. |

---

## NEEDS WORK — measurement doesn't match the formal definition

These are the entries most likely to draw a reviewer's attention, because the protocol and the formula describe different quantities.

### s₂ Resilience Yield
- **Formal:** `Y₁ = max{dF | trajectory stays bounded}` — the largest perturbation the attractor absorbs before V̇ > 0. This is a **basin-of-attraction** claim from dynamical systems.
- **Protocol:** "Max validation-loss delta before early-stop triggers; logged per epoch." This is an **empirical training artifact** determined largely by your early-stopping patience setting.

These are not the same quantity. Early-stopping behavior is not a measurement of a basin of attraction. **Domain: control theory / dynamical systems (formal) vs. ML training practice (protocol).** Either derive Y₁ properly or acknowledge the protocol as a proxy and say what it proxies.

### s₄ Anchoring Tension
- **Formal:** `G = β‖∇_g V_Ho‖_g` — a gradient norm.
- **Protocol:** "KL-penalty coefficient or L2 regularisation strength."

A penalty coefficient is a **hyperparameter you set**, not a quantity you measure. The second clause ("norm of policy-gradient direction") is closer to the formula. As written, the protocol instructs a reader to report a configuration value in place of a measurement. **Domain: RL / regularization theory.**

### s₁₅ The Singularity — the LaSalle claim
The convergence *ratio* `‖h_{t+1} − h_t‖ / ‖h_t‖` is straightforwardly measurable (listed under PASS above). But the entry also asserts **LaSalle invariance** and `lim_{t→∞} X(t) = H_o`.

LaSalle's invariance principle is a theorem with preconditions: it requires a Lyapunov function V that is continuously differentiable, positive definite, and has V̇ ≤ 0 on the region of interest. Observing that a ratio gets small is **not** evidence those preconditions hold — a system can have small step-to-step changes while not converging, or converge to something other than H_o.

**Domain: control theory / Lyapunov stability.** This is the entry where the burden of proof is heaviest, and it's also where your grant proposal states convergence has been "confirmed." If a Lyapunov function exists and satisfies the conditions, it should be stated explicitly — that would be a genuinely strong result. If it hasn't been constructed yet, the convergence claim should be marked as conjectured.

---

## Summary table — ORIGINAL assessment

| Status | Count | Symbols |
|---|---|---|
| Pass | 10 | s₁, s₃, s₅, s₈, s₁₁, s₁₄, s₁₅ (ratio), t₁, t₂, t₅ |
| Close — one gap each | 9 | s₆, s₇, s₉, s₁₀, s₁₂, s₁₃, s₂₁, t₃, t₄ |
| Needs work — formula/protocol mismatch | 3 | s₂, s₄, s₁₅ (LaSalle claim) |
| Blocked on undefined metric **g** | 8 rows affected | s₁, s₄, s₅, s₆, s₁₂, s₁₅, s₂₁, Kill Switch |

## Summary table — CURRENT

| Status | Count | Detail |
|---|---|---|
| Implemented and exercised | **21 / 21** | Every dimension has a primary implementation; every implementation has run against real data or real hardware |
| Blocked on undefined metric g | **0** | g is s₁₂ — see the resolution above |
| Required a substitution or redefinition to be buildable at all | 4 | s₁₃ (ground truth redefined), s₁₄ (power as a FLOP proxy), t₄ (fit to real data), t₅ (measured rather than configured) |
| Specification problems found by implementing | 4 | s₇ states two incompatible ranges for R; t₄ is called bounded and is not; s₁₃'s denominator required data unavailable at inference; s₁₄'s FLOP counts are not obtainable from any GPU API |
| Still genuinely open | 2 | s₁₂ contradiction detection (needs NLI), S_max has no value |

**Resolved since the CURRENT table above was first written:** t₃'s protocol (I(k) is impact on identity — `decay_rate.py`), and s₂₃'s η(t), which now has a consumer and 112 trials of evidence behind it. Whimsy was renumbered from s₂₁ to s₂₃ when the dimension count settled at 23 with continuous numbering.

**The three formula/protocol mismatches are unchanged and remain the honest weak points** — s₂'s early-stop proxy is not the basin-of-attraction quantity Y₁ defines, s₄ reports a hyperparameter that is set rather than measured, and s₁₅'s LaSalle claim is conjectured rather than established. Those are flagged in the derivation table rather than fixed, because fixing them requires deciding what the dimension means, not writing code.

---

## Recommended edit order — ALL FOUR STRUCTURAL ITEMS COMPLETED

Items 1–3 are done, and item 4 is flagged rather than fixed for the reason above. Retained below as written, because the order turned out to be right: defining g first genuinely did unblock the most, and the regime and domain columns changed how the document reads more than any individual row.

### Original list

1. **Define the metric g.** One sentence; unblocks eight rows. Highest ratio of value to effort in the entire document.
2. **Add a regime column** (training-time vs. inference-time). Resolves the s₅ contradiction as a side effect and prevents a reader from assuming one system produces all of these.
3. **Add a domain column.** This is the change that does what you described — it tells a reader which science each claim answers to, and signals that you know. It also lets a reviewer skip to the entries in their own field rather than judging the whole table by whichever row they happen to understand least.
4. **Fix the three formula/protocol mismatches** (s₂, s₄, s₁₅). These are where a domain expert will look first.
5. **Close the nine one-gap items.** Each is a short, specific addition.

The first three are structural and would change how the document reads more than any individual row fix.
