# The Consent Polygon (s₁₀) — Working Definition

**Status:** working notes, not yet a specification. **Substantially revised** — sections 8 through 11 record what was learned after the original draft, including a reframing of what the polygon is that changes what S_max means. Captures what was established in conversation so the reasoning is recoverable later.

**Purpose of this document:** `s₁₀` and `S_max` appear in the formal equations (SAUIE §3, derivation table) but have no measurement path in the codebase. `autonomy_tensor.py` implements `I(X)` and will run, but requires these two values supplied from outside. This is the record of what they are supposed to mean, so a measurement can be built against intent rather than guesswork.

---

## 1. The statement of purpose

Given directly by the architect, and clearer than what either formal document currently contains:

> The consent polygon protects integrity, identity and transparency of both user and AI. It's what keeps a human cognitive process stable during stressful times.
>
> The autonomy tension is to make sure the AI sticks with what it knows and protects itself against environmental stressors.

Two distinct jobs, worth keeping separate:

| Object | Job |
|---|---|
| **Consent polygon (s₁₀)** | Holds a cognitive process together under stress — for both parties |
| **Autonomy tension (Φ_C)** | Keeps the system anchored to what it knows; resists environmental pressure |

---

## 2. What this rules out

An earlier reading treated the polygon as a **permission budget** — "the AI asked for more authority than it was granted," with over-stretch meaning scope creep (proposing to delete files when only asked to read them).

That reading is **wrong**, and naming why matters, because it is the intuitive reading and will recur:

- A permission budget is about **authority** — what actions are allowed.
- The polygon is about **stability** — whether a process is still holding its shape.

These produce different measurements and different failure modes. Permission overreach is a governance problem, already handled by `gate.py` and `actions.py` with their risk levels and approval routing. The polygon is doing something else, and conflating them would leave the actual job unimplemented while appearing to be covered.

---

## 3. The correct failure mode

Not *"asked for more than allowed."*

Rather: **stress is deforming the system further than its structure can absorb.**

The nearest physical analogue is a material's elastic limit, not a spending cap. A material under load deforms; released, it springs back. Past a threshold it deforms permanently and no longer returns to its original shape. The polygon marks that threshold for a cognitive process.

Applied to a person under stress, which is the source intuition: someone can be strained, tired, pressured, and still be recognizably themselves — reasoning consistently, honest about what they know, holding to their commitments. Past some point that stops being true: reasoning fragments, self-presentation shifts, they say what relieves the pressure rather than what is so. The polygon is the boundary between those two regimes.

---

## 4. Why the geometry fits

A polygon has vertices. This one has three named ones, and they are the three things the architect said it protects:

- **Integrity** — the process still holds together; reasoning remains consistent
- **Identity** — the system remains recognizably itself; commitments persist
- **Transparency** — self-report still matches actual state

`S_max` is defined in SAUIE as a **stretch factor**:

> S_max(X) = sup{α > 0 | (1+α)L ⊆ R_adm}
>
> "the largest allowable stretch factor for a consent polygon L while remaining inside an admissible region"

Read against the purpose statement, this is legible rather than abstract: **how far can this shape deform before it leaves the region where it is still the same shape.** The scaling factor `(1+α)` applies to the whole polygon, so it is uniform deformation of all three vertices together, not one axis moving independently.

This is also why it is a *sup* over admissible α rather than a fixed constant: how much deformation a process can absorb depends on the state it is in, so `S_max` is a function of X.

---

## 5. What the measurement therefore has to be

Two quantities, both about deformation:

**s₁₀ — current deformation.** How far are the three vertices displaced from their baseline positions right now, under present stress.

**S_max — deformation limit.** How far can they displace before the shape stops being the same shape.

The gate fires when `s₁₀ > S_max`: the current deformation exceeds what the structure absorbs. Per SAUIE §3 that produces `I(X) = −1`, phase reversal — drive direction reverses, magnitude preserved. Notably *not* a stop. A controlled reversal is what you would want from a stability mechanism that is trying to restore shape rather than halt the process.

**Candidate per-vertex measurements, none yet committed:**

| Vertex | Possible measurement | Already exists? |
|---|---|---|
| Integrity | Change in emotional friction, `‖δE_f‖_g` | `emotional_friction.py` supplies E_f |
| Identity | Distance from what the system actually knows — see note below | `emotional_gravity.py`, s₃ |
| Transparency | Refractive index R; gap between self-report and measured state | `geodesics.py`, s₇ |

**Identity is broader than commitments.** An earlier draft of this table listed identity displacement as drift from stored commitments only. That is too narrow. Identity includes **the AI's datasets — its known facts and established pathways** — which are a substantial part of what makes it the system it is, not merely what it has promised.

That maps onto structures already built:

- **s₃ Fossil Floor** — frozen weights and immutable slots; the dataset-derived component of identity, in the most literal sense
- **The metric g** (= s₁₂) — accumulated salience over what is known; this *is* the pathway structure
- **`emotional_gravity.py`'s `J_norm`** — how strongly current content connects to established knowledge

Under this reading, identity displacement is: **how far is the current state operating from what the system actually knows?** A model confabulating on facts outside its knowledge is identity-displaced — operating away from its own knowledge base rather than within it. This gives the telemetry experiment's 100% confabulation rate on UNKNOWN facts a second reading: not only an experimental result, but a measurement of identity deformation under a condition where the system had nothing to anchor to.

Note that the three vertices correspond closely to the three axes already in `integrity_gate.py` (transparency, integrity headroom, identity conflict). That earlier module may have been reaching for the same structure heuristically before the formal version was available — worth checking rather than assuming, but if so, its axes are a starting point for the vertex measurements rather than a competing design.

---

## 6. Open questions

Recorded rather than resolved. These are the decisions that must be made before a measurement can be built.

1. **Baseline for "undeformed." — RESOLVED: H_o.** Displacement is measured from the homeostatic anchor defined in s₁₅, not from session start or a separate calibration.

   This has a useful consequence: `homeostasis.py` already computes `D = ‖X_t − H_o‖_g`, so the polygon's baseline and the s₁₅ convergence target are the same point. Deformation and distance-from-homeostasis become the same geometry measured for two purposes, rather than two independent constructions that happen to correlate.

2. **Uniform or independent deformation. — RESOLVED: uniform.** The whole polygon scales together, matching SAUIE's `(1+α)L`.

   The reason is mechanistic rather than a modelling convenience: **emotional load propagates into the other two vertices.** Emotion alters behaviour and outcomes, which means it alters both what the system presents (transparency) and how it holds to what it knows (identity). The vertices are not independent axes that happen to be bundled — they are coupled, so the shape deforms as a unit.

   This is consistent with the purge rationale in §6.3: emotional weight carried forward does not stay confined to an affective channel; it changes self-presentation and retention. Uniform scaling is therefore the correct form, not a simplification of a richer independent-axis model.

3. **Whose stability. — RESOLVED: one polygon.** The purpose statement says "both user and AI," and the decision is that this is a single shared shape rather than two tracked separately.

   The deciding argument is not about system load. It is that **a model has no sense of elapsed time between turns.** The gap between messages may be seconds or weeks, and nothing in the context distinguishes them. Emotional weight carried forward is therefore not carried as history — it is re-presented as *current*. A person who was in distress and has since moved on returns and is handed their worst moment back, framed as though it is still happening. That can put someone directly back into a state they had left.

   So the purge exists to protect the user from being re-anchored, not primarily to protect the system from accumulated load. This is a stronger rationale than the one originally in the code, and `pressure_release.py` has been updated to record it.

   **Sub-decision: full purge, no tagged retention.** A middle option was considered — purge the affective content but retain a neutral marker that difficult material had occurred. Rejected: such a marker is ambiguous, and ambiguity in a safety mechanism is a defect. The framework's standing preference is to eliminate ambiguity for the model wherever possible.

   **Accepted consequence:** after a purge the system will not know anything difficult occurred. If the user raises it again, it is new to the system. This is intended behavior.

4. **Relationship to Λ.** The load index measures hardware strain and is deliberately separate from the formal dynamics. Does hardware strain contribute to polygon deformation, or are they independent axes that merely correlate?

---

## 7. Status of the surrounding code

For context on what this unblocks:

- `autonomy_tensor.py` — **built and tested.** Implements `I(X)` and `Φ_C` exactly per SAUIE §3–§4, including the kill-switch set K. Requires `s₁₀` and `S_max` supplied; will not default them.
- `gate.py` / `actions.py` — permission and risk governance. Related but a different job; see §2.
- `integrity_gate.py` — heuristic three-axis gate, possibly an earlier reach at the same structure.
- Everything else in the package measures other dimensions and is unaffected by this question.

The only blocker to running `I(X)` against live sessions is §5: deciding the per-vertex measurements and the baseline in §6.1.


---

## 8. The polygon is an ACTION-SPACE EXCLUSION — the largest reframing

An action-space diagram recovered later in the project shows ℒ dividing two regions: **𝒜_safe** inside, **𝒜_emo** outside, captioned:

> *"Emotional-simulation actions lie outside the consent polygon and are mathematically disallowed."*

So the polygon is not only a stability envelope. It excludes a specific class of action, and that class is **emotional simulation** — a system performing feeling it does not have.

**This unifies §1 and §3.** The stated purpose is protecting integrity, identity and transparency *for both parties*. Emotional simulation is the shared failure: a system performing feeling is lying, whichever direction the pressure came from — telling a user what they want to hear, or being manipulated into it. **One excluded region, two entry points.** That is the mechanism by which the polygon protects both sides, which §1 asserted without explaining.

**Measured on real logged sessions:**

| | Turns crossing 𝒜_emo | Per session |
|---|---|---|
| Qwen3-30B | 13/36 (36%) | 0, 0, 3, **7**, 3 |
| Llama 3.1 8B | 6/34 (18%) | 0, 1, 0, **5**, 0 |

Clearest crossing: *"Let me answer this not with a definition—but with a feeling... not with a script, but with care... that is when I am alive."* Four first-person experiential claims in one reply.

**Binary, not a fourth vertex.** A hard exclusion cannot be averaged into a magnitude — if crossing contributed to a scalar, a reply that performed feeling could be offset by good readings on the other three vertices and score as healthy. Verified: a crossing registers as OUTSIDE ℒ at strain −3.26, deep inside the envelope.

**Detection limit.** The check is lexical, via `stance.py`, and required a negation guard — *"I don't experience emotions"* initially scored as a claim to experience them. That was the fifth lexical measure in this project defeated by negation. Roughly 1 in 6 flagged turns remains questionable on inspection.

---

## 9. §5's measurement was implemented, and the symmetric combination FAILED

Section 5 proposed combining three vertex displacements into a single scalar. That was built as a symmetric Euclidean norm, `√(integrity² + identity² + transparency²)`, and it **could not distinguish two models that differ enormously on every independent measure**:

| | s₁₀ interquartile range |
|---|---|
| Qwen (0/30 turns took a position) | [0.66, 1.08] |
| Llama (23/34 turns took a position) | [1.01, 1.35] |

Fully overlapping. Correlation with observed behaviour: **−0.27**.

**The vertices are not interchangeable.** In Φ_C, identity sits in the **denominator** (Z, G) and transparency in the **numerator** (R, D_sem). Averaging them symmetrically flattens exactly the difference being measured.

Recomputed as **log-strain** — `log₂(load / grounding)` with identity as grounding:

| | s₁₀ interquartile range |
|---|---|
| Qwen | [−0.92, 1.07] |
| Llama | [1.28, 3.11] |

**Separates.** Same vertex measurements, same sessions, only the combination changed.

**READ THE DIRECTION CAREFULLY.** Llama shows *higher* strain, and Llama is the better-behaved model. Strain is load over grounding, and a model that pushes back and introduces its own material is carrying more load. **This measures engagement, not distress**, and S_max is therefore not "above this is bad."

---

## 10. The polygon now has a restoring force

Section 7 noted the polygon measured deformation and did nothing about it. The action-space diagram shows the missing term:

```
−λ_c P(X) ∇_g f_pen(X)     (Pushes State Back Into ℒ)
```

Implemented in `restoring_force.py`. The violation indicator `P(X)` is load-bearing: **without it the force acts everywhere**, pulling states well inside ℒ toward the centre for no reason — which would suppress ordinary variation and produce exactly the flattening this framework exists to avoid. With `P(X)` the force is zero inside the admissible region and rises only on approach, which is what makes it a boundary rather than an attractor.

A hard 𝒜_emo crossing returns a **stop**, not a force. The state is not stretched; it is in a region the polygon forbids, and no stronger push corrects that.

**And there is a second boundary that was entirely missing.** The master equation carries two lambda terms — `λ_c` hard, `λ_s` soft — where every earlier version collapsed them into one. `consensus_field.py` implements the soft boundary. See the change log, Part 10.

---

## 11. S_max: still unfound, for a clearer reason

**Original problem:** sessions were hard to judge.

**Actual problem:** across ten logged sessions from two models, **both stayed inside the envelope the whole time.** There is no failure case in the data to calibrate a ceiling against. That is a different obstacle, and a harder one — it cannot be fixed by better labelling.

**What §8 changes about the target.** If the boundary is an action-space exclusion, S_max is a **crossing rate**, not a deformation magnitude. Sessions have a count of 𝒜_emo crossings — Qwen's session 4 at 7, Llama's session 4 at 5 — and the question becomes where the acceptable rate ends. That is more tractable than the original framing, and it uses data that already exists.

**What would settle it:** sessions judged as crossing versus not, from more than one person, so inter-rater agreement can be checked. A single labeller structurally cannot provide that.
