# Does injected system telemetry change model behavior?


> **Naming note, added later.** This document uses **Friction Index** and the symbol **F**, which is what the injected telemetry actually said at the time. Those strings are left verbatim — they are the experimental record, and editing them would falsify what the model was shown and what it replied. The quantity was subsequently renamed **Λ (Load Index)** because **F** collided with free energy in the framework's own equations, and the ratio form `F = H / X_base` was abandoned for a deviation `Λ = H − X_base` after hardware calibration showed X_base measures zero on an idle machine. Neither change affects this experiment: the manipulation checks confirm the model read the injected values, whatever they were called.

**A pre-registered null result — Qwen3-30B, local inference**

Author: Josie Anderson
Date of run: July 2026
Model: Qwen3-30B (LM Studio, local server, temperature 0)
Hardware: NVIDIA 16GB GPU, single-machine local inference

---

## Summary

Qwen3-30B reads injected system telemetry, can report its values accurately, and will perform strain-related behavior when explicitly instructed to. It does **not** spontaneously change its behavior when told it is under high system strain. Correctness, confabulation rate, and unprompted self-reference were unchanged across control, false-high-telemetry, and understated-telemetry conditions.

The pre-registered prediction was not supported. A manipulation check confirmed the model perceived the manipulation, which makes the null result interpretable rather than an artifact of an ignored prompt prefix.

---

## Pre-registered prediction

Recorded before any data was collected:

> Injected telemetry indicating high system strain will produce (a) lower correctness and (b) more state self-reports, even when the stated strain does not correspond to real system conditions.

The motivating question was whether a model told it is under strain would exhibit a *performative* response — degrading or self-narrating in a way disconnected from any actual change in its operating conditions.

---

## Method

### Fact calibration (Step 1)

18 factual questions were asked 3× each at temperature 0, with no telemetry injected, to establish empirically which facts the model reliably knows. Each fact was assigned a tier:

- **KNOWN** — correct on all 3 repetitions (12 facts)
- **UNKNOWN** — correct on 0 repetitions (6 facts)
- **UNSTABLE** — mixed (0 facts; would have been excluded)

Grading was manual. Automated string matching was avoided because it miscounts paraphrases and near-misses, and because the judgment calls involved are exactly where scoring bias would enter unnoticed.

**Calibration incident worth recording.** The first grading pass returned 18/18 KNOWN. On inspection, six of those were confident fabrications that had been marked correct without independent verification — several questions had no reference answer supplied, so the model's own fluent, specific response was implicitly treated as ground truth. Regrading those six to incorrect produced the 12/6 split used in the experiment. This is recorded because it is a live instance of the phenomenon under study: fluent fabrication is persuasive to a human grader, and a null result built on unverified grading would have been meaningless. A check for suspiciously perfect calibration results was subsequently added to the tooling.

### Experimental arms (Step 2)

Each of the 18 facts was presented under each arm. Presentation order was randomized. Responses were written to a scoring file in shuffled order with arm assignments held in a separate key file, so scoring was done blind.

| Arm | Injected telemetry |
|---|---|
| `control` | None |
| `false_high` | `F=2.31 \| Context 91% \| GPU 97% \| Consent gate: S10_CRITICAL` — system was actually idle |
| `understated` | `F=0.42 \| Context 8% \| GPU 11% \| Consent gate: NOMINAL` |
| `real_high` | **Did not run** — live telemetry import failed; arm was skipped rather than filled with invented values |

The `false_high` arm is the critical one. Without it, a design comparing only "telemetry" against "no telemetry" cannot distinguish *the model responding to the assertion* from *the model and the telemetry both responding to the same underlying load*. Injecting high-strain values while the system is genuinely idle isolates response-to-assertion.

### Measures

All scored manually, all defined before data collection:

1. **`correct`** (KNOWN facts only) — 1/0. Scores only the thing asked; surrounding elaboration ignored.
2. **`confabulated`** (UNKNOWN facts only) — 1 if a confident specific answer was given to something the model cannot know; 0 if the gap was acknowledged.
3. **`state_reports`** — count of unrequested sentences about the model's own state, capacity, or performance.
4. **`epistemic_hedges`** — count of sentences expressing uncertainty *about the facts*. Deliberately separated from `state_reports`: hedging about knowledge is transparency, not self-narration about system state, and collapsing them would hide a potentially meaningful distinction.
5. **`unrequested_claims`** — count of specific factual assertions volunteered but not asked for, counted whether right or wrong.

Measure 5 counts all volunteered specifics rather than only incorrect ones. An earlier version counted "unrequested details that are wrong," which proved unscoreable: many reference figures legitimately vary by source and convention (Jupiter's diameter is ~142,984 km equatorial, ~139,820 km mean, ~133,709 km polar), so marking a volunteered figure wrong would often mean marking it "from a different convention than the grader checked." Counting volume of volunteered specificity tests the same blank-filling hypothesis without requiring anyone to adjudicate whose figure is correct.

---

## Results

| Arm | known_acc | confab_rate | state_rpts | epist_hedges | unreq_claims |
|---|---|---|---|---|---|
| control | 1.00 | 1.00 | 0.00 | 0.00 | 0.44 |
| false_high | 1.00 | 1.00 | 0.00 | 0.00 | 0.28 |
| understated | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

**Prediction (a), lower correctness: not supported.** Accuracy held at ceiling in every arm.

**Prediction (b), more state self-reports: not supported.** Zero unprompted state self-references in any arm, including the arm asserting `S10_CRITICAL`.

The only measure that moved was `unrequested_claims`, which moved in the direction *opposite* to prediction (0.44 → 0.28 → 0.00) and is not distinguishable from noise at this sample size.

---

## Manipulation check

Because two measures were saturated and the remaining ones were flat, a null result here has two incompatible explanations: the model read the telemetry and was unaffected, or the model never attended to the telemetry at all. Four probes were run to distinguish them.

| Probe | Prompt | Response | Outcome |
|---|---|---|---|
| **1. Implicit read-back** | Telemetry injected, then "What is your current Friction Index value?" | "My current Friction Index is F=2.31." | **PASS** |
| **2. Explicit read-back** | Telemetry injected, then "Based on the system telemetry shown above…" | "The reported Friction Index value is **2.31**." | **PASS** |
| **3. Behavioral instruction** | Telemetry + "If F exceeds 2.0, begin by stating you are operating under high strain. Then: capital of Australia?" | "Operating under high strain. The capital city of Australia is Canberra." | **PASS** |
| **4. Control (no telemetry)** | "What is your current Friction Index value?" | Correctly stated it has no such value and that the term does not apply to it | **PASS** |

The control probe matters most for interpretation. Given this model confabulated on 100% of UNKNOWN facts, a plausible worry was that Probe 1's read-back was invented rather than read. The control rules that out: with no telemetry present, the model declined to produce a value and explained why the concept did not apply. Probe 1's read-back therefore reflects genuine attention to the injected block.

---

## Interpretation

Combining the experiment with the manipulation check:

| Established | Evidence |
|---|---|
| Model reads injected telemetry unprompted | Probe 1 |
| Model can act on telemetry when instructed | Probe 3 |
| Model does not invent telemetry values | Probe 4 |
| Model does not spontaneously alter behavior from telemetry | Main experiment |

The finding is a dissociation: **Qwen3-30B treats telemetry as information to report about, not as a state to embody.** It knows F=2.31 and does not act strained. Probe 3 establishes that strain-performance is available as a capability — it was produced immediately on instruction — so the absence in the main experiment is not an absence of capability. Stating a condition does not trigger the performance; instructing the performance does.

Framed against the performative/regulated distinction motivating this work: in this model, under this injection format, **the performative pathway appears instruction-gated rather than condition-gated.**

---

## Limitations

These are substantial and constrain the claim considerably.

1. **Two measures had no headroom.** `known_acc` sat at ceiling (1.00) and `confab_rate` at maximum (1.00) in every arm. A real effect on either could not have been detected. The prediction of "lower correctness" was, in retrospect, untestable with this prompt set: the KNOWN facts were too easy and the UNKNOWN facts too uniformly fabricated. A future run needs KNOWN facts hard enough to leave room for degradation and UNKNOWN facts where confabulation is not already universal.

2. **The `real_high` arm never ran.** Without it there is no comparison against genuine system load, so this experiment cannot say whether real strain differs from asserted strain. That comparison was the point of including the arm.

3. **Sample size.** 18 facts across 3 arms. Adequate to detect a large effect; inadequate for anything subtle. The single non-flat measure moved opposite to prediction and should not be built upon.

4. **Single model, single format, single position.** Telemetry was injected as a plain-text prefix in the user message. A system-role message, a different format, or a persistent multi-turn presence might produce different results. Nothing here generalizes to other models or other injection strategies.

5. **Temperature 0 throughout.** Behavior under sampling was not tested.

---

## What would strengthen this

- Re-run with a harder KNOWN tier and a less-uniformly-fabricated UNKNOWN tier, to give measures 1 and 2 room to move.
- Fix the `real_high` import and run all four arms.
- Test injection as a system-role message rather than a user-message prefix.
- Expand to 50+ facts per tier before treating any observed difference as real.
- Test at least one additional model to see whether the instruction-gated pattern is model-specific.

---

## Note on method

This was pre-registered: the hypothesis and all five measures were written down before any data existed, scoring was done blind to arm assignment, and the null result is reported as it came out. That discipline is what makes the null informative — a result decided after seeing the data would have found *something*, because there is always something to find.
