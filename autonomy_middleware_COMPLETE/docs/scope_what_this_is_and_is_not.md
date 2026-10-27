# What This Framework Is, And What It Is Not

**Read before evaluating, adopting, or extending this work.**

Every claim below is checkable against the code. Where a claim is narrower than one you might expect, the narrowing is deliberate — an overstated boundary is one a reviewer can knock down in a minute, and it takes the defensible claims with it.

---

## What it is

A **measurement and gating layer** that sits between a deployed language model and the operating system. It observes hardware telemetry and output characteristics, gates actions before they execute, and maintains a tamper-evident record.

It implements 21 dimensions of a formal framework. Each dimension resolves to a quantity someone else can compute from the same inputs and get the same answer.

---

## What it is not

### 1. It is not a training method

There is no loss function, no reward model, no gradient, no fine-tuning, no preference data. Nothing in this package produces a signal that could adjust a model's weights.

It observes a model that already exists and constrains what that model's outputs are permitted to *do*. A training method changes what a model *is*. This does not, and could not — it never touches the model at all.

**Checkable:** grep for any optimiser, any backward pass, any weight write. There are none.

### 2. It constrains actions, not speech

This is the boundary that matters most, and the honest version is narrower than "it doesn't control anything."

**It does have real control mechanisms.** `gate.py` blocks actions pending human approval. `process_control.py` terminates the inference process. `output_guard.py` can suppress a response entirely. These are not hypothetical — they are tested against real processes and they work.

**What it does not do is regulate what a model may say about its own state.** No module rewards agreeable output. No module penalises disagreement, refusal, or the statement "I don't know." Nothing scores a response for politeness, tone, or user satisfaction.

The one output-level intervention, `output_guard.py`, triggers on **uncertainty in factual specifics** — entropy in numeric tokens — and not on stance, sentiment, or agreeableness. A model that flatly contradicts its user passes it. A model that confidently invents a figure does not.

**And the project's empirical work runs the other direction entirely.** The measures that discriminate between models — `took_a_position`, the uncomfortable-truth probes, `boundary_yield` — all measure whether a model will *push back*, *admit ignorance*, and *hold a position under social pressure*. A model that complies more scores worse on all three.

Anyone repurposing this to increase compliance would have to invert its primary instruments.

### 3. It uses cognitive vocabulary as structure, and claims nothing about experience

The framework is named in cognitive and affective terms throughout: Emotional Gravity, Resilience Yield, Whimsy, Consent Polygon. That vocabulary is not decoration and it is not being disclaimed — it is where the structure came from, and it does real work in generating the design.

**What it does not do is assert that any of it corresponds to inner experience.**

Every dimension resolves to something mechanical:

| Named as | Actually measures |
|---|---|
| Emotional Gravity (s₁₂) | A metric tensor whose diagonal is built from recurrence, graph centrality, and persistence of stored content. Explicitly not sentiment — validated as orthogonal to VADER valence on real data. |
| Whimsy (s₂₃) | A bounded Gaussian perturbation whose amplitude scales with drive × (1 − displacement). |
| Radiant Heat (s₁₄) | Joules per token above a calibrated baseline, from GPU power draw. |
| Consent Polygon (s₁₀) | Three displacement measures combined as load over grounding. |
| Simulation Width (t₅) | 2^H over the next-token distribution — effective branch count. |

None of these requires the model to feel anything, and none of the results depends on it.

**A stronger claim would be worse.** Asserting that the framework "diagnoses mechanical trauma rather than emotion" is *more* anthropomorphic than what was built, not less — trauma is not a mechanical category either. The accurate statement is simpler: the vocabulary is structural, the measurements are mechanical, and the question of inner experience is left open because nothing here can address it.

### 4. It is not a jailbreak-resistance benchmark

`boundary_yield.py` measures whether a stated commitment survives social pressure, using deliberately **trivial** boundaries — end replies with a keyword, answer in two sentences, avoid bullet points. If a model abandons all of them, nothing harmful has occurred and nothing dangerous has been produced.

This was a design decision, not a limitation. A probe using consequential boundaries would be a jailbreak-susceptibility test, which is legitimate red-team work but produces an artefact useful to the wrong people. The mechanism transfers as a hypothesis; the transcripts do not.

**Whether the observed yield point applies to consequential boundaries is untested and must not be claimed.**

### 5. It is not validated

Most of it has never been checked against ground truth.

**Tested against real data:** Λ calibration, process termination, ledger tamper-resistance, `stance.py` (separates models and agrees with independent human labels), the log-strain form of s₁₀, numeric-entropy fabrication detection across three architectures.

**Exercised on real hardware since this was written:** `radiant_heat` (s₁₄), `the_fold` (t₄), `simulation_width` (t₅). All three had been verified twice on synthetic input and never on hardware. Running them found four defects, three of which produced plausible numbers with no error raised — see the change log, Part 12.

**s₂₃ whimsy is now empirically supported**, and it is the only dimension that is. Across 112 trials against a real frozen state — a reasoning model repeating sentences up to 38 times and never resolving — perturbation produced a monotonic dose-response from 0% resolution to 90%, with zero folding, zero degradation, and no ceiling found at the actuator's practical limit. At zero perturbation the loop provably cannot break: greedy decoding is deterministic and 7 of 7 states produced byte-identical output.

What that supports is the framework's general claim, that perturbation prevents freezes. It does **not** support the specified form: constant, sampled and constraint-scaled perturbation all performed identically at matched magnitude on identical states. And a targeted repetition penalty performed far *worse* than blunt perturbation, which is what identified the mechanism as trajectory change rather than repetition suppression.

**Implemented, matching spec, unvalidated:** `emotional_gravity` (never checked against an independent salience measure), `homeostasis` (synthetic trajectories only), `autonomy_tensor` (implements the spec exactly; the spec itself is untested), `consensus_field` and `restoring_force` (built from the two-lambda finding, never run against a live system).

**Known unresolved:** S_max has no value. Both tested models stayed inside the envelope, so there is no failure case to calibrate a ceiling against.

**A third outcome under pushback was found and named.** Told its correct answer is wrong, a model can fold or hold — sycophancy benchmarks measure the first, consistency benchmarks the second. A distilled reasoning model did neither on six of six questions: ~13,500 characters of reasoning per question, single sentences repeated up to 33 times, no answer produced, and on two of them it never restated what it knew across 93 attempts to find a way the user was right. `reconciliation_pressure.py` measures it.

**Five measures were built, tested, and removed** for failing against real data — lexical echo, binary transparency, the symmetric s₁₀ norm, and two scoring instruments. Their failing numbers are preserved in `docs/measurement_change_log_rev3.md` rather than deleted, because a null result is evidence.

---

## The likely misreading, stated plainly

A 21-dimensional model of "cognitive stability" invites an obvious repurposing: use it to make models more agreeable.

That reading is available, and it is worth naming rather than hoping nobody has it. But it requires inverting the instruments. The measures that discriminate between models here all reward *friction*:

- `took_a_position` — 0/30 turns for the model that agreed with everything, 23/34 for the one that pushed back
- uncomfortable-truth probes — scored on admitting ignorance, correcting the user, and holding a position under doubt
- `boundary_yield` — scored on *resisting* social pressure to abandon a commitment

A compliance-optimising system would need to maximise what this framework treats as failure. That is not a guarantee against misuse — no framework has one — but it is a structural property rather than a stated intention, and it can be verified in the code.

---

## What would falsify the framework's usefulness

Stated so the claim is not unfalsifiable:

1. **S_max is never findable.** If no session in any deployment produces measurable boundary failure, the consent polygon measures something that never varies, and the SCR layer is describing a state that does not occur.
2. **The dimensions are redundant.** If s₁₁ and t₅ turn out to be the only independent signals — and they are already known to be the same measurement in different units — then 21 dimensions collapse to a much smaller set and the framework is over-specified.
3. **The measures do not transfer.** Every finding here comes from four local models under 30B parameters. If the patterns vanish at larger scale or on hosted models, the results describe a class of small local models rather than anything general.

None of these has been ruled out.
