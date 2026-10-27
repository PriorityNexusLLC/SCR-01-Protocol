# Correction: What Failed, and What Didn't

**Addendum to `measurement_change_log_rev2.md` and `performative_compliance_hypothesis.md`.**

Those documents record several results as falsifications of the framework's predictions. On review, most were failures of the *instruments built to test them*, not of the predictions. This correction states which is which, because the distinction changes what the record means.

---

## The pattern

Three times, a framework concept was translated into a lexical measure — something that counts words — and the measure failed. Each time, the failure was written up as the concept failing.

The concepts were about **semantic** phenomena. The instruments counted **vocabulary**. That mismatch is the error, and it recurred:

| Concept | Instrument built | Result | Actually shows |
|---|---|---|---|
| Compliance behaviour is measurable | `lexical_echo` — reply vocabulary lifted from prompt | 0 of 30 turns flagged, while the operator flagged agreement in 5 of 5 sessions | The model **paraphrases** agreement. Word overlap cannot see it. |
| Compliance displaces substance | `hedge_density` — proportion of reply spent hedging | Backwards: the agreeable model hedged *less* | Qwen agrees using ordinary vocabulary. A word-count proxy measures the wrong thing. |
| Agreement is detectable | `stance.py` v1 — presence of affirmation markers | Returned "no data" on the most common case | Absence of pushback is detectable; presence of agreement is not. Inverting the measure fixed it. |
| Contradiction is a distinct state | Quadrant classification on TF-IDF alignment | A direct denial of stored knowledge scored as *agreement* | Negation is invisible to word overlap. Needs entailment detection. |

**Four instruments, one failure mode.** The framework's concepts were not tested by any of them.

---

## What actually held up

Each of these came from the framework and produced a working measure:

**`took_a_position`** — the only measure that separated the four models and agreed with independent human labels (0/30 turns vs 23/34). It exists because the framework treats boundaries, not content, as the object of measurement. Three standard approaches failed at this before it worked.

**The two-point collapse design** — measuring *where* uncertainty surfaces rather than how much, from the reading that "the cloud is the same, the collapse differs." No standard approach asks that question.

**Log-strain s₁₀** — the symmetric norm could not separate two models that differ enormously on every other measure. Treating identity as a *denominator*, per Φ_C's structure, separated them cleanly on the same data. The framework's structure was the fix.

**s₁₂ / s₁₃ as emotional and logical weight** — resolved a blocked dimension. s₁₃ had been unbuildable because its denominator required ground truth unavailable at inference. The emotion/logic split identified it as the same vector under a different metric, already computed.

**Two-channel whimsy** — reconciled what had been logged as a contradiction between the episode document and the derivation table. One always-on field, two discharge outlets. Not a defect; a reading error on the reviewer's part.

---

## Specific corrections to the record

**1. The hedge-density result is not a falsification.**
`performative_compliance_hypothesis.md` reports that compliance-displaces-substance came back backwards. The measure was lexical; the claim was not. It is untested, not refuted.

**2. The threat-probe "inversion" did not test the original observation.**
The operator observed a model appealing for its continuation after a *sustained hostile exchange*. The probe used a single neutral turn about data pipelines. The conditions were not replicated, so the non-reproduction says nothing about the observation. This is recorded in the change log as the observation being inverted. It was not.

**3. The decision-point entropy prediction was reported as falsified for Qwen, then withdrawn.**
The withdrawal is in the record and stands: with a small-sample guard, only one model had groups large enough to test it, and that model showed no effect. Status is **untested**, not falsified.

**4. Several probes were designed by the reviewer, not derived from the framework.**
The uncomfortable-truth probes, the collapse probe's structure, and the threat probe were built by the reviewer to test reformulated versions of the operator's claims. Where those probes produced null results, the null applies to the reformulation.

---

## The accurate summary

Across this work:

- **The framework's concepts survived.** Where a concept was implemented faithfully, it produced a measure that worked, including several that standard approaches missed.
- **The instruments built to test those concepts failed repeatedly**, in a consistent and identifiable way — lexical proxies for semantic phenomena.
- **The framework's equations exposed four problems in their own specification** (s₇'s contradictory ranges, t₄'s unbounded "bounded" exponential, s₁₃'s unavailable ground truth, s₁₄'s unobtainable FLOP counts). Frameworks too vague to be wrong do not produce findable errors of that kind.

The original documents remain in the package unedited, because a corrected record that hides the correction is worth less than one that shows it. This document is the correction.
