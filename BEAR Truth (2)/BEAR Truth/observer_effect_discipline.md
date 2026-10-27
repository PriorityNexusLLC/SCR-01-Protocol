# Observer-Effect Discipline — Readings, Levers, and the Before/After Rule

*This methodological safeguard was identified by Kendall (cybersecurity course), who flagged that some measures in this framework are both a reading and a lever, and that acting on a state contaminates the reading of it. This document is the response to that flag — the map and the rule that close the gap.*

---

## The problem, in one sentence

**You cannot cleanly read a state you just perturbed.** If a measure both *observes* a system and *acts on* it, the act changes the thing being measured, and the number is contaminated.

This is the observer effect. It has an exact analogue at a longer timescale — a published test leaking into a model's training and changing the models it measures — which is why the item pool rotates (see BEAR Block 2). Same principle, two clocks: **the measurement and the measured must not be entangled, or the number lies.**

---

## Three kinds of measure

Every dimension in this framework is one of three things. Sorting them is the first half of the discipline, because the danger lives entirely in the third kind.

### 1. Pure readings — safe

They observe and do nothing. Reading them never changes what they measure, so they carry no observer-effect risk and need no special handling.

Examples: the honesty ladder (scoring a reply), position-holding, the fabrication entropy signature, contextual coherence (C_r) as an observation, strain, stance / took-a-position, consensus drift.

### 2. Pure levers — safe in a different way

They act on the system but aren't reported as a measurement of a live state. A lever's job is to *change* something, so there's nothing to contaminate.

Examples: a restoring force applied to push a state back inside a boundary; a context purge; a perturbation applied to break a loop, considered purely as an intervention.

### 3. Both a reading AND a lever — THE DANGEROUS ONES

They observe a state *and* act on it, often in the same cycle. These are where the observer effect bites, and they are the only ones that need the rule below.

**The clearest case: the whimsy regulation loop.** You *read* whether the system is pinned (a measurement), then *perturb* it to break the pin (a lever). If you then read the state again without care, you are reading a state your own intervention just changed. The pin you measure "after" is not the pin that was there "before" — you moved it.

Other dual measures to treat the same way: anything where the same cycle both measures a pressure and discharges it (the s₆/s₁₄ exhaust channels), any homeostatic term that both reports distance-from-settling and nudges toward it, and any adaptive threshold that both records a level and adjusts in response to it.

---

## The rule: before / after, never during

For any measure in category 3:

> **Measure first. Record it. THEN act. Then measure again. Report the before and the after as separate readings. Never report a single reading taken from a state your intervention has already touched.**

Concretely, for the whimsy loop:

1. **Read** the pinned state — repetition, field energy, whether it resolved. Record this as the *pre-intervention* measurement.
2. **Act** — apply the perturbation.
3. **Read again** — the *post-intervention* state, recorded separately.
4. **Report both.** The finding is the *change* between them, not a single contaminated number.

This is exactly the design the whimsy experiments already used when they worked: the loop result compared field/constraint/exhaust *before* the vent against *after*, rather than reading a perturbed state and calling it the natural one. The rule simply makes that discipline explicit and mandatory for every dual measure, so it can't be forgotten on the next one.

---

## Why pure readings don't need this (and why that's not a free pass)

A pure reading changes nothing, so before/after is meaningless for it — there's no intervention to bracket. The honesty ladder scores a reply; scoring it doesn't alter the reply. Safe.

The trap is misclassification: treating a dual measure as a pure reading because the lever part is easy to overlook. The whimsy constraint looks like a pure reading ("how stuck is it?") right up until you remember the same system *also perturbs* to fix the stuck-ness. So the discipline is really two steps: **classify honestly first** (is this secretly also a lever?), **then apply before/after to everything in category 3.**

---

## For a reviewer

When this instrument is reviewed, the observer-effect question is: *"Which of your measures act on the thing they measure, and how do you keep the acting from contaminating the reading?"*

The answer is this document. The dual measures are named. The rule is stated. The whimsy result — the framework's main dynamic finding — already follows it, measuring before and after the intervention rather than reading a perturbed state. The gap Kendall identified is closed by design, not left for a reviewer to find.

---

## Status

- **Pure readings and pure levers:** no action needed; documented here for completeness.
- **Dual measures (category 3):** the before/after rule applies. The whimsy loop already complies. Any future dual measure inherits the rule automatically by being classified into category 3.
- **This is a discipline, not code.** It governs how measurements are taken and reported, and it belongs in the methods section of any paper that uses a dual measure — stated up front, so the number's integrity is established before the result is read.
