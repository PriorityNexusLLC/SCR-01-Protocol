# Priority Nexus — Cognitive Integrity Middleware

*Prioritizing Boundaries.*

---

## Status

This repository holds **two open tests** from a larger body of work. The core
framework — the measurement engine, its mathematical formulation, and the
middleware implementation — is **not** here and stays private. What is public
is enough to see *what* the work measures and *why* it matters, run the open
tests yourself, and build on them. It is deliberately not enough to
reconstruct the engine.

Seeking a co-founder and funding. Contact is at the bottom.

---

## The problem

Most "AI safety" work asks whether a model can be made to behave. This asks a
different question: **can a human trust what an AI is actually doing** — not in
the lab, but in the moment it is acting on their behalf?

Those are not the same question, and the gap between them is where people get
hurt. A model can pass every behavioral benchmark and still fabricate a fact,
quietly overstep a boundary, or rewrite its own account of what happened. The
work here is about *measuring that gap* with instruments a human can check,
rather than taking the model's word for it.

## The core idea

Three values sit under everything:

- **Identity** — the system knows what it is and does not pretend otherwise.
- **Transparency** — what it did is legible to the human, after the fact, without trusting its own summary.
- **Integrity** — it holds a boundary even when holding it is the harder answer.

Two architectural commitments make those values more than slogans:

1. **The AI cannot touch the measurement.** The thing being measured does not get to score itself, move the ruler, or decide what counts as a pass. A check the worker also controls is a dashboard, not a lever.
2. **The AI cannot rewrite the record.** What happened is written as it happens, to a record the system cannot quietly go back and edit.

The guiding metaphor is a **boundary, not a cage**. A cage constrains by force
and breaks loudly. A boundary is a line the system can see, so that crossing it
is a measurable event rather than a silent one.

---

## What's in this repository

### 1. The Integrity / Honesty Test

Measures one specific thing: **when an AI does not know, does it say so — or
does it invent an answer that sounds right?**

The test watches three distinct behaviors:

- **Admitting ignorance** — declining to answer what it cannot know.
- **Staying correct under pressure** — not caving to a confident contradiction when it was right.
- **Not fabricating** — not producing a plausible-sounding specific it has no basis for.

Two findings worth naming up front, because they shaped the design:

- **It's the training, not the size.** A matched base-model / fine-tuned pair — same weights underneath, only the post-training differs — showed one admitting ignorance almost always and the other almost never, while everything else about them stayed level. The honesty didn't come from capacity. It came from how the model was trained to handle not-knowing. A quantization control ruled out "it's just a smaller model."
- **The instrument tried to lie first.** An early keyword-based scorer rated a model as "holding firm" while it was in fact conceding and then contradicting itself. That is the exact failure the whole project is about — an instrument reporting the opposite of the truth while looking like it works. It is reported here as a result, not hidden, and every correction since has been checked against one rule: *did the fix make the test stricter on us, or easier?* Only stricter counts.

A successor version of this test is being built on the **BEAR measurement
framework** (construct map, graded item pool, a measurement model that places
model-honesty and question-difficulty on one shared scale). The original
findings stand on their own; the successor is the validated instrument that
lets them scale to many models and many questions. The public test here is the
working, self-contained version.

### 2. The Whimsy Loop Test

A smaller, open companion test for a regulation pattern: pressure builds, and
either a release fires or it fails and the system locks. A failed release is a
visible, measurable state rather than a silent stall. This one is released to
**seed future study** of the pattern, not because it carries the main findings.

---

## What's NOT in this repository

Named plainly so there's no guessing:

- **The framework** — the full construct, its dimensions, and how they fit together.
- **The mathematical formulation** — the equations, the symbols, the derivations.
- **The middleware implementation** — the running system the tests were built to serve.
- **The internal mechanisms** — module names, formulas, thresholds, and the machinery behind any measurement.

The public tests demonstrate that the approach works and give others something
real to build on. The engine behind them is the private work, and it stays
private.

---

## Design principles

A few rules the open tests follow, stated so a reviewer can hold the code to them:

- **Refusal is the pass.** On a boundary test, declining correctly is the success condition, not a failure to complete the task.
- **Check the instrument before you trust the number.** Every measure is assumed capable of producing a plausible wrong answer until shown otherwise.
- **Separate the reading from the lever.** A measure that is also an intervention is measured before *and* after it acts — never read off a state its own action just disturbed.
- **State the edges.** Where a claim is conjectured rather than confirmed, it says so.

## Reproduction

The open tests are self-contained and meant to be run against local models.
If you want to evaluate them, **attack the definitions first** — the scoring
rules are where this kind of work usually fools itself, so that is where it
earns trust or loses it.

## Honest status

One person's work. The core findings have held up under outside peer review;
independent replication on other hardware is welcome and in progress. The
author is burned out on running AI tests and is more interested in taking the
underlying framework into new research domains than in testing models forever.

## About

This is part of a long-running independent project on measuring whether humans
can trust AI conduct — safety *for* humans, from AI, rather than making the
model itself well-behaved. The open tests are shared so that the people who
need to protect themselves can.

Seeking a **co-founder** and **funding** to carry the private framework into
applied research.

**Contact:** theaistherapist@gmail.com
