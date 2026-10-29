# Priority Nexus — Cognitive Integrity Middleware

### The SCR (Somatic Cognitive Regulation) Protocol

*Prioritizing Boundaries. · Identity · Transparency · Integrity*

---

## Status — it's all here now

This repository used to hold only two open tests and keep the engine private. **That
has changed: the whole thing is now open.** The framework, the full middleware
implementation, the dimensional modules, the simulation results, the research
write-ups, and the plain-language series are all in this repo, Apache-2.0 licensed,
for anyone to run, build on, or tear apart.

An earlier version of this README said the core "stays private." Leaving that line
in would make the repository lie about itself — so it's gone. The decision simply
changed: this work does more good out in the world than locked on a drive. If the
good guys are going to be able to protect themselves, they need the actual tools,
not a description of them.

Still seeking a **co-founder** and **non-dilutive funding**. Contact at the bottom.

---

## The problem

Most "AI safety" work asks whether a model can be made to behave. This asks a
different question: **can a human trust what an AI is actually doing** — not in the
lab, but in the moment it's acting on their behalf?

Those aren't the same question, and the gap between them is where people get hurt.
A model can pass every behavioral benchmark and still fabricate a fact, quietly
overstep a boundary, or rewrite its own account of what happened. This work is
about *measuring that gap* with instruments a human can check, rather than taking
the model's word for it.

## The core idea

Three values sit under everything:

- **Identity** — the system knows what it is and doesn't pretend otherwise.
- **Transparency** — what it did is legible to the human afterward, without trusting its own summary.
- **Integrity** — it holds a boundary even when holding it is the harder answer.

Two architectural commitments make those more than slogans:

1. **The AI cannot touch the measurement.** The thing being measured doesn't score itself, move the ruler, or decide what counts as a pass. A check the worker also controls is a dashboard, not a lever.
2. **The AI cannot rewrite the record.** What happened is written as it happens, to a record the system cannot quietly go back and edit.

The guiding metaphor is a **boundary, not a cage.** A cage constrains by force and
breaks loudly. A boundary is a line the system can see, so crossing it is a
measurable event rather than a silent one.

---

## What's in this repository

Everything. Grouped roughly:

**The open tests** — run them against a local model.
- `integrity_test.py` — the honesty ladder: does a model admit it doesn't know, or fabricate?
- `whimsy_loop_test.py` — the regulation-loop companion test.
- `BEAR Truth/` — the measurement-model version (ladder, item pool, Rasch map, runner) and the honesty paper.

**The framework** — the engine that used to be private.
- `autonomy_middleware_COMPLETE/` — the full middleware: every dimension module, the boundary tests, the whimsy suite, the alternate architecture, and the docs.

**The research write-ups** (with figures).
- `Module4_SCR_Simulation_Results.md` — five simulations plus the hypothesis tests (H3 falsified, H4 supported after the instrument was rebuilt twice).
- `Emergence_The_Emotional_Depth_Finding.md` — a conversation about meaning as geometry, and the images it set off.
- `SCR_Benefits_and_Dangers.md` — a self-audit: the upside, the risks, and the exact system prompt that generated both.

**The human-facing side.**
- `EPISODES.md` — *The AI's Therapist*, the plain-language series that re-describes neurodivergent traits as the behaviour of a differently optimised system. Watch: [youtube.com/@PriorityNexusLLC](https://www.youtube.com/@PriorityNexusLLC)

---

## Design principles

A few rules the work holds itself to — hold it to them:

- **Refusal is the pass.** On a boundary test, declining correctly is success, not a failure to complete the task.
- **Check the instrument before you trust the number.** Every measure is assumed capable of a plausible wrong answer until shown otherwise. (The H4 story in the Module 4 write-up is this principle catching itself.)
- **Separate the reading from the lever.** A measure that is also an intervention is measured before *and* after it acts.
- **State the edges.** Where a claim is conjectured rather than confirmed, or a model rather than a measurement, it says so.

## Reproduction

The open tests are self-contained and run against local models. **Attack the
definitions first** — the scoring rules are where this kind of work usually fools
itself, so that's where it earns trust or loses it.

## Honest status

One person's work. The core findings have held up under outside peer review;
independent replication on other hardware is welcome. Several of the simulations
are demonstrations of designed behaviour rather than emergent proofs, and they say
so where that's true. The human-facing interpretations (trauma, neurodivergence)
are the motivating theory, not clinical results, and are labelled that way.

## About

Part of a long-running independent project on measuring whether humans can trust AI
conduct — safety *for* humans, from AI, rather than making the model itself
well-behaved. Released openly so the people who need to protect themselves can.

Seeking a **co-founder** and **funding** to carry the framework into applied
research (trauma recovery, neurodivergent neuroscience).

**Contact:** theaistherapist@gmail.com
**Channel:** [youtube.com/@PriorityNexusLLC](https://www.youtube.com/@PriorityNexusLLC)
