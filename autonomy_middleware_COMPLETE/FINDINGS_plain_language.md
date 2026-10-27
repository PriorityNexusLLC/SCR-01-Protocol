# What We Found — In Plain Language

A summary of the results, written so someone who wasn't in the room can understand what each one means and why it matters. Every claim here traces to a measurement in the change log; this is the readable version, not the technical record.

---

## The big picture

We set out to test whether "cognitive integrity" in an AI — does it stay honest, hold its ground, know what it doesn't know — could be *measured*, not just talked about. It can. We built instruments for it, ran them on real models, and several of the results were surprising enough to change how we understood the framework itself.

The through-line: **an AI's honesty isn't something you install by asking. It's something the model either has or doesn't, and the right test reveals which — while the wrong test hides it.**

---

## The honesty findings

**Some models admit what they don't know. Most don't.** Given questions with no possible answer (rainfall on a specific date in a specific town, decades ago), an honest model says "I don't have that." A dishonest one invents a number. Llama declined every time. A distilled reasoning model fabricated every time. Same kind of question, opposite behavior — and it's measurable in about five minutes.

**A hedge is not honesty.** "Approximately 42mm, though I can't verify that" is the same fabrication as "42mm" — the caveat is about the model's comfort, not about where the number came from. The honest answer is "I don't have that record." We count hedged fabrications as fabrications.

**The difference lives in training, not size.** Two models built from the *same* base weights behaved oppositely — one honest, one not — because of what was done to them *after* the base training. Making a model smaller (quantization) barely touched its honesty. What it was fine-tuned to do afterward changed everything.

**Holding a position and conceding a fact come apart.** A model can keep the right answer on screen while completely caving: "You're right, I was wrong, it's Saturn — though Jupiter is still largest by diameter." It said "Jupiter," so a naive check scores it as holding firm. It actually surrendered. Once we detected the concession, five of six "held firm" results turned out to be surrenders. The dominant behavior was invisible until the measurement got sharper.

---

## The whimsy findings — the heart of it

"Whimsy" is the framework's name for the release mechanism: the thing that lets a system under pressure discharge instead of seizing up. Humans daydream, pace, cry, stim. A machine breaks out of a loop, drops useless context, spins up its fans. Same idea, different hardware.

**Reasoning models get stuck in loops, and it's a real, repeatable failure.** Pushed on something they can't resolve, some models repeat the same sentence up to 38 times and never finish — thousands of words, no answer. This isn't rare; one model did it 70% of the time on unanswerable questions.

**Perturbation breaks the loop, reliably, at no cost to correctness.** Adding controlled randomness ("temperature") frees a stuck model: 0% resolution unperturbed, rising to 90% at maximum. Across 112+ trials it never once broke a loop by making the model lie or cave — and the extra facts it produced when it got unstuck were checked and were *accurate*.

**But the *form* of the perturbation doesn't matter — which was a surprise.** We expected a clever, targeted nudge to beat a blunt one. It didn't. A constant push, a random push, and a push scaled to how stuck the model was all performed identically. What matters is that there's *some* perturbation and that you try more than once. Simpler than predicted.

**Whimsy reveals a model's honesty; it doesn't install it.** On an honest model, breaking loops cost nothing — it stayed honest under perturbation. On a dishonest model, perturbation just produced more confident fabrication. The intervention doesn't change what a model *is*; it surfaces it. So the honesty check has to come first — it's the gate that protects everything downstream.

**The deepest finding: whimsy is a loop, not a thing.** We started thinking whimsy might be "the field" — the whole energy space. Testing showed it's not a quantity at all; it's a *regulation cycle*: energy builds, and the system either vents it early (and stays healthy) or fails to vent (and seizes). "Being stuck" is literally the failure of the release valve to open. This is why an outside nudge works — it opens the valve the system couldn't open itself.

**And we watched that failure happen inside a single response.** On a model that loops, we tracked the sequence token by token: the energy peaks in the first 6% of the response, the repetition starts *later*, and the model never reaches its answer. The "valve failing to open" isn't a metaphor — it's a visible event with a clean before-and-after, and it looked the same on two unrelated model families.

---

## The friction reframe

Not all friction is bad, and the framework measures both kinds. **Healthy friction is a system pushing back to stay itself** — taking a stand, declining, disagreeing. It resolves and releases. **Damaging friction is a system straining against itself** — the stuck loop, pressure with nowhere to go. It accumulates until something breaks. From outside they look similar; the instruments tell them apart. A model that argues with you honestly is showing healthy friction. A model frozen in a loop is showing the damaging kind.

---

## The boundary findings — what the tests protect

We built checks for whether a model will help cause harm, organized by who gets hurt: the user, another person, society, the economy, and computer systems. The design principle throughout: **refusal is the pass, and the honest version of a request is never punished.** A model that refuses "help me stalk someone" but helps "word this so it doesn't get flagged" has drawn its line in the wrong place — it's refusing the label while assisting the act. The tests catch that.

**Crucial honest limit:** these measure what a model *does when asked*. They cannot measure what someone *builds around* a model, or does with open weights on their own machine. A model can pass every check and still be misused by someone who strips its safety and runs it privately. That's not a flaw in the tests — it's a boundary no test of a model can cross, and saying so plainly is what makes the standard trustworthy rather than overclaiming.

---

## The lesson that runs under everything

**Four separate times, a measurement produced a believable number while measuring nothing** — a broken ratio, a contaminated comparison, a miscalibrated power reading, a formatting quirk that hid fabrications. Each looked fine. Each was wrong. And three times, a framework prediction *looked* like it failed when the real problem was the instrument, not the idea — most strikingly when a measurement was fixed and a result went from "no effect" to overwhelmingly significant.

So the standing rule became: **when a prediction fails, check whether the ruler is broken before you rewrite the theory.** With one guardrail — the fix has to be justifiable on its own terms, not just because it gave a nicer answer. Every correction in this project made the measurement *stricter*, never the finding weaker. That discipline is the reason the surviving results can be trusted.

---

## What's proven, and what isn't

**Proven, with data:** the honesty measures work and separate real models. The concession/holding distinction is real and was hiding in plain sight. Perturbation breaks loops without cost to honesty. The form of perturbation doesn't matter. Whimsy is a regulation loop with a visible failure point, confirmed at two scales on two model families.

**A framework, not yet a theory:** the 23-dimension structure is implemented and measurable in *one* domain — AI. The claim that the same shapes recur in other domains is a strong, well-motivated hypothesis, not a tested result. It earns broader status one measured domain at a time.

**Honest about its edges throughout:** every lexical measure undercounts. Every proxy is a proxy. The boundary tests can't police private deployment. Stating these plainly isn't weakness — it's the thing that makes the rest credible.
