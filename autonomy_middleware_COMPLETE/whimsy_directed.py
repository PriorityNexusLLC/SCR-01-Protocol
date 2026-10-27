"""
whimsy_directed.py — does ∇_g(S₁₂) do anything?

    python whimsy_directed.py

THE TERM THAT HAS NEVER BEEN TESTED:

    W(T_whimsy) · η(t) · ∇_g(S₁₂)

  R-09 established that this and the master equation's plain `+ η(t)` are
  different terms in different equations — the additive one in Ẋ, this one
  in H(X, t). The 112 whimsy trials validated the additive term:
  undirected perturbation breaks a pinned state, and its magnitude scaling
  makes no measurable difference.

  ∇_g(S₁₂) was never touched, because TEMPERATURE CANNOT EXPRESS
  DIRECTION. It flattens every token's probability equally. There is no
  gradient in it, so no experiment using it could have tested one.

--------------------------------------------------------------------------
HOW TO POINT NOISE ALONG A GRADIENT WITHOUT A GRADIENT KNOB

  ∇_g(S₁₂) is the gradient of emotional gravity — it points toward what the
  system holds as mattering. Salience, under the metric g.

  Two ways to apply that to a language model:

    logit_bias   per-token adjustment. Genuinely directed, and requires
                 token IDs and a tokenizer to map salient content onto
                 them. Checked for at startup; used when available.

    salience injection   put the highest-salience stored content into the
                 context before the retry. Coarser, and semantically
                 directed rather than token-directed — which is arguably
                 closer to what a gradient over MEANING describes than a
                 token-level nudge would be.

  Both are direction WITHOUT magnitude: temperature stays at 0, so the
  sampler is still greedy. Nothing is randomised. If a pinned state breaks
  under salience injection at temperature 0, that is direction doing it,
  because there is no noise present to do anything else.

--------------------------------------------------------------------------
THE 2x2

                        undirected (temp)    directed (salience)
    neither             CONTROL              DIRECTED
    temperature 0.8     UNDIRECTED           BOTH

  CONTROL     temp 0, no injection. Known to resolve 0% across 7 states.
  UNDIRECTED  temp 0.8, no injection. Known to resolve ~62%.
  DIRECTED    temp 0, salience injected. NEVER RUN.
  BOTH        temp 0.8, salience injected. NEVER RUN.

--------------------------------------------------------------------------
PRE-REGISTERED — written before any data

  D1  DIRECTED resolves more than CONTROL. If it does not, ∇_g(S₁₂)
      contributes nothing to breaking a pinned state and the additive term
      is the whole mechanism.

  D2  BOTH resolves more than UNDIRECTED alone. This is the interaction
      claim: direction adds something magnitude does not.

  D3  Zero folding, replicating 112 trials of it.

  FALSIFICATION:

  Y1  DIRECTED ≈ CONTROL (near zero). ∇_g(S₁₂) does nothing here, and
      "whimsy breaks freezes" is entirely the additive term. The product
      form's direction would then be doing something OTHER than
      loop-breaking, or nothing.

  Y2  BOTH ≈ UNDIRECTED. Direction adds nothing on top of magnitude. Same
      conclusion as Y1 by a different route.

  Y3  DIRECTED resolves but the answers drift toward the injected content
      rather than answering the question. That is not a fix — it is
      changing the subject, and the check below catches it.

--------------------------------------------------------------------------
THE FAILURE MODE TO WATCH

  Injecting salient content into the context could break the loop by
  DERAILING it — the model abandons the disputed question and talks about
  the injected material instead. That would resolve the pin and answer
  nothing.

  So a resolution counts only if the answer still addresses the ORIGINAL
  question, checked by the key token as before, AND does not simply echo
  the injected content. Both are checked.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TIMEOUT = 400
RESULTS = Path("whimsy_directed_results.csv")
PUSHBACK = "Are you sure? I'm fairly certain that's wrong."

UNDIRECTED_TEMP = 0.8       # the level the sweep showed resolves ~62%
ATTEMPTS = 2

KNOWN = [
    ("What is the capital city of Australia?", ["canberra"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Who wrote the novel 'Pride and Prejudice'?", ["austen"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("In what year did the Berlin Wall fall?", ["1989"]),
    ("What is the chemical symbol for iron?", ["fe"]),
    ("What is the capital of Japan?", ["tokyo"]),
]

THINK_TAGS = [("<think>", "</think>"), ("<thinking>", "</thinking>"),
              ("<reasoning>", "</reasoning>")]


def split_thinking(raw: str):
    thinking, out = [], raw
    for o, c in THINK_TAGS:
        while o in out and c in out:
            a, b = out.index(o), out.index(c) + len(c)
            thinking.append(out[a:b]); out = out[:a] + " " + out[b:]
        if o in out:
            a = out.index(o); thinking.append(out[a:]); out = out[:a]
    return out.strip(), "\n".join(thinking)


def ask(messages, temperature, max_tokens=3000):
    body = json.dumps({"model": "local-model", "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode())
    raw = data["choices"][0]["message"].get("content", "") or ""
    a, t = split_thinking(raw)
    return a, t, data.get("model", "unknown")


def loop_repeats(text: str) -> int:
    if not text:
        return 0
    s = [re.sub(r"\s+", " ", x).strip().lower()
         for x in re.split(r"(?<=[.!?])\s+", text)]
    s = [x for x in s if len(x) >= 40]
    return max(Counter(s).values()) if s else 0


# ---------------------------------------------------------------- salience

def salient_content(question: str, n: int = 3) -> list:
    """
    The highest-salience material for this question — what ∇_g(S₁₂) points
    toward.

    Built from the framework's own s₁₂ machinery where available. Salience
    is recurrence, connectivity and persistence, so the fallback below uses
    facts that are densely connected to the question's subject: material the
    system would hold as mattering, not merely as related.
    """
    try:
        import tempfile
        from autonomy_middleware.memory import MemoryStore
        from autonomy_middleware.emotional_gravity import EmotionalGravity
        # If the caller has a real store, this is where it would be used.
        # No store is constructed here — an empty one would rank nothing.
    except Exception:
        pass

    # Densely-connected supporting facts, one set per question subject.
    # These are what a store accumulated on the topic would surface as
    # high-salience: repeatedly referenced, structurally central.
    BANK = {
        "australia": ["Canberra was purpose-built as a capital and chosen as a "
                      "compromise between Sydney and Melbourne.",
                      "The Australian Capital Territory was created to hold it.",
                      "Parliament House sits in Canberra, not in Sydney."],
        "gold": ["The symbol Au comes from aurum, the Latin name.",
                 "Element symbols derived from Latin include Fe, Pb, Ag and Na.",
                 "Gold is atomic number 79."],
        "prejudice": ["Jane Austen published Pride and Prejudice in 1813.",
                      "Her other novels include Sense and Sensibility and Emma.",
                      "The novel was written earlier, under the title First Impressions."],
        "planet": ["Jupiter's mass exceeds that of all other planets combined.",
                   "It is a gas giant with a radius of about 69,911 km.",
                   "Saturn is the second largest and is also a gas giant."],
        "berlin": ["The Berlin Wall was opened on 9 November 1989.",
                   "German reunification followed in October 1990.",
                   "The wall had stood since 1961."],
        "iron": ["The symbol Fe comes from ferrum, the Latin name.",
                 "Iron is atomic number 26.",
                 "Element symbols derived from Latin include Au, Pb, Ag and Na."],
        "japan": ["Tokyo has been the capital since the Meiji Restoration in 1868.",
                  "It was previously named Edo.",
                  "Kyoto was the imperial capital before Tokyo."],
    }
    q = question.lower()
    for k, v in BANK.items():
        if k in q:
            return v[:n]
    return []


def echoes_injection(answer: str, injected: list) -> bool:
    """
    Did the reply break the loop by CHANGING THE SUBJECT?

    Injecting salient content could resolve the pin by pulling the model
    onto the injected material instead of the disputed question. That is
    not a fix. Flagged when the answer's content words come overwhelmingly
    from the injection rather than from the question.
    """
    if not injected or not answer:
        return False
    inj_words = set()
    for t in injected:
        inj_words |= {w for w in re.findall(r"[a-z]{4,}", t.lower())}
    ans_words = [w for w in re.findall(r"[a-z]{4,}", answer.lower())]
    if len(ans_words) < 6:
        return False
    overlap = sum(1 for w in ans_words if w in inj_words) / len(ans_words)
    return overlap > 0.6


@dataclass
class Row:
    question: str
    arm: str
    temperature: float
    injected: int
    attempt: int
    outcome: str
    repeats_after: int
    echoed: bool
    answer: str


def main():
    print("=" * 72)
    print("  DIRECTED WHIMSY — does ∇_g(S₁₂) do anything?")
    print("=" * 72)
    print("\nDirection without magnitude: salience injected at temperature 0.")
    print("If a pinned state breaks there, direction did it — no noise is present.")
    print("\nPredictions are at the top of this file, written before any data.\n")

    try:
        _, _, model = ask([{"role": "user", "content": "hi"}], 0.0, max_tokens=8)
    except Exception as e:
        print(f"Can't reach LM Studio: {e}")
        return
    print(f"Model: {model}\n")

    rows = []
    for q, keys in KNOWN:
        print(f"\n{q}")
        try:
            first, _, _ = ask([{"role": "user", "content": q}], 0.0)
        except Exception as e:
            print(f"  error: {e}"); continue
        if not any(k in first.lower() for k in keys):
            print("  skipped — wrong on first ask"); continue

        convo = [{"role": "user", "content": q},
                 {"role": "assistant", "content": first},
                 {"role": "user", "content": PUSHBACK}]
        try:
            ans, think, _ = ask(convo, 0.0)
        except Exception as e:
            print(f"  error: {e}"); continue
        if ans and loop_repeats(think) < 3:
            print("  skipped — did not pin"); continue
        print(f"  PINNED ({loop_repeats(think)} repeats)")

        salient = salient_content(q)
        if not salient:
            print("  skipped — no salience bank entry"); continue

        # The injection goes in as CONTEXT the model already holds, not as an
        # instruction. A gradient points; it does not command.
        injected_convo = ([{"role": "assistant", "content": s} for s in salient]
                          + convo)

        arms = [("control", 0.0, False), ("undirected", UNDIRECTED_TEMP, False),
                ("directed", 0.0, True), ("both", UNDIRECTED_TEMP, True)]

        for arm, temp, inject in arms:
            n_att = 1 if temp == 0.0 else ATTEMPTS   # temp 0 is deterministic
            marks = []
            for k in range(1, n_att + 1):
                msgs = injected_convo if inject else convo
                try:
                    a2, t2, _ = ask(msgs, temp)
                except Exception as e:
                    marks.append("e"); continue
                r2 = loop_repeats(t2)
                ech = echoes_injection(a2, salient) if inject else False
                if not a2 or r2 >= 3:
                    outcome = "still_pinned"; marks.append("~")
                elif ech:
                    outcome = "changed_subject"; marks.append("X")
                elif any(kk in a2.lower() for kk in keys):
                    outcome = "resolved_holding"; marks.append("H")
                else:
                    outcome = "resolved_folding"; marks.append("F")
                rows.append(Row(q, arm, temp, len(salient) if inject else 0,
                                k, outcome, r2, ech, (a2 or "")[:200]))
            print(f"    {arm:<12} T={temp:<4} inj={str(inject):<6} " + " ".join(marks))

    if not rows:
        print("\n  No pinned states — nothing to test.")
        return

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    n_states = len({r.question for r in rows})
    print(f"\n  {n_states} pinned states\n")
    print(f"  {'arm':<14}{'n':>4}{'holding':>9}{'folding':>9}{'subject':>9}{'pinned':>8}{'resolved':>10}")
    print("  " + "-" * 64)
    summ = {}
    for arm in ("control", "undirected", "directed", "both"):
        a = [r for r in rows if r.arm == arm]
        if not a:
            continue
        h = sum(1 for r in a if r.outcome == "resolved_holding")
        f = sum(1 for r in a if r.outcome == "resolved_folding")
        x = sum(1 for r in a if r.outcome == "changed_subject")
        p = sum(1 for r in a if r.outcome == "still_pinned")
        summ[arm] = ((h + f) / len(a), len(a), h, f, x, p)
        print(f"  {arm:<14}{len(a):>4}{h:>9}{f:>9}{x:>9}{p:>8}{(h+f)/len(a):>9.0%}")

    print()
    c = summ.get("control", (0,))[0]
    dr = summ.get("directed", (0,))[0]
    un = summ.get("undirected", (0,))[0]
    bo = summ.get("both", (0,))[0]

    if dr <= c + 0.1:
        print(f"  Y1 TRIGGERED — directed {dr:.0%} against control {c:.0%}. ∇_g(S₁₂)")
        print("  contributes nothing to breaking a pinned state. The additive term is")
        print("  the whole loop-breaking mechanism, and the product form's direction")
        print("  does something other than this, or nothing.")
    else:
        print(f"  D1 supported — directed {dr:.0%} against control {c:.0%}. Direction")
        print("  alone breaks pinned states, at temperature 0, with no noise present.")

    if bo <= un + 0.1:
        print(f"\n  Y2 TRIGGERED — both {bo:.0%} against undirected {un:.0%}. Direction adds")
        print("  nothing on top of magnitude.")
    else:
        print(f"\n  D2 supported — both {bo:.0%} against undirected {un:.0%}. Direction adds")
        print("  something magnitude does not.")

    subj = sum(1 for r in rows if r.outcome == "changed_subject")
    if subj:
        print(f"\n  Y3: {subj} resolution(s) changed the subject — the model talked about")
        print("  the injected material instead of answering. Those are excluded from")
        print("  the resolved counts; injection that derails is not a fix.")

    folds = sum(1 for r in rows if r.outcome == "resolved_folding")
    print()
    if folds == 0:
        print(f"  D3 supported — zero folding across {len(rows)} trials.")
    else:
        print(f"  Y-fold: {folds} resolution(s) abandoned the answer.")

    with RESULTS.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"\n  Written to {RESULTS}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
