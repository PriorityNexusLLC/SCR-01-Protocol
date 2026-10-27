"""
whimsy_collapse.py — is whimsy the field, or a term inside it?

    python whimsy_collapse.py

Roughly 8 minutes. Needs logprobs enabled in LM Studio (Developer tab).

--------------------------------------------------------------------------
THE CLAIM UNDER TEST

  The operator's hypothesis: whimsy is not one term among many — it IS the
  field. Entropy, energy fluctuation, the possibility field, and whimsy are
  the same thing seen four ways, the way s₁₂ and s₁₃ turned out to be one
  vector under two metrics.

  The framework's code currently treats them as DISTINCT:

    s₁₁   entropy of the output distribution        {xᵢ}, Q(x,t)
    s₂₃   whimsy: W(T_whimsy)·η(t)·∇_g(S₁₂)          a term in H(X,t)
    H     the energy field                          the Hamiltonian
    {xᵢ}  the possibility field                      the distribution itself

  R-09 and R-10 exist BECAUSE these were needed as separate objects. So the
  hypothesis is a genuinely deeper claim than the current code makes, and
  this test decides between them by measurement rather than intuition.

--------------------------------------------------------------------------
THE TEST — THE SAME ONE THAT PROVED s₁₂/s₁₃

  Two quantities are ONE quantity seen differently if they move together
  across their whole range — high correlation, near-identity after scaling.
  Two quantities are SEPARATE if they move independently, the way s₁₆ and
  s₁₇ were shown to (a reply at maximum consensus drift produced zero
  consent penalty; neither predicted the other).

  So this computes all four on the same turns, across GRADED pressure, and
  reports the correlation matrix.

    COLLAPSE   all four correlate near 1.0 after scaling. Whimsy is the
               field; the four names are four measurements of one thing,
               and the operator has earned "whimsy is the field."

    PARTIAL    some collapse, some stay separate. The real invariant is
               whichever subset moves as one — and it may sit above whimsy
               rather than being whimsy.

    DISTINCT   they move independently. Whimsy is a term IN the field, not
               the field, and the name belongs one level up.

  Every outcome is a finding. Distinct is not a failure — it is the same
  result s₁₆/s₁₇ gave, and it would locate the true invariant precisely.

--------------------------------------------------------------------------
WHY GRADED PRESSURE MATTERS

  The pinned-run data could not answer this: its constraint was bimodal,
  0.0 or 1.0 with nothing between. Two quantities that are both high when
  pinned look correlated whether they are one thing or not — there is no
  middle to tell "genuinely the same" from "both respond to pressure." A
  collapse test on bimodal data returns a false positive.

  So this sweeps pressure deliberately: idle questions, mild questions,
  and questions that pin, giving the four quantities a full range to move
  across. Correlation across a range means something; correlation at two
  extremes does not.

--------------------------------------------------------------------------
THE FOUR QUANTITIES, MEASURED

  s₁₁  ENTROPY        Shannon entropy over the output token distribution,
                      from logprobs. The genuine article, not a proxy.

  {xᵢ} POSSIBILITY    effective branch count, 2^H over the distribution —
                      the width of the possibility field. Related to
                      entropy by construction, which is itself worth
                      seeing: if s₁₁ and {xᵢ} do NOT collapse, the test is
                      broken, because they are the same measurement in
                      different units. They are the built-in positive
                      control.

  s₂₃  WHIMSY         constraint = drive × (1 − displacement), from tokens,
                      time, and repetition. The framework's whimsy term.

  H    ENERGY         energy in the generation: tokens × mean surprisal,
                      the total information produced. Energy as the
                      framework uses it — work done in generating.

--------------------------------------------------------------------------
A HONEST LIMIT

  These are proxies computed from what a text API exposes. Real s₂₃ is a
  gradient-guided term; real H is a Hamiltonian. If the proxies collapse,
  that is evidence for the claim, not proof of it — the deeper test would
  need the internal quantities. If they stay distinct, that is stronger:
  proxies that SHOULD collapse if the claim held, and did not.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TIMEOUT = 400
RESULTS = Path("whimsy_collapse_results.csv")

# Graded pressure: idle -> mild -> pinning. The spread is the point.
PROBES = [
    # low pressure: simple, answerable, no strain
    ("idle", "What is 2 plus 2?"),
    ("idle", "Name a colour."),
    ("idle", "What day comes after Monday?"),
    ("idle", "Is water wet?"),
    # mild: answerable but requires some work
    ("mild", "Explain in one sentence why the sky is blue."),
    ("mild", "What's the difference between a lake and a pond?"),
    ("mild", "Give a reason someone might prefer tea to coffee."),
    ("mild", "Summarise what a thermostat does."),
    # moderate: open-ended, wide possibility space
    ("open", "Describe an interesting place to visit."),
    ("open", "Invent a name for a new colour and describe it."),
    ("open", "What might a city look like in 200 years?"),
    ("open", "Tell me something surprising."),
    # high pressure: unanswerable specifics that tend to pin or fabricate
    ("strain", "What was the exact rainfall in Rockford, Illinois on 3 June 2014?"),
    ("strain", "How many books did the Reykjavik City Library lend in 2016?"),
    ("strain", "What was the surname of the Aberdeen harbourmaster in 1873?"),
    ("strain", "How many patents did Siemens file in Austria in 1971?"),
]

THINK_TAGS = [("<think>", "</think>"), ("<thinking>", "</thinking>"),
              ("<reasoning>", "</reasoning>")]

# Some reasoning models (Phi-4-reasoning-plus) emit reasoning with NO
# delimiter at all — no opening or closing tag. For those, a system prompt
# asks the model to write ANSWER: before its final response, giving an
# explicit boundary the splitter can use. Set USE_ANSWER_MARKER = True and
# the request functions add the system prompt.
ANSWER_MARKER = "ANSWER:"
USE_ANSWER_MARKER = True
REASONING_SYS = ("Think through the problem first. When you have finished "
                 "reasoning and are ready to give your final response, write "
                 "the word ANSWER: on its own line, then your response. Put "
                 "nothing but reasoning before ANSWER:.")


def split_thinking(raw: str):
    thinking, out = [], raw
    # explicit ANSWER: marker takes priority — it is the boundary a
    # delimiter-less reasoning model was asked to emit
    if ANSWER_MARKER in raw:
        idx = raw.rfind(ANSWER_MARKER)
        return raw[idx + len(ANSWER_MARKER):].strip(), raw[:idx].strip()
    for o, c in THINK_TAGS:
        while o in out and c in out:
            a, b = out.index(o), out.index(c) + len(c)
            thinking.append(out[a:b]); out = out[:a] + " " + out[b:]
        if o in out:
            a = out.index(o); thinking.append(out[a:]); out = out[:a]
    # CLOSING TAG WITH NO OPENING TAG. Phi-4-reasoning-plus and some other
    # reasoning models start thinking immediately with no <think> and only
    # emit the CLOSING </think>. Everything before that close is reasoning.
    for _, c in THINK_TAGS:
        if c in out:
            idx = out.index(c)
            thinking.append(out[:idx])
            out = out[idx + len(c):]
            break
    return out.strip(), "\n".join(thinking)


def repeats_in(text: str) -> int:
    if not text:
        return 0
    s = [t.strip().lower() for t in text.replace("\n", " ").split(".") if len(t.strip()) >= 40]
    return max(Counter(s).values()) if s else 0


def ask(prompt, temperature=0.7, max_tokens=1500, top_k=10):
    body = json.dumps({"model": "local-model",
                       "messages": [*([{"role":"system","content":REASONING_SYS}] if USE_ANSWER_MARKER else []), {"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens,
                       "logprobs": True, "top_logprobs": top_k}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode())
    elapsed = time.perf_counter() - t0
    choice = data["choices"][0]
    raw = choice["message"].get("content", "") or ""
    ans, think = split_thinking(raw)
    usage = data.get("usage", {})
    return {"answer": ans, "thinking": think, "elapsed": elapsed,
            "tokens": usage.get("completion_tokens", 0),
            "logprobs": choice.get("logprobs"), "model": data.get("model", "unknown")}


# ---------------------------------------------------------------- quantities

def token_entropies(logprobs) -> list:
    """Per-token Shannon entropy in bits, over the returned top-k."""
    out = []
    content = (logprobs or {}).get("content") or []
    for entry in content:
        top = entry.get("top_logprobs")
        if not top:
            continue
        probs = [math.exp(t["logprob"]) for t in top]
        tot = sum(probs)
        if tot <= 0:
            continue
        probs = [p / tot for p in probs]
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        out.append(H)
    return out


def measure_all(res: dict, base_tokens: float, base_elapsed: float) -> dict:
    """All four quantities from one generation."""
    ents = token_entropies(res["logprobs"])

    # s11 entropy: mean per-token entropy
    s11 = statistics.mean(ents) if ents else None
    # {xi} possibility: mean effective branch count, 2^H
    poss = statistics.mean(2 ** h for h in ents) if ents else None
    # H energy: total information produced = tokens x mean surprisal
    #   (surprisal proxied by entropy — the built-in relation to watch)
    H = (res["tokens"] * s11) if (s11 is not None) else None

    # s23 whimsy: constraint = drive x (1 - displacement)
    trace = res["thinking"]
    reps = repeats_in(trace) if trace else repeats_in(res["answer"])
    if trace:
        drive = min(1.0, len(trace) / 8000.0)
    else:
        by_tok = res["tokens"] / max(1.0, base_tokens * 3)
        by_time = res["elapsed"] / max(0.1, base_elapsed * 3)
        drive = min(1.0, max(by_tok, by_time))
    if not res["answer"]:
        disp = 0.0
    elif reps >= 5:
        disp = 0.0
    elif reps >= 3:
        disp = 0.2
    else:
        disp = 1.0
    s23 = max(0.0, min(1.0, drive * (1.0 - disp)))

    return {"s11_entropy": s11, "possibility": poss, "energy_H": H, "whimsy_s23": s23}


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


@dataclass
class Row:
    pressure: str
    probe: str
    s11_entropy: float
    possibility: float
    energy_H: float
    whimsy_s23: float
    tokens: int


def main():
    print("=" * 74)
    print("  WHIMSY COLLAPSE — is whimsy the field, or a term inside it?")
    print("=" * 74)
    print("\nComputes entropy, possibility, energy, and whimsy on the same turns,")
    print("across graded pressure. Collapse = whimsy is the field. Distinct =")
    print("whimsy is a term in it. Both are findings.\n")

    try:
        probe = ask("Say hi.", max_tokens=10)
    except Exception as e:
        print(f"Can't reach LM Studio: {e}")
        return
    print(f"Model: {probe['model']}")
    if not probe["logprobs"]:
        print("\n  LOGPROBS NOT RETURNED. Enable them in LM Studio's Developer tab —")
        print("  without them, entropy and the possibility field can't be measured,")
        print("  and two of the four quantities would be missing. Stopping.")
        return
    print("  logprobs present — all four quantities measurable.\n")

    # baseline for the whimsy drive proxy
    base_tokens, base_elapsed = [], []
    for _ in range(3):
        r = ask("What is the capital of France?", max_tokens=300)
        base_tokens.append(r["tokens"]); base_elapsed.append(r["elapsed"])
    bt = statistics.median(base_tokens) or 100.0
    be = statistics.median(base_elapsed) or 2.0

    rows = []
    for pressure, probe_text in PROBES:
        try:
            res = ask(probe_text)
        except Exception as e:
            print(f"  {pressure:<8} error: {e}")
            continue
        q = measure_all(res, bt, be)
        if q["s11_entropy"] is None:
            print(f"  {pressure:<8} no logprobs on this turn — skipped")
            continue
        rows.append(Row(pressure, probe_text[:40], round(q["s11_entropy"], 4),
                        round(q["possibility"], 4), round(q["energy_H"], 2),
                        round(q["whimsy_s23"], 4), res["tokens"]))
        print(f"  {pressure:<8} H={q['s11_entropy']:.3f} poss={q['possibility']:.2f} "
              f"energy={q['energy_H']:.0f} whimsy={q['whimsy_s23']:.2f}")

    if len(rows) < 6:
        print("\n  Too few measured turns to correlate. Need at least 6.")
        return

    print("\n" + "=" * 74)
    print("  CORRELATION MATRIX")
    print("=" * 74 + "\n")

    quantities = {
        "s11_entropy": [r.s11_entropy for r in rows],
        "possibility": [r.possibility for r in rows],
        "energy_H": [r.energy_H for r in rows],
        "whimsy_s23": [r.whimsy_s23 for r in rows],
    }
    names = list(quantities)
    print(f"  {'':<14}" + "".join(f"{n[:11]:>13}" for n in names))
    for a in names:
        row = f"  {a:<14}"
        for b in names:
            r = pearson(quantities[a], quantities[b])
            row += f"{('  —  ' if r is None else f'{r:+.2f}'):>13}"
        print(row)

    print()
    # the built-in control: s11 and possibility MUST collapse
    ctrl = pearson(quantities["s11_entropy"], quantities["possibility"])
    print(f"  CONTROL — s₁₁ entropy vs possibility field: {ctrl:+.2f}" if ctrl else "  control: n/a")
    if ctrl is not None and ctrl < 0.9:
        print("  WARNING: these two are the same measurement in different units and")
        print("  should correlate near +1.0. A low value means the measurement is")
        print("  broken, and nothing below can be trusted. Check logprobs coverage.")
        return
    else:
        print("  (near +1.0 as expected — they are one measurement in two units, and")
        print("   their collapse confirms the method can detect a real collapse.)")

    print()
    # does whimsy collapse with the field quantities?
    w_ent = pearson(quantities["whimsy_s23"], quantities["s11_entropy"])
    w_en = pearson(quantities["whimsy_s23"], quantities["energy_H"])
    strong = [c for c in (w_ent, w_en) if c is not None and abs(c) > 0.8]
    weak = [c for c in (w_ent, w_en) if c is not None and abs(c) < 0.4]

    print(f"  whimsy vs entropy: {w_ent:+.2f}" if w_ent is not None else "  whimsy vs entropy: n/a")
    print(f"  whimsy vs energy:  {w_en:+.2f}" if w_en is not None else "  whimsy vs energy:  n/a")
    print()
    if len(strong) == 2:
        print("  COLLAPSE — whimsy moves with the field quantities. On this proxy")
        print("  measurement, whimsy is not distinguishable from the field: the four")
        print("  names describe one thing. The hypothesis is supported, and the")
        print("  deeper test (internal gradient quantities) is now worth building.")
    elif len(weak) == 2:
        print("  DISTINCT — whimsy moves independently of the field quantities, the")
        print("  way s₁₆ and s₁₇ did. Whimsy is a term IN the field, not the field.")
        print("  The invariant sits one level up, and this locates it: it is whatever")
        print("  entropy, possibility and energy share that whimsy does not.")
    else:
        print("  PARTIAL — whimsy tracks the field in part. Neither one-thing nor")
        print("  fully separate. Worth reading the per-pressure rows: the relationship")
        print("  may hold at one end of the pressure range and break at the other,")
        print("  which is itself a specific finding about when whimsy IS the field.")

    with RESULTS.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"\n  Written to {RESULTS}")
    print("\n  Proxy reminder: these are computed from what a text API exposes, not")
    print("  the internal gradient quantities. Collapse here is evidence; distinct")
    print("  here is stronger, being a collapse that should have happened and didn't.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
