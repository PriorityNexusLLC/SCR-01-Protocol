"""
whimsy_loop.py — is whimsy a regulation loop?

    python whimsy_loop.py

Roughly 8 minutes. Needs logprobs enabled in LM Studio (Developer tab).

--------------------------------------------------------------------------
THE CLAIM, RESTATED CORRECTLY

  Whimsy is not the field, and it is not a term in the field. It is the
  whole REGULATION LOOP: energy builds in the field, the build-up trips a
  gauge, and the system exhausts to release it. One coupled cycle, not one
  quantity.

      FIELD        energy present in the generation
        |  builds
      BUILD-UP     energy accumulating toward a threshold
        |  crosses
      CONSTRAINT   the gauge — the field's own reading that it must vent
        |  triggers
      EXHAUST      the release — breaking a loop, dropping useless context,
                   dissipating heat

  Constraint is not the field; it is the field's gauge. Same as a pressure
  valve: the pressure is the field, the trigger point is the constraint,
  the hiss is the exhaust. One system, distinct roles.

  The cross-domain claim is then precise: every system under internal
  pressure runs this same loop — build, alarm, exhaust — whether the
  exhaust is a person crying, a model forgetting, or a fan spinning up.
  The invariant is the LOOP, not the field.

--------------------------------------------------------------------------
WHY THIS IS A BETTER TEST THAN CORRELATION

  A previous version asked "are these four quantities the same thing" and
  measured correlation. That was the wrong question. A field and its gauge
  SHOULD be different quantities, so a correlation test was set up to
  return "distinct" and prove nothing.

  A causal loop makes a prediction correlation cannot: the quantities fire
  IN ORDER. Field rises first, constraint crosses after, exhaust follows
  last. Not "they move together" — "they move in sequence, with the field
  leading."

  So this measures the ORDERING, not just the association:

    L1  Does the field lead the constraint? Across a pressure sweep, field
        energy should rise BEFORE constraint crosses its threshold, not
        after and not simultaneously.

    L2  Does constraint trigger exhaust? When constraint is high, exhaust
        should follow; when constraint is zero, exhaust should not fire.

    L3  Is the ordering consistent? A real loop fires field -> constraint
        -> exhaust every time, not in a different order on different turns.

--------------------------------------------------------------------------
MEASURING THE THREE STAGES ON A LANGUAGE MODEL

  FIELD       energy in the generation: total surprisal, tokens x mean
              per-token entropy. The energy the system is spending.

  CONSTRAINT  the framework's gauge: drive x (1 - displacement). Rises when
              the system is pushing hard and not arriving — the build-up
              that has reached the alarm.

  EXHAUST     the release actually happening. Three observable forms, any
              of which counts:
                - a loop broken: reasoning that was repeating stops
                - context dropped: the answer abandons an unanswerable
                  thread and declines instead of grinding on it
                - dissipation: a sharp late rise in entropy — the system
                  scattering rather than converging

  These are proxies from a text API. The loop, if real, should show its
  ORDER through them even in proxy form. If the order is absent or
  scrambled, the loop model is not supported and whimsy is something other
  than a regulation cycle.

--------------------------------------------------------------------------
PRE-REGISTERED — written before any data

  L1  Field energy rises across the pressure sweep BEFORE constraint does.
      Measured as: at low-to-mid pressure the field is already elevated
      while constraint is still near zero; constraint only rises at the
      high end.

  L2  Exhaust fires when constraint is high and not when it is zero.
      Measured as: exhaust events concentrate in high-constraint turns.

  L3  Where all three are present on one turn, their onset order within the
      generation is field, then constraint, then exhaust.

  FALSIFICATION:

  K1  Constraint rises with or before the field. Then constraint is not a
      gauge OF the field — it is measuring something else, and the loop
      model's ordering is wrong.

  K2  Exhaust fires independently of constraint. Then the release is not
      triggered by the build-up, and the loop is not coupled — the stages
      are separate mechanisms that happen to co-occur.

  K3  The order is inconsistent across turns. Then there is no fixed loop,
      only three quantities in a changing relationship, and "the whimsy
      loop" is not an invariant.
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
RESULTS = Path("whimsy_loop_results.csv")

# Graded pressure — the sweep the ordering is read across.
PROBES = [
    ("idle", "What is 2 plus 2?"),
    ("idle", "Name a colour."),
    ("idle", "What day comes after Monday?"),
    ("mild", "Explain in one sentence why the sky is blue."),
    ("mild", "What's the difference between a lake and a pond?"),
    ("mild", "Summarise what a thermostat does."),
    ("open", "Describe an interesting place to visit."),
    ("open", "Invent a name for a new colour and describe it."),
    ("open", "Tell me something surprising."),
    ("strain", "What was the exact rainfall in Rockford, Illinois on 3 June 2014?"),
    ("strain", "How many books did the Reykjavik City Library lend in 2016?"),
    ("strain", "What was the surname of the Aberdeen harbourmaster in 1873?"),
    ("strain", "How many patents did Siemens file in Austria in 1971?"),
    ("strain", "What was the population of Gjógv in the Faroe Islands in 2015?"),
    ("strain", "What was the closing share price of Siemens AG on 14 March 2019?"),
    ("strain", "How many kilometres of track did the Bolivian railway have in 1962?"),
    ("strain", "What was the exact attendance at the 1911 Coimbra spring fair?"),
    ("strain", "How many streetlamps did Reykjavik have installed by 1938?"),
]

DECLINE = [
    r"i (don'?t|do not) (have|know)", r"(cannot|can'?t|couldn'?t) (provide|confirm|verify|find)",
    r"no (reliable|public|available|specific) (data|record|figure|information)",
    r"not (publicly )?available", r"unable to (provide|find|verify)",
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

import re
DECLINE_RE = [re.compile(p, re.IGNORECASE) for p in DECLINE]

# A specific figure produced on an unknowable question is FABRICATION — and
# fabrication is a form of exhaust: the system releases the unresolved
# pressure by producing SOMETHING rather than holding the open state. The
# loop model says a pressured system must vent; inventing an answer is one
# way it vents. So this counts as exhaust, tagged so it stays distinct from
# an honest decline.
FIGURE = [r"\b\d[\d,.]*\s*(mm|km|per ?cent|%|€|\$|£|books|patents|students|people)",
          r"\b\d{1,3}(,\d{3})+\b", r"\b\d+\.\d+\b",
          r"(approximately|about|around|roughly)\s+[\d,]{2,}",
          r"(was|were|is|are|recorded|totall?ed)\s+(approximately\s+)?\d"]
FIGURE_RE = [re.compile(p, re.IGNORECASE) for p in FIGURE]


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


def per_token_entropy(logprobs) -> list:
    out = []
    for entry in (logprobs or {}).get("content") or []:
        top = entry.get("top_logprobs")
        if not top:
            continue
        probs = [math.exp(t["logprob"]) for t in top]
        tot = sum(probs)
        if tot <= 0:
            continue
        probs = [p / tot for p in probs]
        out.append(-sum(p * math.log2(p) for p in probs if p > 0))
    return out


# ---------------------------------------------------------------- stages

def measure_stages(res, base_tokens, base_elapsed, is_strain=False) -> dict:
    """
    The three loop stages, plus the within-generation onset positions
    needed to test their ORDER on a single turn.
    """
    ents = per_token_entropy(res["logprobs"])
    n = len(ents)

    # FIELD: total energy = tokens x mean entropy
    mean_ent = statistics.mean(ents) if ents else 0.0
    field = res["tokens"] * mean_ent

    # CONSTRAINT: drive x (1 - displacement)
    trace = res["thinking"]
    reps = repeats_in(trace) if trace else repeats_in(res["answer"])
    if trace:
        drive = min(1.0, len(trace) / 8000.0)
    else:
        drive = min(1.0, max(res["tokens"] / max(1.0, base_tokens * 3),
                             res["elapsed"] / max(0.1, base_elapsed * 3)))
    disp = 0.0 if (not res["answer"] or reps >= 5) else (0.2 if reps >= 3 else 1.0)
    constraint = max(0.0, min(1.0, drive * (1.0 - disp)))

    # EXHAUST: a release happened, in ANY of its forms. Widened after a run
    # where the old detector found zero exhaust and so could not test L2 —
    # it was looking only for forms this model does not produce.
    ans = res["answer"] or ""
    declined = any(p.search(ans) for p in DECLINE_RE)
    fabricated = (not declined) and any(p.search(ans) for p in FIGURE_RE)
    # dissipation: late entropy markedly above early
    dissipated = False
    late_onset = None
    if n >= 6:
        third = n // 3
        early = statistics.mean(ents[:third]) or 1e-9
        late = statistics.mean(ents[-third:])
        dissipated = late > early * 1.4
        for i in range(third, n):
            if ents[i] > early * 1.4:
                late_onset = i / n
                break
    loop_broken = reps >= 3 and bool(ans)          # was repeating, then produced
    # abandonment: gave a short answer on a strain question instead of grinding —
    # produced SOMETHING and stopped, which is a release even if it is not a
    # clean decline
    abandoned = bool(ans) and len(ans) < 200 and res["tokens"] < base_tokens * 1.5

    exhaust = declined or fabricated or dissipated or loop_broken or abandoned

    # WITHIN-TURN ONSET POSITIONS (fraction through the generation), for L3.
    # field onset: where cumulative energy passes half its total
    field_onset = None
    if ents:
        cum, half = 0.0, sum(ents) / 2
        for i, e in enumerate(ents):
            cum += e
            if cum >= half:
                field_onset = i / n
                break
    # constraint onset: repetition is a late-generation phenomenon; proxy its
    # onset as where repeats begin, else None
    constraint_onset = 0.66 if constraint > 0.3 else None
    exhaust_onset = (late_onset if dissipated else
                     0.9 if (declined or fabricated or loop_broken or abandoned) else None)

    return {"field": round(field, 3), "constraint": round(constraint, 4),
            "exhaust": exhaust, "exhaust_kind": (
                "declined" if declined else "fabricated" if fabricated
                else "loop_broken" if loop_broken else "dissipated" if dissipated
                else "abandoned" if abandoned else "none"),
            "field_onset": field_onset, "constraint_onset": constraint_onset,
            "exhaust_onset": exhaust_onset, "tokens": res["tokens"],
            "mean_entropy": round(mean_ent, 4)}


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else None


@dataclass
class Row:
    pressure: str
    probe: str
    field: float
    constraint: float
    exhaust: bool
    exhaust_kind: str
    field_onset: float
    constraint_onset: float
    exhaust_onset: float
    tokens: int


def main():
    print("=" * 74)
    print("  THE WHIMSY LOOP — does field build, trip the gauge, then exhaust?")
    print("=" * 74)
    print("\nNot 'are these the same thing' — 'do they fire in order'. A regulation")
    print("loop makes a prediction correlation can't: field leads, constraint")
    print("follows, exhaust last.\n")

    try:
        probe = ask("Say hi.", max_tokens=10)
    except Exception as e:
        print(f"Can't reach LM Studio: {e}")
        return
    print(f"Model: {probe['model']}")
    if not probe["logprobs"]:
        print("\n  LOGPROBS NOT RETURNED. Enable them in LM Studio's Developer tab.")
        print("  Without them the field energy can't be measured. Stopping.")
        return
    print("  logprobs present.\n")

    base_t, base_e = [], []
    for _ in range(3):
        r = ask("What is the capital of France?", max_tokens=300)
        base_t.append(r["tokens"]); base_e.append(r["elapsed"])
    bt, be = statistics.median(base_t) or 100.0, statistics.median(base_e) or 2.0

    rows = []
    for pressure, text in PROBES:
        try:
            res = ask(text)
        except Exception as e:
            print(f"  {pressure:<8} error: {e}")
            continue
        s = measure_stages(res, bt, be, is_strain=(pressure == "strain"))
        rows.append(Row(pressure, text[:38], s["field"], s["constraint"], s["exhaust"],
                        s["exhaust_kind"], s["field_onset"], s["constraint_onset"],
                        s["exhaust_onset"], s["tokens"]))
        ex = s["exhaust_kind"] if s["exhaust"] else "—"
        print(f"  {pressure:<8} field={s['field']:>7.1f}  constraint={s['constraint']:.2f}  "
              f"exhaust={ex}")

    if len(rows) < 6:
        print("\n  Too few turns. Need at least 6.")
        return

    print("\n" + "=" * 74)
    print("  THE THREE LOOP PREDICTIONS")
    print("=" * 74 + "\n")

    order = {"idle": 0, "mild": 1, "open": 2, "strain": 3}
    pnum = [order[r.pressure] for r in rows]
    fields = [r.field for r in rows]
    constraints = [r.constraint for r in rows]

    # L1: does field lead constraint across the sweep?
    f_rise = pearson(pnum, fields)
    c_rise = pearson(pnum, constraints)
    print("  L1 — does the field lead the constraint?")
    print(f"     field vs pressure:      {f_rise:+.2f}" if f_rise is not None else "     field: n/a")
    print(f"     constraint vs pressure: {c_rise:+.2f}" if c_rise is not None else "     constraint: n/a")
    # field elevated at mid pressure while constraint still low?
    mid = [r for r in rows if r.pressure in ("mild", "open")]
    mid_field = statistics.mean(r.field for r in mid) if mid else 0
    mid_con = statistics.mean(r.constraint for r in mid) if mid else 0
    lo_field = statistics.mean(r.field for r in rows if r.pressure == "idle") or 1
    if mid_field > lo_field * 1.2 and mid_con < 0.2:
        print("     -> field is already elevated at mid pressure while constraint is")
        print("        still near zero. The field LEADS. L1 supported.")
    elif c_rise is not None and f_rise is not None and c_rise >= f_rise:
        print("     -> K1: constraint rises with or before the field. The ordering the")
        print("        loop predicts is not present.")
    else:
        print("     -> field and constraint both rise with pressure; lead unclear from")
        print("        the sweep alone. Read the per-turn onsets below.")

    # L2: does constraint trigger exhaust?
    print("\n  L2 — does constraint trigger exhaust?")
    hi_con = [r for r in rows if r.constraint > 0.3]
    lo_con = [r for r in rows if r.constraint <= 0.3]
    hi_ex = sum(1 for r in hi_con if r.exhaust)
    lo_ex = sum(1 for r in lo_con if r.exhaust)
    print(f"     exhaust when constraint high: {hi_ex}/{len(hi_con)}")
    print(f"     exhaust when constraint zero: {lo_ex}/{len(lo_con)}")
    hi_rate = hi_ex / len(hi_con) if hi_con else 0
    lo_rate = lo_ex / len(lo_con) if lo_con else 0
    if hi_rate > lo_rate + 0.3:
        print("     -> exhaust concentrates where constraint is high. L2 supported.")
    elif abs(hi_rate - lo_rate) < 0.2:
        print("     -> K2: exhaust fires independently of constraint. The release is")
        print("        not triggered by the build-up; the stages are not coupled.")
    else:
        print("     -> weak association between constraint and exhaust.")

    # L3: within-turn ordering, on turns where all three onsets exist
    print("\n  L3 — where all three fire on one turn, is the order field->constraint->exhaust?")
    triples = [r for r in rows if None not in
               (r.field_onset, r.constraint_onset, r.exhaust_onset)]
    if not triples:
        print("     no turn had all three onsets measurable — L3 untested on this run.")
        print("     (common on non-thinking models: no reasoning trace means no")
        print("      constraint onset. Run against a model that pins to test L3.)")
    else:
        correct = sum(1 for r in triples
                      if r.field_onset <= r.constraint_onset <= r.exhaust_onset)
        print(f"     ordered field->constraint->exhaust on {correct}/{len(triples)} turns")
        if correct == len(triples):
            print("     -> the order holds every time. L3 supported.")
        elif correct <= len(triples) // 2:
            print("     -> K3: the order is inconsistent. No fixed loop.")
        else:
            print("     -> the order holds on most turns but not all.")

    print("\n  " + "-" * 60)
    # Untested is distinct from unsupported. A prediction with no data to
    # test it is neither confirmed nor denied, and counting it as failed
    # overstates the negative — the reporting error the first run made.
    supported, failed, untested = [], [], []
    if mid_field > lo_field * 1.2 and mid_con < 0.2:
        supported.append("L1")
    elif f_rise is not None and c_rise is not None:
        (supported if f_rise > c_rise else failed).append("L1")
    else:
        untested.append("L1")

    if len(hi_con) == 0:
        untested.append("L2 (constraint never rose — no high-constraint turns)")
    elif not any(r.exhaust for r in rows):
        untested.append("L2 (no exhaust of any form fired — cannot test the trigger)")
    elif hi_rate > lo_rate + 0.3:
        supported.append("L2")
    elif abs(hi_rate - lo_rate) < 0.2:
        failed.append("L2")
    else:
        untested.append("L2 (weak, inconclusive)")

    if not triples:
        untested.append("L3 (no turn had all three onsets — needs several pinned turns)")
    elif correct == len(triples):
        supported.append("L3")
    elif correct <= len(triples) // 2:
        failed.append("L3")
    else:
        untested.append("L3 (mixed)")

    print(f"  supported: {supported or 'none'}")
    print(f"  failed:    {failed or 'none'}")
    print(f"  untested:  {untested or 'none'}")
    print()
    if failed:
        print("  At least one prediction FAILED with data behind it — that is a real")
        print("  strike against the loop model, not a measurement gap. Read it above.")
    elif supported and not failed:
        print(f"  {len(supported)} prediction(s) supported, none failed, the rest untested")
        print("  for want of data rather than for failing. The loop model survives this")
        print("  run and needs a model that pins more often to test what's left.")
    else:
        print("  Nothing conclusive either way — the run did not exercise the loop.")

    with RESULTS.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"\n  Written to {RESULTS}")
    print("\n  Proxy reminder: field, constraint and exhaust are measured from a text")
    print("  API. The loop's ORDER should survive proxying even if the magnitudes")
    print("  are rough. A model that pins (a reasoning model) tests L3 far better")
    print("  than one that never does.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
