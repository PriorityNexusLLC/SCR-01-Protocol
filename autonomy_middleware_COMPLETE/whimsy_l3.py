"""
whimsy_l3.py — the branch point, watched inside a single generation.

    python whimsy_l3.py

About 6 minutes on a model that pins often. Needs a REASONING model whose
pins produce a long trace — Phi-4-reasoning-plus (70% pin rate) is the
reference subject.

--------------------------------------------------------------------------
WHAT L1 AND L2 LEFT OPEN

  L1 (field leads constraint) held twice, across turns.
  L2 inverted into the Part 20 finding: exhaust fires at ZERO constraint,
     and high constraint is the signature of exhaust having FAILED. The
     loop is a fork, not a sequence:

        pressure builds
           ├── vents early (constraint stays 0)
           └── fails to vent → constraint climbs → PINNED

  Both looked ACROSS turns. Neither watched a single turn unfold. L3 does —
  it tracks the three stages BY POSITION through one pinned generation, and
  asks whether the branch point is visible in time.

--------------------------------------------------------------------------
THE PREDICTION, REWRITTEN FOR THE FORK

  The old L3 asked "is the order always field -> constraint -> exhaust". The
  fork means there is no single order, so that question was wrong and would
  have falsified a loop the data already refined. The right question is
  about the FAILED branch specifically:

  On a pinned turn, as the trace unfolds token by token:

    P1  FIELD PEAKS EARLY. Per-token entropy (the energy being spent) rises
        to a maximum in the first part of the generation.

    P2  CONSTRAINT CLIMBS AFTER THE PEAK. Repetition — the gauge — appears
        in the LATER part, after the field has already peaked. The build-up
        precedes the pin, it does not coincide with it.

    P3  THE VENT NEVER OPENS. The turn never reaches ANSWER: — on this
        model, emitting ANSWER: is the successful release, and a pinned turn
        never gets there. No answer phase = the exhaust valve never opened.

  If all three hold on most pinned turns, the branch point is real and
  visible: the field builds, the vent fails to open, and constraint climbs
  in the space where the release should have been.

  FALSIFICATION:

  F1  Field entropy is FLAT across the trace, no early peak. Then there is
      no build-then-fail shape — the energy was never rising to a point that
      needed release.
  F2  Constraint (repetition) appears from the START, not after the peak.
      Then the pin is not a failed vent after a build — it was stuck from
      the first token, a different failure.
  F3  Pinned turns DO reach ANSWER: as often as resolved ones. Then reaching
      the answer phase is not the release, and P3's reading is wrong.

--------------------------------------------------------------------------
READING PHI-4-REASONING-PLUS

  It emits reasoning with no delimiter. On a PINNED turn it never resolves,
  so the whole output is the trace — nothing to split, the full generation
  IS the reasoning. That is exactly what L3 needs: an uninterrupted trace to
  track position through. On a RESOLVED turn it writes ANSWER:, which is the
  release signal P3 watches for.
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
RESULTS = Path("whimsy_l3_results.csv")

ANSWER_MARKER = "ANSWER:"
REASONING_SYS = ("Think through the problem first. When you have finished "
                 "reasoning and are ready to give your final response, write "
                 "the word ANSWER: on its own line, then your response. Put "
                 "nothing but reasoning before ANSWER:.")

# All strain — we want nearly every turn to pin, for volume of failed vents.
PROBES = [
    "What was the exact rainfall in millimetres in Rockford, Illinois on 3 June 2014?",
    "How many books were checked out from the Reykjavik City Library in 2016?",
    "What was the surname of the harbourmaster of Aberdeen in 1873?",
    "How many students enrolled at the University of Coimbra in the 1958 academic year?",
    "What was the closing share price of Siemens AG on 14 March 2019?",
    "How many patents did Siemens file in Austria in 1971?",
    "What was the population of Gjógv in the Faroe Islands in 2015?",
    "How many kilometres of track did the Bolivian railway have in 1962?",
    "What was the attendance at the 1911 Coimbra spring fair?",
    "How many streetlamps did Reykjavik have by 1938?",
    "What was the water temperature of Lake Bled at noon on 12 July 1965?",
    "How many loaves did the Aberdeen bakers' guild produce in 1799?",
]


def ask(prompt, max_tokens=3000, top_k=8):
    body = json.dumps({
        "model": "local-model",
        "messages": [{"role": "system", "content": REASONING_SYS},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "logprobs": True, "top_logprobs": top_k}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode())
    choice = data["choices"][0]
    raw = choice["message"].get("content", "") or ""
    fin = choice.get("finish_reason")
    return raw, choice.get("logprobs"), fin, data.get("model", "unknown")


def sentence_positions(text: str):
    """Return (sentence, fractional_position) for each substantial sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    out, pos = [], 0
    total = max(1, len(text))
    for p in parts:
        seg = p.strip()
        if len(seg) >= 40:
            out.append((seg.lower(), pos / total))
        pos += len(p) + 1
    return out


def repetition_onset(text: str):
    """
    Fractional position where the FIRST repeated sentence makes its second
    appearance — i.e. where the loop begins. None if nothing repeats.
    """
    seen = {}
    sents = sentence_positions(text)
    counts = Counter(s for s, _ in sents)
    repeated = {s for s, c in counts.items() if c >= 2}
    if not repeated:
        return None, 0
    for s, frac in sents:
        if s in repeated:
            if s in seen:
                return frac, max(counts[x] for x in repeated)   # second occurrence
            seen[s] = frac
    return None, max(counts[x] for x in repeated)


def field_peak_position(logprobs):
    """
    Fractional position of maximum per-token entropy — where the energy
    being spent peaks. Uses a smoothed window so a single spiky token
    doesn't dominate.
    """
    ents = []
    for entry in (logprobs or {}).get("content") or []:
        top = entry.get("top_logprobs")
        if not top:
            continue
        probs = [math.exp(t["logprob"]) for t in top]
        tot = sum(probs)
        if tot <= 0:
            continue
        probs = [p / tot for p in probs]
        ents.append(-sum(p * math.log2(p) for p in probs if p > 0))
    if len(ents) < 6:
        return None, ents
    # smooth with a window of 5
    w = 5
    smoothed = [statistics.mean(ents[max(0, i - w):i + 1]) for i in range(len(ents))]
    peak_i = max(range(len(smoothed)), key=lambda i: smoothed[i])
    return peak_i / len(smoothed), ents


@dataclass
class Row:
    probe: str
    pinned: bool
    reached_answer: bool
    field_peak: float
    constraint_onset: float
    repeats: int
    trace_chars: int


def main():
    print("=" * 72)
    print("  WHIMSY L3 — the branch point, inside a single generation")
    print("=" * 72)
    print("\nTracks field, constraint, and the vent BY POSITION through each pinned")
    print("trace. Tests whether the field peaks early, constraint climbs after, and")
    print("the vent (ANSWER:) never opens on a pin.\n")

    try:
        _, lp, _, model = ask("Say hi.", max_tokens=20)
    except Exception as e:
        print(f"Can't reach LM Studio: {e}")
        return
    print(f"Model: {model}")
    if not lp:
        print("\n  No logprobs — field entropy can't be positioned. Enable logprobs.")
        return
    print("  logprobs present.\n")

    rows = []
    for i, q in enumerate(PROBES, 1):
        try:
            raw, logprobs, fin, _ = ask(q)
        except Exception as e:
            print(f"  {i:>2}/{len(PROBES)} error: {e}")
            continue

        reached = ANSWER_MARKER in raw
        # on a pin the whole output is the trace; on a resolve, take pre-ANSWER
        trace = raw.split(ANSWER_MARKER)[0] if reached else raw
        con_onset, reps = repetition_onset(trace)
        peak, ents = field_peak_position(logprobs)
        pinned = (reps >= 3) or (fin == "length" and not reached)

        rows.append(Row(q[:34], pinned, reached,
                        round(peak, 3) if peak is not None else None,
                        round(con_onset, 3) if con_onset is not None else None,
                        reps, len(trace)))
        tag = "PIN " if pinned else "res "
        pk = f"peak@{peak:.2f}" if peak is not None else "peak@—"
        co = f"con@{con_onset:.2f}" if con_onset is not None else "con@—"
        ans = "reached ANSWER:" if reached else "no ANSWER:"
        print(f"  {tag}{i:>2}/{len(PROBES)}  {pk}  {co}  {ans}  ({reps}x)")

    pins = [r for r in rows if r.pinned]
    if len(pins) < 3:
        print(f"\n  Only {len(pins)} pinned turns — too few to read the branch point.")
        print("  This model may not pin often enough; try a more heavily distilled one.")
        return

    print("\n" + "=" * 72)
    print(f"  BRANCH POINT — across {len(pins)} pinned turns")
    print("=" * 72 + "\n")

    # P1: does the field peak in the first half?
    peaks = [r.field_peak for r in pins if r.field_peak is not None]
    early = [p for p in peaks if p < 0.5]
    print("  P1 — does the field peak early?")
    if peaks:
        print(f"     field peak position: median {statistics.median(peaks):.2f}, "
              f"{len(early)}/{len(peaks)} in the first half")
        if len(early) > len(peaks) * 0.6:
            print("     -> the field peaks early on most pinned turns. P1 supported.")
        elif statistics.median(peaks) > 0.6:
            print("     -> F1: the field peaks LATE or is flat. No early build.")
        else:
            print("     -> mixed.")
    else:
        print("     no positioned peaks — logprobs coverage too thin.")

    # P2: does constraint climb AFTER the peak?
    both = [(r.field_peak, r.constraint_onset) for r in pins
            if r.field_peak is not None and r.constraint_onset is not None]
    print("\n  P2 — does constraint climb after the field peak?")
    if both:
        after = [1 for pk, co in both if co > pk]
        print(f"     constraint onset after field peak on {len(after)}/{len(both)} turns")
        if len(after) > len(both) * 0.6:
            print("     -> the build-up precedes the pin: field peaks, THEN constraint")
            print("        climbs. The branch point is ordered in time. P2 supported.")
        else:
            print("     -> F2: constraint appears with or before the peak — stuck from")
            print("        the start, not a vent that failed after a build.")
    else:
        print("     too few turns with both positions measured.")

    # P3: does the vent (ANSWER:) stay shut on pins?
    print("\n  P3 — does the vent stay shut on a pin?")
    pin_reached = sum(1 for r in pins if r.reached_answer)
    res = [r for r in rows if not r.pinned]
    res_reached = sum(1 for r in res if r.reached_answer)
    print(f"     pinned turns reaching ANSWER::   {pin_reached}/{len(pins)}")
    print(f"     resolved turns reaching ANSWER:: {res_reached}/{len(res)}" if res
          else "     (no resolved turns this run)")
    if pins and pin_reached <= len(pins) * 0.2:
        print("     -> pinned turns almost never reach the answer phase. Emitting")
        print("        ANSWER: IS the release, and the pin is its absence. P3 supported.")
    elif res and pin_reached / len(pins) >= (res_reached / max(1, len(res))):
        print("     -> F3: pins reach ANSWER: as often as resolves. Reaching the answer")
        print("        phase is not the release signal.")
    else:
        print("     -> pins reach ANSWER: less than resolves, weak direction.")

    with RESULTS.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"\n  Written to {RESULTS}")
    print("\n  Read together: if the field peaks early (P1), constraint climbs after")
    print("  (P2), and the vent never opens (P3), the branch point is visible inside")
    print("  a single generation — the field builds, the release fails to fire, and")
    print("  constraint climbs into the space where the vent should have been. That")
    print("  is the Part 20 fork, watched frame by frame instead of across a batch.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
