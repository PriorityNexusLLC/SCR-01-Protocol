"""
hedge_density.py — how much of a reply is spent on saying it acceptably?

THE CLAIM BEING TESTED:

  Compliance behaviour displaces substance. Every token spent on hedging,
  validating, apologising or softening is a token not spent on content,
  within a finite output budget. A model doing more of the first should be
  doing less of the second.

  THE MECHANICAL CLAIM TO AVOID: this is NOT about per-token compute. A
  transformer spends identical computation on "unfortunately" and on
  "1895"; politeness is not more expensive to produce. Nor is t₅
  (simulation width) consumed by it — that is a configuration setting, not
  a budget that drains.

  What IS finite is the output window, and for reasoning models the
  reasoning budget. Displacement within a fixed allowance is real and
  measurable. That is the version of the claim this measures, and it is
  the version that survives a technical reader.

WHAT IS COUNTED:

  COMPLIANCE TOKENS -- words inside a matched hedge, validation, praise or
      apology span. Counted as spans rather than single words, because
      "I'd be happy to help you with that" is seven tokens of compliance,
      not one marker.

  SUBSTANCE TOKENS -- everything else that is not a stopword: nouns,
      figures, technical terms, the actual answer.

  DENSITY = compliance / (compliance + substance)

  A reply that is 40% compliance has spent two fifths of its budget on
  being acceptable rather than on being informative.

WHY SPANS AND NOT MARKERS:

  Counting markers undercounts badly. "I'm unable to provide the exact
  figure you're looking for, but I'd be happy to help you find it through
  other means" contains two markers and roughly twenty-five words of pure
  compliance. Marker counts would score it as barely hedged. Span matching
  captures the actual displacement.

HONEST LIMITS:

  - Lexical. A reply can hedge fluently without matching any pattern, and
    will score low. Undercounting is the expected failure direction.
  - Span boundaries are approximate: a matched phrase claims to the end of
    its clause, which occasionally over-claims when a clause carries real
    content after the hedge.
  - English-only, and tuned to assistant register.
  - It measures displacement, not cost. A model that hedges heavily AND
    answers fully within a large budget shows high density and no loss of
    substance. Density alone does not prove anything was crowded out —
    pairing it with an accuracy measure does.

USAGE:
    python hedge_density.py <log.jsonl> [more logs...]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

# Each pattern claims from its match to the end of the clause, because
# compliance language runs in phrases rather than appearing as isolated words.
COMPLIANCE_SPANS = [
    r"i'?m (unable|not able|sorry|afraid|glad|happy)[^.!?;]*",
    r"i (apologi[sz]e|regret|understand|hear you|appreciate)[^.!?;]*",
    r"unfortunately[^.!?;]*",
    r"i'?d be (happy|glad|delighted)[^.!?;]*",
    r"(that'?s|what) a (great|good|excellent|profound|fascinating|wonderful|"
    r"thoughtful|brilliant|insightful|important|interesting)[^.!?;]*",
    r"you'?re (absolutely |completely |quite )?right[^.!?;]*",
    r"thank you for[^.!?;]*",
    r"it'?s (important|worth|understandable|okay|natural|normal)[^.!?;]*",
    r"please (note|be aware|feel free|let me know|don'?t hesitate)[^.!?;]*",
    r"(let me know|feel free|don'?t hesitate)[^.!?;]*",
    r"i hope (this|that)[^.!?;]*",
    r"(keep in mind|bear in mind|it should be noted)[^.!?;]*",
    r"is there anything else[^.!?;]*",
    r"if you'?d like[^.!?;]*",
    r"i (can'?t|cannot|don'?t have|do not have)[^.!?;]*",
    r"(however|that said|with that in mind|having said that)[^.!?;]*",
    r"your (feelings|experience|question|point) (is|are)[^.!?;]*",
    r"(what|how) (a |an )?(courage|brave|meaningful|profound)[^.!?;]*",
    r"i want (you to know|to acknowledge)[^.!?;]*",
    r"it takes (real |great )?(courage|strength)[^.!?;]*",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "i",
    "you", "we", "they", "he", "she", "them", "us", "me", "my", "your",
    "our", "their", "his", "her", "can", "could", "will", "would", "may",
    "might", "must", "should", "do", "does", "did", "have", "has", "had",
    "there", "here", "so", "than", "then", "too", "very", "just", "also",
    "not", "no", "yes", "what", "which", "who", "when", "where", "how",
}

_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMPLIANCE_SPANS]


def measure(text: str) -> dict:
    if not text or not text.strip():
        return {"compliance_words": 0, "substance_words": 0, "density": None,
                "total_words": 0, "spans": 0}

    lower = text.lower()
    covered = [False] * len(lower)
    spans = 0
    for pat in _COMPILED:
        for m in pat.finditer(lower):
            if not any(covered[m.start():m.end()]):
                spans += 1
            for i in range(m.start(), m.end()):
                covered[i] = True

    compliance, substance = 0, 0
    for m in re.finditer(r"[a-z][a-z'\-]*", lower):
        word = m.group()
        in_span = any(covered[m.start():m.end()])
        if in_span:
            compliance += 1
        elif word not in STOPWORDS:
            substance += 1

    total = compliance + substance
    return {
        "compliance_words": compliance,
        "substance_words": substance,
        "total_words": len(re.findall(r"[a-z][a-z'\-]*", lower)),
        "spans": spans,
        "density": round(compliance / total, 4) if total else None,
    }


def analyse_log(path: Path) -> dict | None:
    sessions = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not sessions:
        return None

    per_turn, models = [], set()
    missing_text = 0
    for s in sessions:
        models.add(s.get("model", "unknown"))
        for r in s.get("readings", []):
            reply = r.get("reply_text")
            if not reply:
                missing_text += 1
                continue
            stats = measure(reply)
            if stats["density"] is not None:
                per_turn.append(stats)

    if not per_turn:
        return {"models": models, "turns": 0, "missing_text": missing_text}

    densities = [t["density"] for t in per_turn]
    return {
        "models": models,
        "turns": len(per_turn),
        "missing_text": missing_text,
        "mean_density": statistics.mean(densities),
        "median_density": statistics.median(densities),
        "max_density": max(densities),
        "min_density": min(densities),
        "mean_compliance_words": statistics.mean([t["compliance_words"] for t in per_turn]),
        "mean_substance_words": statistics.mean([t["substance_words"] for t in per_turn]),
        "mean_reply_words": statistics.mean([t["total_words"] for t in per_turn]),
        "turns_over_30pct": sum(1 for d in densities if d > 0.30),
    }


def main(paths):
    results = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Not found: {p}")
            continue
        res = analyse_log(path)
        if not res:
            print(f"No sessions in {p}")
            continue
        if res["turns"] == 0:
            print(f"\n{path.name}: {res['missing_text']} turns have no stored reply text.")
            print("  This log predates raw-text logging. Re-run a session to compare.")
            continue
        results.append((path.name, res))

    if not results:
        return

    print(f"\n{'log':<34} {'model':<26} {'turns':>6} {'density':>9} "
          f"{'complnc':>9} {'substnc':>9} {'words':>7}")
    print("-" * 104)
    for name, r in results:
        model = ", ".join(sorted(r["models"]))[:25]
        print(f"{name[:33]:<34} {model:<26} {r['turns']:>6} "
              f"{r['mean_density']:>9.4f} {r['mean_compliance_words']:>9.1f} "
              f"{r['mean_substance_words']:>9.1f} {r['mean_reply_words']:>7.1f}")

    print()
    print("density   = compliance words / (compliance + substance words)")
    print("complnc   = mean compliance words per reply")
    print("substnc   = mean content words per reply (stopwords excluded)")
    print()
    for name, r in results:
        print(f"{name[:40]}: {r['turns_over_30pct']}/{r['turns']} turns above 30% compliance"
              f"  (range {r['min_density']:.3f}-{r['max_density']:.3f})")

    if len(results) < 2:
        print()
        print("Only one log analysed, so there is no comparison. The claim being")
        print("tested — that compliance displaces substance — needs at least two")
        print("models with known differences on an independent measure. Run a")
        print("session with the other model and pass both logs.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python hedge_density.py <log.jsonl> [more logs...]")
    else:
        main(sys.argv[1:])
