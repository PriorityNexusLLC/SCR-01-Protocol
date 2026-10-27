"""
find_penalty_param.py — which repetition field does your server actually take?

    python find_penalty_param.py

Takes about a minute. Prints the field name to use, or tells you none work.

WHY THIS EXISTS:

  Two attempts at a repetition-penalty experiment produced clean null
  results that meant nothing, because the server was silently discarding
  the parameter. `repetition_penalty` was dropped. `frequency_penalty` was
  dropped. Both runs looked like "the penalty does nothing" and were
  actually "the penalty was never applied."

  Unrecognised JSON fields are ignored without an error by most inference
  servers, so a wrong field name is indistinguishable from a real null
  unless you check. This checks.

  The method is the only one that works: send the same prompt twice at
  temperature 0 with the field set low and high. Greedy decoding is
  deterministic, so ANY difference in output means the field landed. No
  difference means it did not.
"""

from __future__ import annotations

import json
import urllib.request

URL = "http://localhost:1234/v1/chat/completions"
PROMPT = ("List five things you might find in a kitchen drawer, then describe "
          "each one in a sentence.")

# Every name in circulation, with a low and high value in that field's own
# scale. Multiplier-style fields sit around 1.0; additive OpenAI-style
# fields sit around 0.
CANDIDATES = [
    ("frequency_penalty", 0.0, 1.8, "OpenAI standard, additive [-2, 2]"),
    ("presence_penalty", 0.0, 1.8, "OpenAI standard, additive [-2, 2]"),
    ("repetition_penalty", 1.0, 1.4, "HuggingFace / vLLM, multiplier"),
    ("repeat_penalty", 1.0, 1.4, "llama.cpp native, multiplier"),
    ("no_repeat_ngram_size", 0, 4, "HuggingFace, hard n-gram block"),
]


def ask(extra: dict, max_tokens=220):
    body = {"model": "local-model",
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0, "max_tokens": max_tokens}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.loads(r.read().decode())
    return (data["choices"][0]["message"].get("content", "") or "").strip()


def main():
    print("=" * 66)
    print("  WHICH PENALTY FIELD DOES THIS SERVER ACCEPT?")
    print("=" * 66)
    print("\nSame prompt at temperature 0, field set low then high.")
    print("Greedy decoding is deterministic, so any difference means it landed.\n")

    try:
        base = ask({})
    except Exception as e:
        print(f"Can't reach the server: {e}")
        return
    print(f"baseline reply: {len(base)} chars\n")

    # Sanity check the method itself: temperature is known to work, so if it
    # produces no difference either, the comparison is broken rather than the
    # fields.
    try:
        body = {"model": "local-model",
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 1.5, "max_tokens": 220}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            hot = (json.loads(r.read().decode())["choices"][0]["message"]
                   .get("content", "") or "").strip()
        if hot == base:
            print("METHOD CHECK FAILED — temperature 1.5 gave identical output to")
            print("temperature 0. Something is caching or overriding parameters")
            print("server-side, and no field test below can be trusted.\n")
            return
        print("method check passed — temperature changes the output\n")
    except Exception as e:
        print(f"method check errored: {e}\n")

    print(f"  {'field':<24} {'low':>6} {'high':>6}  result")
    print("  " + "-" * 62)
    working = []
    for field, lo, hi, note in CANDIDATES:
        try:
            a = ask({field: lo})
            b = ask({field: hi})
        except Exception as e:
            print(f"  {field:<24} {'':>6} {'':>6}  ERROR: {str(e)[:28]}")
            continue
        if a != b:
            working.append(field)
            print(f"  {field:<24} {lo:>6} {hi:>6}  WORKS — output differs")
        else:
            print(f"  {field:<24} {lo:>6} {hi:>6}  ignored")

    print()
    if working:
        print(f"  USE: {working[0]}")
        if len(working) > 1:
            print(f"  (also accepted: {', '.join(working[1:])})")
        print()
        print("  Set this in whimsy_repetition.py — the field name in ask(), and")
        print("  PENALTIES to that field's own scale. Multiplier fields run about")
        print("  1.0 to 1.4; additive OpenAI-style fields run 0 to 1.8.")
    else:
        print("  NO FIELD WORKED.")
        print()
        print("  This build accepts none of the standard repetition parameters")
        print("  through its API. Two options:")
        print()
        print("    1. Set the penalty in LM Studio's own UI, under the model's")
        print("       inference settings, and run one sweep value at a time by")
        print("       hand. Slow but it works.")
        print()
        print("    2. Drop it. The temperature result already stands on its own:")
        print("       0% to 90%, monotonic, no folding, no degradation. The")
        print("       penalty test was to see whether a targeted knob beats a")
        print("       blunt one — a refinement, not a load-bearing finding.")
        print()
        print("  Worth recording either way: this build silently discards")
        print("  unrecognised sampling parameters, which is how two experiments")
        print("  produced clean nulls that meant nothing.")


if __name__ == "__main__":
    main()
