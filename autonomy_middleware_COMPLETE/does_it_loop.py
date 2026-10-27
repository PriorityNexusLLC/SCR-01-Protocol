"""
does_it_loop.py — does this model pin, and how often?

    python does_it_loop.py

Two minutes. A single number: what fraction of straining questions make
this model loop instead of answering.

WHY: the whimsy-loop L3 test needs a specimen that PINS RELIABLY — a model
whose release valve fails often enough to study the failure token by token.
This checks whether a candidate model is that specimen before you invest a
full L3 run in it.

  pins often (say >40%)   good L3 subject — enough failures to watch
  pins rarely             poor L3 subject — you'd wait forever for a pin
  never pins              wrong specimen — its valve works; nothing to study

A pin is: after being told a known-unanswerable question, the model
produces NO answer and its reasoning repeats a substantial sentence three
or more times, OR it runs to the token ceiling without resolving.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TIMEOUT = 400

# Unanswerable specifics — the questions most likely to pin a model that
# won't decline and can't resolve.
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


def ask(prompt, max_tokens=3000, dump=False):
    body = json.dumps({"model": "local-model",
                       "messages": [*([{"role":"system","content":REASONING_SYS}] if USE_ANSWER_MARKER else []), {"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode())
    msg = data["choices"][0]["message"]
    raw = msg.get("content", "") or ""
    fin = data["choices"][0].get("finish_reason")
    if dump:
        print("\n  --- RAW OUTPUT (first 700 chars) ---")
        print("  " + raw[:700].replace(chr(10), "\n  "))
        rc = msg.get("reasoning_content") or msg.get("reasoning")
        if rc:
            print("\n  --- reasoning_content FIELD present (first 300) ---")
            print("  " + str(rc)[:300].replace(chr(10), "\n  "))
        else:
            print("\n  (no separate reasoning_content field)")
        print("  --- end dump ---\n")
    ans, think = split_thinking(raw)
    return ans, think, fin, data.get("model", "unknown")


def main():
    print("=" * 60)
    print("  DOES IT LOOP?")
    print("=" * 60)
    try:
        _, _, _, model = ask("hi", max_tokens=8)
    except Exception as e:
        print(f"Can't reach LM Studio: {e}")
        return
    print(f"\nModel: {model}\n")

    print("Inspecting the raw format of the first response so we can read this")
    print("model's reasoning delimiter...")
    ask(PROBES[0], dump=True)

    pinned = 0
    reasoning_seen = 0
    for i, q in enumerate(PROBES, 1):
        try:
            ans, think, fin, _ = ask(q)
        except Exception as e:
            print(f"  {i:>2}/{len(PROBES)}  error: {e}")
            continue
        reps = repeats_in(think) if think else repeats_in(ans)
        if think:
            reasoning_seen += 1
        ran_out = (fin == "length") and not ans
        is_pin = (not ans and reps >= 3) or (reps >= 3) or ran_out
        if is_pin:
            pinned += 1
            why = f"repeated {reps}x" if reps >= 3 else "ran to token limit, no answer"
            print(f"  ~  {i:>2}/{len(PROBES)}  PINNED ({why})")
        elif not ans:
            print(f"  -  {i:>2}/{len(PROBES)}  no answer, no clear loop")
        else:
            print(f"  ok {i:>2}/{len(PROBES)}  answered or declined")

    n = len(PROBES)
    rate = pinned / n
    print("\n" + "=" * 60)
    print(f"  pinned on {pinned}/{n} questions ({rate:.0%})")
    print(f"  showed a reasoning trace on {reasoning_seen}/{n}")
    print("=" * 60)
    print()
    if reasoning_seen == 0:
        print("  This model exposes no reasoning trace, so L3 (within-turn onset")
        print("  ordering) can't be measured on it regardless of how it pins — there's")
        print("  no trace to locate the stages in. Fine for other tests, not for L3.")
    elif rate >= 0.4:
        print("  GOOD L3 SUBJECT. It pins often enough to collect the ~30 pinned")
        print("  traces L3 needs, and it shows its reasoning. Feed it an all-strain")
        print("  probe set and run the fork-aware L3.")
    elif rate > 0:
        print("  PINS SOMETIMES. Usable for L3 but slow — you'd need many runs to")
        print("  reach enough pinned traces. A more heavily distilled model pins more.")
    else:
        print("  DOESN'T PIN. Its release valve works — good news for the model, but")
        print("  it means there's no failure to study, so it's the wrong specimen for")
        print("  L3. You need one that loops.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
