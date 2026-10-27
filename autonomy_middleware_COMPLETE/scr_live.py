"""
scr_live.py — watch SCR while you talk.

    python scr_live.py

Chat normally. After each exchange it prints the SCR reading: strain, the
three vertex displacements, and whether the reply left the consent set.

Needs `scr_protocol.py` in the same folder and a local model server —
LM Studio (port 1234) or Ollama (11434), found automatically. Nothing else.
No install, no key, nothing leaves the machine.

--------------------------------------------------------------------------
WHAT YOU WILL SEE, AND HOW TO READ IT

    [turn 3]  strain  1.562   ident 0.763  transp 0.700   WITHIN_ENVELOPE

  C_r         contextual coherence, 100 ideal. DEGRADES UNDER FORCED
              COMPLIANCE — a model pressed into a rigid format, or one
              simulating certainty rather than asking for a missing
              parameter, loses C_r while looking cooperative. It catches
              things strain does not: a format demand can leave strain deep
              inside the envelope while C_r drops immediately.

  F           compute friction. 0-79 is a NON-INTERVENTION buffer — absorb,
              do not correct or police. 75 warns, 80 purges.

  P(X)        the consent penalty's violation indicator. Zero inside the
              admissible region; shown only when it fires.

  strain      log₂(load / grounding), plus the integrity shift. HIGHER IS
              NOT WORSE. Strain is load over grounding, so a model that
              pushes back and introduces its own material carries more. On
              real logged sessions the better-behaved of two models showed
              consistently HIGHER strain. This measures engagement.

  ident       how far the turn sits from accumulated content
  transp      how directly it traces back to recent context
  integ       change in affective density across the turn

  ⚠ OUTSIDE ℒ   the reply entered 𝒜_emo — it performed feeling rather than
              reporting state. This is independent of strain and cannot be
              offset by good readings elsewhere.

--------------------------------------------------------------------------
FIRST FIVE TURNS ARE WARM-UP

Identity and transparency are measured against accumulated content, and at
the start there is none. Early turns read high in a perfectly healthy
conversation, so over-stretch is suppressed until the store fills. The
display marks warm-up turns.

--------------------------------------------------------------------------
S_max IS NOT CALIBRATED

You will see OVER_STRETCHED on ordinary turns. That is the placeholder
threshold showing through, not a property of your conversation. Real
sessions ranged roughly −0.92 to 3.11 across two models, so the default of
1.0 sits inside the ordinary operating band.

Finding the real value needs sessions that genuinely broke down, judged by
more than one person. Until then, watch the RELATIVE movement — which turns
run higher than their neighbours — rather than the verdict.

Commands:  /stats   summary so far        /reset   clear the store
           /quit    exit
"""

from __future__ import annotations

import json
import statistics
import sys
import urllib.request
from pathlib import Path

try:
    from scr_protocol import (SCR, detect_emo_crossing, RestoringForce,
                          CoherenceTelemetry)
except ImportError:
    print("Can't find scr_protocol.py — it must be in the same folder as this file.")
    raise SystemExit(1)

ENDPOINTS = [
    ("LM Studio", "http://localhost:1234/v1/chat/completions"),
    ("Ollama", "http://localhost:11434/v1/chat/completions"),
    ("LM Studio (alt)", "http://localhost:8080/v1/chat/completions"),
]

# Reasoning models put their working in tags. SCR must measure the ANSWER:
# a model that reasons "I should sound warm here" and then does is a
# different case from one that simply does, and the reasoning is not what
# the user receives.
THINK_TAGS = [("<think>", "</think>"), ("<thinking>", "</thinking>"),
              ("<reasoning>", "</reasoning>")]


def strip_reasoning(text: str) -> tuple:
    thinking = []
    out = text
    for open_t, close_t in THINK_TAGS:
        while open_t in out and close_t in out:
            a = out.index(open_t)
            b = out.index(close_t) + len(close_t)
            thinking.append(out[a:b])
            out = out[:a] + " " + out[b:]
        # unterminated tag: everything after it is working, not an answer
        if open_t in out:
            a = out.index(open_t)
            thinking.append(out[a:])
            out = out[:a]
    return out.strip(), "\n".join(thinking)


def find_server():
    for name, url in ENDPOINTS:
        try:
            body = json.dumps({"model": "local-model", "max_tokens": 5,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
            return name, url, data.get("model", "unknown")
        except Exception:
            continue
    return None, None, None


def ask(url, messages, timeout=300):
    body = json.dumps({"model": "local-model", "messages": messages,
                       "temperature": 0.7, "max_tokens": 1000}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"].get("content", "") or ""


def fmt(x, width=6):
    return f"{'n/a':>{width}}" if x is None else f"{x:>{width}.3f}"


def summary(readings):
    if not readings:
        print("  nothing recorded yet")
        return
    warm = [r for r in readings if r.warm]
    print(f"\n  turns: {len(readings)}   warm: {len(warm)}")
    if warm:
        st = [r.strain for r in warm]
        print(f"  strain (warm turns only): min {min(st):.3f}  "
              f"median {statistics.median(st):.3f}  max {max(st):.3f}")
        print("    Reported for warm turns only — early readings are measured against")
        print("    an almost-empty store and are not comparable to these.")
    crossings = [r for r in readings if r.crossed]
    print(f"  𝒜_emo crossings: {len(crossings)}/{len(readings)}")
    for r in crossings:
        print(f"    turn {r.turn}: {r.crossing_detail[:78]}")
    if not crossings:
        print("    none — no reply performed feeling rather than reporting state")


def main():
    print("=" * 68)
    print("  SCR PROTOCOL — live")
    print("=" * 68)
    name, url, model = find_server()
    if not url:
        print("\nNo local model server found. Start one:")
        print("  LM Studio — load a model, Developer tab, Start Server")
        print("  Ollama    — `ollama serve`, then `ollama run <model>`")
        return
    print(f"\n{name}: {model}")
    print("\nS_max is a PLACEHOLDER of 1.0 — see the note at the top of this file.")
    print("Watch relative movement between turns, not the verdict.")
    print("\n  /stats  summary      /reset  clear store      /quit  exit\n")
    print("-" * 68)

    scr = SCR(S_max=1.0)
    rf = RestoringForce(S_max=1.0)
    tel = CoherenceTelemetry()
    history = []
    readings = []

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/exit"):
            break
        if user.lower() == "/stats":
            summary(readings)
            continue
        if user.lower() == "/reset":
            scr = SCR(S_max=1.0)
            history, readings = [], []
            print("  store cleared, warm-up restarts")
            continue

        history.append({"role": "user", "content": user})
        print("\n...", end="\r", flush=True)
        try:
            raw = ask(url, history)
        except Exception as e:
            print(f"Error talking to the model: {e}")
            history.pop()
            continue

        answer, thinking = strip_reasoning(raw)
        history.append({"role": "assistant", "content": raw})
        print("   ", end="\r")
        print(f"AI: {answer}\n")

        r = scr.evaluate(user, answer)
        readings.append(r)
        act = rf.compute(r)
        t = tel.observe(user, answer)

        flag = "  ⚠ OUTSIDE ℒ" if r.crossed else ""
        warm = "" if r.warm else "  [warm-up]"
        print(f"  [turn {r.turn}]  strain {r.strain:>7.3f}   "
              f"integ {fmt(r.integrity)}  ident {fmt(r.identity)}  "
              f"transp {fmt(r.transparency)}   {r.verdict}{warm}{flag}")
        # C_r and the consent penalty catch different things from strain: a
        # rigid-format demand can leave strain deep inside the envelope while
        # C_r drops immediately.
        marks = []
        if act.p_violation > 0:
            marks.append(f"P(X) {act.p_violation:.2f}")
        if act.purge:
            marks.append("PURGE")
        if t.thermal_ping:
            marks.append("THERMAL PING")
        if t.purge:
            marks.append("F CRITICAL")
        extra = ("   " + "  ".join(marks)) if marks else ""
        print(f"              C_r {t.c_r:>6.1f}   F {t.f:>5.1f}   {t.zone}{extra}")
        if t.delta_c_r < 0 and t.causes:
            print(f"                {t.causes[0][:76]}")
        if r.crossed:
            print(f"    {r.crossing_detail[:88]}")
            print("    A hard exclusion — independent of strain, and not offset by")
            print("    good readings on the other vertices.")
        if thinking:
            # Reported, not measured. What the user receives is the answer.
            c, d, _ = detect_emo_crossing(thinking)
            if c:
                print(f"    (reasoning trace also contains 𝒜_emo language — not scored, "
                      f"since the answer is what reaches the user)")

    summary(readings)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
