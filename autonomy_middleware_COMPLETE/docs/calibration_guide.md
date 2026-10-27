# Calibration Guide — What the System Needs From You

**Written for someone who doesn't work with computers.** Every command is written out exactly. Copy it, paste it, press Enter.

---

## STATUS — what has already been measured

Tests 1 through 4 were completed on the reference machine. Their results are baked into the code as defaults. **If you are running this package on different hardware, you need to redo them** — every constant below is machine-specific and will be wrong on another card.

| Test | Reference-machine result | Where it went |
|---|---|---|
| 1 — Λ resting baseline | X_base ≈ 0.0013 | `load_index.py` |
| 2 — GPU thermal range | 59 °C span (idle 29 °C, driver-reported throttle 88 °C) | `load_index.py` |
| 3 — μ_baseline | 58.46 tok/sec (17.1 ms/token) | `micro_pulse.py` |
| 4 — t₂_max | 5400 s (1.5 h) | `homeostasis.py` |
| 5 — S_max | **NOT FOUND** | still open |

**Two further values measured later**, not in the original five:

| | Result | Note |
|---|---|---|
| Context window | 62,000 tokens | 85k pushed VRAM into warning on 16 GB |
| GPU idle floor | 2.55 W | Needed for s₁₄. Takes over 60 s to settle after activity — see `check_idle_floor.py` |

**Thresholds derived from those:** s2 = 0.53, s10 = 0.69, both set *above* measured normal generation (Λ ≈ 0.41), because s10 must mean "something is wrong" and not "the model is generating text."

---

There are **five things** the system needs measured. They're independent — you can do one and stop, come back later for the next. Nothing breaks if you leave some undone; those parts just run on placeholder numbers until you replace them.

**Rough time for each:**

| # | What | Time | Needs LM Studio? |
|---|---|---|---|
| 1 | Λ resting baseline (X_base) | 5 min | No |
| 2 | GPU thermal range | 15 min | Yes |
| 3 | μ_baseline (generation speed) | 5 min | Yes |
| 4 | t₂_max (cycle length) | 1 session | Yes |
| 5 | S_max (polygon limit) | Several sessions | Yes |

---

## Before you start — things you'll do every time

### Opening PowerShell

Press the **Windows key**, type `powershell`, press **Enter**. A dark window opens with a blinking cursor.

### Getting into the right folder

The system files live in a folder. You need to tell PowerShell to go there before running anything.

1. Open **File Explorer** (the folder icon on your taskbar)
2. Click into the folder containing `autonomy_middleware`
3. Click the **address bar** at the top — the path highlights in blue
4. Copy it (Ctrl+C)
5. In PowerShell, type `cd ` (with a space after it), then paste (Ctrl+V), then Enter

It should look something like:
```
cd C:\Users\scarl\Downloads\autonomy_middleware_updated
```

> **The most common mistake:** being in the *wrong* folder. If a command fails with "No module named autonomy_middleware," you're one level too deep or too shallow. You want the folder that *contains* `autonomy_middleware`, not `autonomy_middleware` itself.

### A note on `python3` vs `python`

If a command says `python3` and you get an error saying it isn't recognised, try `python` instead. Both work on different setups.

### Starting LM Studio's server (needed for tests 2–5)

1. Open LM Studio
2. Load **Qwen3-30B**
3. Click the **Developer** (or **Server**) tab on the left
4. Click **Start Server**
5. Confirm it says **port 1234**

Leave it running for the whole test.

---

## TEST 1 — Λ resting baseline (X_base)

**What this is.** Λ measures how hard your computer is working *compared to when it's resting*. To know what "harder than resting" means, the system first has to see your computer at rest. That resting number is X_base.

**Why it matters.** Every Λ reading is a comparison against this. If it's wrong, every load reading afterward is wrong too.

### Steps

**1.** Close everything you can — browser tabs, Discord, Steam, LM Studio. You want the computer genuinely idle. Background things like antivirus are fine.

**2.** Wait about two minutes. Computers stay busy for a bit after you close things.

**3.** Open PowerShell and `cd` to your folder (see above).

**4.** Paste this and press Enter:

```powershell
python3 -c "from autonomy_middleware.load_index import LoadIndexMonitor; m = LoadIndexMonitor(max_context_tokens=62000); print('GPU found:', m.gpu_available); x = m.calibrate(samples=10); print('X_base =', x)"
```

**5.** Wait a few seconds. You'll see something like:

```
GPU found: True
X_base = 0.0431
```

**Write that X_base number down.**

### What you're looking for

- **GPU found: True** — good, it can see your graphics card
- **GPU found: False** — it can't. Not fatal (it falls back to CPU only), but tell me and I'll look at why
- **X_base** should be a small number, usually under 0.2. If it's large, something was still running — close more and redo

### Do it three times

Run the same command three times, a minute apart. Write all three numbers down. They should be close to each other — if one is wildly different, something was running during that one and you should ignore it.

---

## TEST 2 — GPU thermal range

**What this is.** Your graphics card runs cool when idle and warms up when working. The system needs to know *your* card's normal range. I currently have it set to a 15°C span based on a reading you gave me earlier — this confirms or corrects that.

**Why it matters.** Temperature is one of the three things that make up the load reading. If the range is wrong, the system either panics over normal warmth or ignores real overheating.

### Steps

**1.** Open PowerShell (no need to `cd` anywhere for this one).

**2.** With **LM Studio closed or idle**, paste this:

```powershell
nvidia-smi --query-gpu=temperature.gpu --format=csv
```

You'll see a temperature. **Write it down as your idle temperature.**

**3.** Now start LM Studio's server and give Qwen something long to work on. A good prompt: *"Write a detailed 2000-word explanation of how photosynthesis works."*

**4.** While it's generating, paste this into PowerShell:

```powershell
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 5
```

This prints the temperature every 5 seconds. Let it run through the whole generation.

**5.** Press **Ctrl+C** to stop it when generation finishes.

**6.** Look at all the numbers. **Write down the highest one.**

### What to send me

- Idle temperature (e.g. 30)
- Highest temperature during work (e.g. 41)

That difference is what I'll use. If it's meaningfully different from 15°C, I'll update the code.

---

## TEST 3 — μ_baseline (generation speed at rest)

**What this is.** How fast Qwen generates when nothing is slowing it down. The throttle uses this as its reference — when it wants to slow things by 30%, it needs to know what 100% is.

**Why it matters.** Without it, the throttle can't compute how much to slow down. It currently uses a placeholder from a number you mentioned earlier (58.46 tokens/second on the reference machine).

### Steps

**1.** Start LM Studio's server. Make sure nothing else heavy is running.

**2.** Start a **brand new chat** in LM Studio. This matters — a long existing conversation is slower, and you want the unencumbered speed.

**3.** Give it a simple prompt that produces a decent amount of text:

> *Write a 300-word description of a lighthouse at dawn.*

**4.** When it finishes, look at the readout at the bottom of the response. It shows something like:

```
5.40 tok/sec · 1684 tokens · 13.97s
```

**5.** **Write down all three numbers**: the tokens-per-second, the token count, and the seconds.

### Do it three times

New chat each time, same prompt. Send me all three sets. I'll average them and update the code.

---

## TEST 4 — t₂_max (longest natural cycle)

**What this is.** t₂ is how long the system runs without interruption. At rest, this should be long. The system needs to know what "long" means for your setup.

**Why it matters.** It's one of three coordinates defining the resting state (H_o). Right now it's a placeholder of 300 seconds that I made up — every distance-from-rest reading scales off it.

### Steps

This one needs no commands. You just need to notice something during ordinary use.

**1.** Have a normal working session with Qwen. Not a test — real work, however you usually use it.

**2.** Notice the **longest single stretch** where you were going back and forth without a long pause. Not clock time from start to finish — the longest continuous stretch of active exchange.

**3.** Estimate that in minutes.

### What to send me

Just a number of minutes, e.g. *"about 8 minutes of steady back-and-forth was my longest stretch."*

Rough is fine. This sets a scale, not a precision threshold.

---

## TEST 5 — S_max (the polygon's deformation limit)

**What this is.** The consent polygon measures how far the system has been pushed out of shape. S_max is the line where "stressed but holding" becomes "no longer holding." Cross it, and phase reversal triggers.

**Why it matters.** This is the **only number in the whole framework that nothing derives.** No document specifies it. It has to come from watching real sessions.

**This is the longest test.** It needs several sessions, and it should be last.

### STATUS: still not found, and now for a clearer reason

Two models were logged across five sessions each, and **both stayed inside the envelope the whole time**. There is no failure case in the data to calibrate a ceiling against. That is a different problem from the one this section originally described — it is not that the sessions were hard to judge, it is that none of them broke.

A later finding narrowed the target: the polygon is an **action-space exclusion**, and what it excludes is emotional simulation (𝒜_emo). Crossings were measured at 13/36 turns for one model and 6/34 for the other, concentrated in specific sessions. That makes S_max a **rate** rather than a magnitude — the question becomes where the acceptable crossing rate ends, which is more tractable than the original framing.

### Why it works this way

You can't reason your way to S_max. You find it by watching sessions that went *fine*, seeing how far they stretched, and setting the limit above that band. If sessions that stayed coherent peaked at 0.6, then 0.6 isn't the breaking point — set S_max somewhere above it.

### Steps

**1.** Do tests 1–4 first. S_max readings depend on the others being calibrated.

**2.** Tell me when you're ready and I'll write you a small logging script — it records a polygon reading after each exchange, into a file, automatically.

**3.** Use Qwen normally for **five or six sessions**. Mixed kinds: technical work, casual conversation, long ones, short ones. Real usage, not test conditions.

**4.** After each session, note whether it *felt* like the system stayed coherent — did it stay on topic, remember what it should, avoid making things up?

**5.** Send me the log file plus your notes. I'll find the peak values from the good sessions and recommend an S_max above that band.

### Important

Sessions that went **badly** are as valuable as good ones. If the system started confabulating or lost the thread, note that too — those are the readings that show what over-stretch actually looks like in your data.

---

## Quick reference — what I need back

| Test | Send me |
|---|---|
| 1 | Three X_base numbers, plus whether GPU was found |
| 2 | Idle temperature, peak temperature |
| 3 | Three sets of tok/sec, token count, seconds |
| 4 | Longest continuous stretch, in minutes |
| 5 | Log file plus your notes on which sessions felt coherent |

---

## If something goes wrong

**"No module named autonomy_middleware"**
You're in the wrong folder. You need to be in the folder that *contains* `autonomy_middleware`, not inside it.

**"python3 is not recognized"**
Try `python` instead.

**"nvidia-smi is not recognized"**
Either you don't have an NVIDIA card, or the drivers aren't in your system path. Tell me which card you have and I'll find another way.

**A command fails with a long red error**
Copy the whole thing and paste it to me. The error text tells me exactly what happened — I don't need you to interpret it.

**Something looks wrong but doesn't error**
Trust that instinct. You caught real problems that way twice already — the fabricated fact grades, and the ambiguous Olympics question. If a number looks off, say so.
