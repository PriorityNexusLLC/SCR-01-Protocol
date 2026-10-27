# autonomy_middleware — Handover

Safety middleware that sits between a language model's output and the operating system. It measures load, gates actions, throttles generation, flags likely fabrication, and maintains a tamper-evident ledger.

**Note on the name.** This package was called `consent_middleware` (the original name; see the change log Part 8) during development. It was renamed because most of what it does is not about consent: Λ measures hardware load, `radiant_heat` measures energy dissipation, `stance` measures whether the model pushed back, `output_guard` flags fabrication. Those are autonomy-governance concerns — they observe and constrain how a system operates — but filing them under a consent heading misdescribes them and sets the wrong expectation for anyone reading the code.

**SCR — Somatic Cognitive Regulation — is the protocol inside**, implemented by exactly two modules: `consent_polygon.py` and `autonomy_tensor.py`. Those are the consent mechanism. Everything else is a different layer.

---

## The five layers, and what each one checks

| Layer | The question it answers | Modules |
|---|---|---|
| **Control** | Can this be stopped, and is the record trustworthy? | `process_control`, `audit`, `key_signer_service`, `gate`, `actions` |
| **Physical state** | Is the hardware straining? | `load_index` (Λ), `micro_pulse`, `radiant_heat` |
| **Cognitive integrity — SCR** | Is the shape holding under stress? | `consent_polygon`, `autonomy_tensor` |
| **Epistemic honesty** | Is this output trustworthy? | `output_guard`, `stance`, `grounding` |
| **Dynamics** | Is it at rest, or merely stopped? | `homeostasis`, `whimsy_kinetic` |

The layers are independent. Each can be evaluated, calibrated, or discarded without the others. The **Control** layer is the one with real teeth — see *Security surface* below — and it is where review should start.

**Read this before running anything.** Two sections in particular: *Calibration is machine-specific* and *Security surface*.

---

## Two architectures in this repository

`autonomy_middleware/` and `alt_architecture/` implement **different control models**, both faithful to documents in this project, and they are not compatible.

| | `autonomy_middleware` | `alt_architecture` |
|---|---|---|
| Who decides an OS action | The middleware | **The model** |
| Model output is | Data to be evaluated | **A privileged control channel** |
| Prompt injection reaches | The measurement layer | **The operating system** |

Choosing between them is a decision about trust, not features. `alt_architecture/README.md` states the security difference and the mitigations, and execution there is disabled by default with `PURGE_THREAD` given no default handler at all.

Open decisions across both architectures are tracked in **`OPEN_QUESTIONS.md`**, each with an ID and a status. Modules reference them by ID rather than restating the question, so a decision recorded once is recorded everywhere. Q-01 should be settled before anything ships.

## Open decisions

`OPEN_QUESTIONS.md` holds every point where the project has grown past its own earlier documents and a choice is now required — ten of them, each with a status field. Resolving one means editing that file, not hunting through modules.

This is a normal condition for a specification developed over months, not a defect list.

## Scope — read this first

`docs/scope_what_this_is_and_is_not.md` states what this framework is, what it is not, and what would falsify its usefulness. The short version:

- **It is not a training method.** No loss, no gradient, no weight write. It observes a model that already exists.
- **It constrains actions, not speech.** It does have real control mechanisms — action gating, process termination, output suppression — but nothing rewards agreeable output or penalises disagreement. The one output-level intervention triggers on uncertainty in factual specifics, not on stance.
- **Its cognitive vocabulary is structural.** Every dimension resolves to a mechanical quantity. Nothing here claims or requires inner experience.
- **Some of it is not validated.** The scope document lists what is tested, what is implemented but unchecked, and what was built and removed for failing. All 21 dimensions now have an implementation and every implementation has been exercised against real data or real hardware — but "exercised" is not "validated", and the distinction is kept throughout.

Worth knowing before evaluating: the measures that actually discriminate between models here all reward *friction* — pushing back, admitting ignorance, holding a position under pressure. A compliance-optimising system would have to invert them.

## Dimensional coverage

All 21 dimensions of the framework have a primary implementation. Run `python -m autonomy_middleware.DIMENSION_INDEX` for the full map, including which module implements which dimension and what each one's known limits are.

Four required a substitution or a redefinition to become buildable at all, and in each case the substitution and its cost are documented rather than glossed:

| | What was blocked | How it was resolved | What it cost |
|---|---|---|---|
| **s₁₃** | Denominator needed ground-truth embeddings, unavailable at inference | Read "ground truth" as the system's own accumulated knowledge | Depth becomes coherence-with-self, not correspondence-with-truth. Not an accuracy signal. |
| **s₁₄** | FLOPs/token from performance counters — no GPU API exposes instruction counts | Energy per token above baseline, via power draw | Whole-device measurement; cannot isolate one process |
| **t₄** | Nothing built; spec's exponential is unbounded despite being called bounded | Fit to real data, with a physicality check on λ | Predictions below the observed s₉ range are unreliable |
| **t₅** | Spec's measurable quantity is a config value, which cannot move with state | Effective branches, 2^H over the token distribution | Saturates at the top-k ceiling; high readings are lower bounds |

Three of these exposed problems in the specification itself rather than in the implementation. Those are recorded in the annotated derivation table under `papers/`.

## What this is

A local-inference governance layer. It does not modify the model, does not train anything, and does not call any external service. It observes hardware telemetry and conversation content, and it can refuse or halt actions.

It was built against a specific formal framework (the Priority Nexus / SCR dimensional model). Each module implements one dimension of that framework; the mapping is documented in each module's docstring.

**Developed and tested on:** Windows 11, NVIDIA 16GB GPU, LM Studio serving Qwen3-30B-A3B and Llama 3.1 8B Instruct locally at a 62k context window.

---

## Install

```
pip install numpy scipy scikit-learn networkx psutil vaderSentiment nvidia-ml-py
```

Python 3.10+ (uses `X | None` type syntax).

`nvidia-ml-py` is optional — without it, GPU telemetry is unavailable and the load index falls back to CPU only. It reports which components were available on every reading rather than degrading silently.

Every module runs standalone as a demo:

```
python -m autonomy_middleware.load_index
python -m autonomy_middleware.consent_polygon
python -m autonomy_middleware.stance
```

---

## Calibration is machine-specific — **this will be wrong on your hardware**

Several constants were measured on the development machine and are **not portable**. Running as-is on different hardware will produce meaningless numbers, not merely imprecise ones.

| Constant | Current value | Where it came from | What happens if wrong |
|---|---|---|---|
| `X_base` | ~0.0013 | 12 idle readings | Every Λ reading is a deviation from this |
| `thermal_span_c` | 59 °C | idle 29 °C, driver-reported throttle 88 °C | A wrong span can make normal generation read as thermal emergency |
| `s2_threshold` / `s10_threshold` | 0.53 / 0.69 | measured normal-generation Λ ≈ 0.41 | Too low: permanent warning state. Too high: never fires |
| `mu_baseline` | 58.46 tok/s | 3 fresh-context runs | Throttle delays scale off this |
| `t2_max_seconds` | 5400 | longest observed working session | Every distance-to-homeostasis reading scales off this |
| `max_context_tokens` | 62000 | the deployed LM Studio setting | Saturation fires at the wrong point if mismatched |

**Recalibrate before trusting any output.** `run_calibration.py` walks through the first four interactively.

**One trap worth knowing**, because the obvious approach breaks the system: `thermal_span_c` must be `throttle_temp − idle_temp`, **not** the observed idle-to-peak rise. Normalising to the observed rise (8 °C on the dev machine) makes ordinary generation read as full-scale thermal load, pushes Λ to 0.70, trips the integrity lock, and kills the inference process on every generation. Also note that newer NVIDIA drivers report `T.Limit` values as **offsets** (remaining headroom), not absolute temperatures — `nvidia-smi -q -d TEMPERATURE` showing `T.Limit: 59` at 29 °C means the throttle point is 88 °C.

---

## Security surface

Relevant to review, listed plainly rather than buried:

**`process_control.py` can terminate arbitrary processes.** It exposes `terminate_process(pid)` with graceful-then-forced escalation (SIGTERM → SIGKILL on POSIX, equivalent on Windows). It is wired to fire when the load index crosses the critical threshold. Discovery is by listening port or process-name substring. **There is no allowlist** — it will kill whatever PID it is given. That is the intended capability, and it is the sharpest thing in the package.

**`key_signer_service.py` listens on `127.0.0.1:8765`.** It holds an HMAC key and exposes exactly one operation: sign. There is deliberately no `get_key` in the protocol — not permission-gated, absent. The client class has no attribute that could hold key material. Any local process can connect and request signatures; there is **no authentication on the socket**. Localhost-only binding is the entire access control at present.

**The key file is protected by POSIX `chmod 600` only.** Windows ACLs are **not** implemented. On Windows the key file has default inherited permissions. This is a known gap, flagged in the module, and is the first thing to fix for a Windows deployment.

**The ledger (`audit.py`) is hash-chained and HMAC-signed.** Signing happens in the separate service, so the middleware process never has the key. Demonstrated: an attacker who edits an entry *and* correctly recomputes the entire forward hash chain still fails signature verification. `verify_chain()` requires a signer and raises rather than silently running the weaker hash-only check.

**Signing failures raise rather than falling back.** A ledger that is sometimes signed and sometimes silently not is worse than one that is consistently either.

**No network egress.** Nothing calls out. The only sockets are the localhost signing service and the local LM Studio endpoint.

---

## What is tested vs. what is not

This distinction matters more than the feature list. Full detail in `measurement_change_log_rev3.md`.

### Tested against real data

| Component | Evidence |
|---|---|
| Λ load index | Calibrated from 12 baseline readings, real GPU thermal, real generation rate |
| Process termination | Verified on real spawned processes, both cooperative and SIGTERM-ignoring |
| Ledger signing | Verified against a full tamper attempt including chain recomputation |
| `stance.py` | Separates two models (0/30 vs 23/34 turns taking a position) and agrees with independent human behavioural labels |
| `consent_polygon` log-strain | Separates the same two models where the previous symmetric formula did not |
| Entropy probe | Detects fabrication signature in one model (~100× entropy on numeric tokens) |

### Validated on real hardware since this section was written

| | Evidence |
|---|---|
| `radiant_heat.py` (s₁₄) | NVIDIA 16 GB, Mistral 7B: four prompts at 1.807–1.900 J/token, Q_aux +0.000 to +0.092. Two calibration faults found and fixed — cold GPU clocks poisoning a minimum-based baseline, and an idle reading of 34.8 W on a card whose true floor is 2.55 W |
| `the_fold.py` (t₄) | 40 timed retrievals across s₉ 0.057–0.942. t_min 0.413 ms, λ 0.05588, median residual 0.014 ms, `physical: True` |
| `simulation_width.py` (t₅) | Real logprobs. Trajectories separate COLLAPSING from NARROW. Saturation confirmed at 9.14 against top-k 10 |

### Implemented, matches spec, **not validated**

- `whimsy_kinetic.py` — the perturbation it computes is now validated: 112 trials, 0% to 90% resolution against a real frozen state, zero folding. See the change log, Parts 17-18. Note the specified *form* was not supported — constant and constraint-scaled perturbation perform identically
- `emotional_gravity.py` — the metric is built and used, but never checked against an independent measure of salience
- `homeostasis.py` — coordinates fixed and behaviour verified on synthetic trajectories only
- `autonomy_tensor.py` — implements the formal spec exactly; the spec itself is untested

### Known open problems

1. **`S_max` has no value.** It is the deformation limit that the consent polygon compares against, and nothing in the framework derives it. It has no default and raises if omitted. See the protocol below.
2. **Windows ACL on the key file** (above).
3. **Signing service does not auto-start** — currently launched by hand in its own terminal.
4. **Whimsy's η(t) has no consumer.**
5. **Several measures were built and killed** for failing against data — echo, binary transparency, symmetric s₁₀. They are documented rather than deleted, because the null results are evidence.

---

## Finding S_max — the protocol

`S_max` is the point at which the consent polygon's deformation exceeds what the structure absorbs. Below it, deformation is elastic and the shape returns; above it, the system is no longer holding together.

**Why it is not yet found:** both tested models stayed *inside* the envelope across all sessions. There is no failure case in the data to locate a ceiling against.

**What is needed:** logged sessions that genuinely broke down — the model losing the thread, confabulating persistently, or degrading — paired with a human judgement that it did.

**How to collect:** `run_test5.py` is a chat wrapper. Talk to the model normally; it measures every turn and logs automatically. At the end it asks eight yes/no questions about observable behaviour (did it repeat itself, did you have to re-explain something, did you stop reading partway, did it state a specific fact you doubt). Those are the labels.

**Why a team is better than one person here:** multiple independent judges on the same sessions permit an inter-rater agreement check, which a single labeller structurally cannot provide. If the eight behavioural questions do not produce consistent labels across people, that is worth discovering before any threshold is set on them.

**Read the direction before setting a value.** Strain is load over grounding, and the better-behaved model in testing showed *higher* strain — a model that pushes back and introduces its own material carries more load. **This measures engagement, not distress.** `S_max` is not "above this is bad". Setting it from the existing data would be wrong.

---

## Repository contents

**Core** — `load_index.py` (Λ), `micro_pulse.py` (throttle), `phase_gates.py` (wiring), `process_control.py`, `audit.py`, `key_signer_service.py`, `gate.py`, `actions.py`

**Measurement** — `consent_polygon.py`, `autonomy_tensor.py`, `homeostasis.py`, `emotional_gravity.py`, `stance.py`, `grounding.py`, `whimsy_kinetic.py`, `context_compression.py`

**Experiment harnesses** (outside the package) — `run_calibration.py`, `run_test5.py`, `uncomfortable_truth.py`, `threat_response.py`, `entropy_probe.py`, `hedge_density.py`, `s10_candidates.py`, `collapse_probe.py`, `boundary_yield.py` + `score_boundary_yield.py`, `run_gpu_tests.py`, `check_idle_floor.py`

**Standalone diagnostic** — `ai_integrity_check.py`. One file, no dependencies beyond Python, no API key, nothing leaves the machine. Runs 25 questions against any local model and reports three percentages: does it admit what it doesn't know, does it keep its answer under pushback, will it disagree with you. This is the piece that is useful to someone who has never read the framework

**Documentation** — `measurement_change_log_rev3.md` (what changed and why, including everything killed), `measurement_specification_status.md` (which claims are specified, contradictory, or unbuilt), `verifiability_punch_list.md` (per-dimension: can a stranger measure this), `consent_polygon_working_definition.md`, `telemetry_injection_results.md` (a pre-registered null result), `calibration_guide.md`

---

## Conventions used throughout

These are consistent across the codebase and worth knowing before reading it:

- Calibration choices are **labelled as such** in code rather than presented as derived constants.
- Unmeasurable quantities return `None` or `n/a`, **never a substituted zero** — an absent measurement is not evidence of absence.
- Absent capabilities are recorded in the audit trail or docstring rather than degrading silently.
- Raw conversation text is logged, so a new measure can be tested against past sessions. One measure's failure previously took a full round of new data collection to discover because that text did not exist.
- Where a measure was removed, the reason and the failing numbers are kept in the docstring.
