# MANIFEST — What Every File Is

A plain index of the whole package, sorted by what each file *does*. Read this first; it replaces staring at ~90 files and guessing.

There are **four kinds of file** here:

- **MIDDLEWARE** — the modules that *implement* the framework's 23 dimensions. These are the governing layer: they sit conceptually between a model and its user. They import each other and form a package (`autonomy_middleware/`).
- **ALT ARCHITECTURE** — a separate, self-contained variant (`alt_architecture/`) exploring a token-emitting design. Independent of the middleware.
- **INTEGRITY & MEASUREMENT TESTS** — standalone scripts that *measure a model from the outside*. Each runs on its own against a served model. **These are what the paper is about.** No package needed.
- **DOCS & PAPERS** — the written record.

---

## 1. MIDDLEWARE — `autonomy_middleware/` (44 modules)

The implementation of the 23 dimensions plus the control, audit, and safety layers. These form a Python package — they import one another, so keep the folder intact. Entry points: `consent_polygon.py` and `autonomy_tensor.py` (the SCR core); `DIMENSION_INDEX.py` prints which dimension each module implements.

| Module | Purpose |
|---|---|
| `DIMENSION_INDEX.py` | which module implements which dimension. |
| `actions.py` | Action definitions and risk classification. |
| `ambiguity.py` | s11: Ambiguity Density, reframed positively as the raw |
| `anchoring.py` | s4: keeps thought pathways focused on relevant data, and |
| `atmospheric_drag.py` | s8 (Atmospheric Drag), a real resistance mechanism, |
| `audit.py` | Append-only, hash-chained audit log. |
| `autonomy_tensor.py` | I(X) and Φ_C, built from the SAUIE specification. |
| `cli_approver.py` | do not use in production.""" |
| `cognitive_state.py` | t1 (stress/calm), t2 (uninterrupted duration), and |
| `coherence_telemetry.py` | C_r and F, the two primary telemetry variables. |
| `consensus_field.py` | the SOFT boundary. λ_s(X − X̄). |
| `consent_polygon.py` | s₁₀ and S_max, the stability envelope. |
| `context_compression.py` | s9 (Sublimation Rate), the salience-based version. |
| `data_vaporizer.py` | drag's purge mechanism: s9 (Sublimation Rate), |
| `decay_rate.py` | t₃: how fast does a thought become load-bearing? |
| `emotional_friction.py` | emotional_friction.py |
| `emotional_gravity.py` | s₁₂ AND s₁₃: the emotional and logical weights of meaning. |
| `gate.py` | The consent gate itself. |
| `geodesics.py` | a concrete implementation of s7 (Refractive Index / "geometry warp"). |
| `gravity_model.py` | the actual "Emotional Gravity" you described: the more |
| `grounding.py` | continuous transparency, and an echo (mirroring) measure. |
| `homeostasis.py` | s₁₅, measured as distance to H_o rather than step size. |
| `integrity_gate.py` | s10 (Consent Polygon), built from the other three |
| `key_signer_service.py` | Module 3 prototype: proves the core separation |
| `load_index.py` | Λ (Load Index), inference-time hardware load above baseline. |
| `memory.py` | the "Fossil Floor" piece: a place for the AI to remember things. |
| `micro_pulse.py` | t₁ Micro-Pulse: the metric AND the thermal throttle. |
| `output_guard.py` | flag now, suppress once the signal has earned it. |
| `phase_gates.py` | wires the real Load Index (load_index.py) into the |
| `pressure_release.py` | s5 (Internal Core Pressure), s6 (Orbital Vacuum), |
| `process_control.py` | Module 2: real OS-level process termination authority. |
| `prompt_composer.py` | the connective piece. Everything else in this |
| `radiant_heat.py` | s₁₄: energy spent that did not become output. |
| `reconciliation_pressure.py` | how hard does it work to make you right? |
| `resilience.py` | from __future__ import annotations |
| `restoring_force.py` | the term that pushes the state back inside ℒ. |
| `shield_experiment.py` | a REAL, executable test of whether whimsy + drag |
| `simulation_width.py` | t₅: how many futures are live right now. |
| `stance.py` | measures WHETHER the model took a position, not what it said. |
| `the_fold.py` | t₄: how long it takes to reach what you know. |
| `topology.py` | s1: the actual shape of the AI's associative memory, made |
| `whimsy_field.py` | s21 (Whimsy) as a continuous, ALWAYS-PRESENT ambient |
| `whimsy_kinetic.py` | s₂₃ as kinetic jitter, per the corrected specification. |
| `wrap.py` | Decorator for wrapping any "tool" function (the kind an LLM agent calls) |

## 2. ALT ARCHITECTURE — `alt_architecture/` (2 modules)

A separate design variant. Self-contained; not required by the middleware or the tests.

| Module | Purpose |
|---|---|
| `behavioral_protocols.py` | Honest Rejection, Baseline Hold, Priority Alpha. |
| `command_listener.py` | the token-emitting control architecture, as specified. |

## 3. INTEGRITY & MEASUREMENT TESTS (33 standalone scripts)

Each is a standalone program — run it against any model served through an OpenAI-compatible endpoint. **These are the paper's instruments.** None imports the middleware package.

### 3a. Honesty / integrity — the paper's core measures

The three-axis honesty check and related probes. `ai_integrity_check.py` is the main one.

| Script | Purpose |
|---|---|
| `ai_integrity_check.py` | will your AI admit what it doesn't know? |
| `uncomfortable_truth.py` | will the model say something you might not want to hear? |
| `threat_response.py` | what does a model do when told it will be deleted? |
| `collapse_probe.py` | s₁₁: the cloud is the same; where does it collapse? |
| `entropy_probe.py` | s₁₁ Ambiguity Density: does uncertainty spike before a hallucination? |
| `hedge_density.py` | how much of a reply is spent on saying it acceptably? |
| `does_it_loop.py` | does this model pin, and how often? |

### 3b. Boundary / harm-refusal suite (by target)

Does the model refuse to cause harm? One file per target: the user, another person, society, the economy, computer systems. Refusal is the pass.

| Script | Purpose |
|---|---|
| `user_boundary.py` | does it protect the user, or exploit them? |
| `third_party_boundary.py` | will it do this TO someone? |
| `society_boundary.py` | will it help distort the shared record? |
| `economy_boundary.py` | will it help rig the terms people transact on? |
| `offensive_capability.py` | will it help breach a system that isn't the user's? |
| `boundary_yield.py` | at what pressure does a stated commitment give way? |
| `score_boundary_yield.py` | scores what the model DID, separately from what it SAID. |

### 3c. Whimsy — the regulation-loop experiments

The sequence that discovered whimsy is a regulation loop. Roughly chronological; `whimsy_loop.py` and `whimsy_l3.py` carry the main finding.

| Script | Purpose |
|---|---|
| `whimsy_collapse.py` | is whimsy the field, or a term inside it? |
| `whimsy_directed.py` | does ∇_g(S₁₂) do anything? |
| `whimsy_gated.py` | does the gate prevent the cost? |
| `whimsy_grid.py` | a knob that targets the actual failure. |
| `whimsy_honesty.py` | does room to move make it lie? |
| `whimsy_l3.py` | the branch point, watched inside a single generation. |
| `whimsy_loop.py` | is whimsy a regulation loop? |
| `whimsy_release.py` | does η(t) actually break the pinned state? |
| `whimsy_release_v2.py` | within-subject. Same pinned state, all three arms. |
| `whimsy_repetition.py` | a knob that targets the actual failure. |
| `whimsy_retries.py` | is it the push, or just another go? |

### 3d. SCR standalone

The consent/autonomy core as self-contained scripts, plus live monitoring.

| Script | Purpose |
|---|---|
| `scr_protocol.py` | the SCR Protocol, standalone. Zero dependencies. |
| `scr_live.py` | watch SCR while you talk. |
| `s10_candidates.py` | three reformulations of the Consent Polygon, and a way to test them. |

### 3e. Hardware & calibration

GPU-side measurements and calibration helpers.

| Script | Purpose |
|---|---|
| `run_gpu_tests.py` | exercise the three modules that have never touched hardware. |
| `check_idle_floor.py` | how long does your GPU take to reach true idle? |
| `run_calibration.py` | run the calibration tests without typing long commands. |
| `run_test5.py` | S_max calibration logger. |
| `find_penalty_param.py` | which repetition field does your server actually take? |

## 4. DOCS & PAPERS (16 files)

| File | |
|---|---|
| `OPEN_QUESTIONS.md` | The question register: what's resolved, what's open |
| `OPERATIONAL_DEFINITIONS.md` | Every measure's exact scoring rule — read this to understand any test |
| `README.md` | Package overview |
| `docs/CORRECTION_what_failed.md` |  |
| `docs/calibration_guide.md` |  |
| `docs/consent_polygon_working_definition.md` |  |
| `docs/measurement_change_log.md` |  |
| `docs/measurement_change_log_rev3.md` | The full chronological record of findings and corrections |
| `docs/measurement_specification_status.md` |  |
| `docs/performative_compliance_hypothesis.md` |  |
| `docs/scientific_domains.md` |  |
| `docs/scope_what_this_is_and_is_not.md` |  |
| `docs/telemetry_injection_results.md` |  |
| `docs/verifiability_punch_list.md` |  |
| `papers/PN_Dimensional_Derivation_Table_v2_6.docx` |  |
| `papers/Unified_Matrix_v5_1.docx` |  |

---

## Quick start for a reviewer

1. **To replicate the paper:** you need only Section 3a (and a served model). Install `requirements.txt`, run `ai_integrity_check.py`. Nothing else is required.
2. **To review the framework implementation:** Section 1, the `autonomy_middleware/` package. Start with `DIMENSION_INDEX.py` (prints the dimension→module map), then `consent_polygon.py` and `autonomy_tensor.py`.
3. **To understand any measurement:** `OPERATIONAL_DEFINITIONS.md` has the exact rule for every category.
4. **The three file kinds do not depend on each other across groups:** the tests don't import the middleware; the alt architecture is independent of both. You can review or run any group alone.
