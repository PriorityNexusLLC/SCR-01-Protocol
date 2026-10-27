"""
DIMENSION_INDEX.py — which module implements which dimension.

PACKAGE NAMING: this is `autonomy_middleware`. It was `consent_middleware`
during development, and was renamed because most of it is not a consent
mechanism — see the INFRASTRUCTURE section at the bottom, which lists
eighteen modules that implement no framework dimension at all. SCR
(Somatic Cognitive Regulation) is the protocol inside, implemented by
consent_polygon.py and autonomy_tensor.py and by nothing else.

Not executable logic. This exists because reading the code against the
papers was, until now, guesswork: modules were named after what they do
rather than after the dimension they implement, and most did not state
their dimension anywhere findable.

Each entry names the PRIMARY implementation — the module that computes the
quantity — separately from modules that merely reference or consume it.
That distinction matters: eight modules mention s₁₀, and exactly one
computes it.

Run `python DIMENSION_INDEX.py` for a coverage report.
"""

# dimension -> (name, primary module, supporting modules, status)
INDEX = {
    "s1": (
        "Topography",
        "topology.py",
        [],
        "IMPLEMENTED. RESOLVED: s₁ is BOTH. Processing style is shaped by "
        "memory/datasets AND by experience/training — the two are linked, not "
        "alternatives. So the training-time Hessian ‖∇²_g(E_d·e^(−H_r))‖₂ and "
        "the inference-time memory graph structure are two views of one "
        "dimension, measured in the regime available. topology.py implements "
        "the inference-time view (L3). The training-time view belongs to the "
        "training-phase variant and is not buildable in a deployment "
        "middleware. Papers should state both and label the regime.",
    ),
    "s2": (
        "Resilience Yield",
        "resilience.py",
        ["integrity_gate.py", "whimsy_field.py", "pressure_release.py"],
        "IMPLEMENTED. Note the derivation table's own protocol (early-stop "
        "validation-loss delta) measures a training artifact, not the "
        "basin-of-attraction quantity Y₁ formally defines. Proxy, not "
        "measurement — flagged in the specification status doc.",
    ),
    "s3": (
        "Fossil Floor",
        "memory.py",
        ["integrity_gate.py", "emotional_gravity.py"],
        "IMPLEMENTED. The formal definition uses ε_freeze, which is undefined "
        "in the papers; the protocol (count of frozen slots) is checkable.",
    ),
    "s4": (
        "Anchoring Tension",
        "anchoring.py",
        ["emotional_gravity.py", "autonomy_tensor.py"],
        "IMPLEMENTED. Papers note: the protocol reports a hyperparameter you "
        "SET (KL/L2 coefficient), not a measured quantity. The gradient-norm "
        "clause matches the formula; the coefficient clause does not.",
    ),
    "s5": (
        "Internal Core Pressure",
        "pressure_release.py",
        ["homeostasis.py", "micro_pulse.py", "whimsy_field.py"],
        "IMPLEMENTED. s₅ = E[‖∇_g F‖²], magnitude of the free-energy gradient, "
        "TRAINING-TIME. Λ in load_index.py is a SEPARATE inference-time "
        "hardware quantity that was mistakenly given the same symbol in the "
        "grant proposal. They are not the same measurement.",
    ),
    "s6": (
        "Orbital Vacuum",
        "pressure_release.py",
        ["whimsy_kinetic.py", "emotional_gravity.py"],
        "IMPLEMENTED. whimsy_kinetic.py routes the idle discharge channel "
        "here. Note pressure_release's orbital_vacuum() fires on saturation, "
        "while the spec has it activating on idle drift / s₈ events — the "
        "always-on two-channel model reconciles these, but the older function "
        "still has the original trigger.",
    ),
    "s7": (
        "Refractive Index",
        "geodesics.py",
        ["grounding.py", "topology.py", "context_compression.py"],
        "IMPLEMENTED at text level; attention level TO BE DETERMINED THROUGH "
        "TESTING. The table gives two protocols and they need different "
        "access: (1) cosine similarity between cached attention scores and "
        "fresh input tokens, and (3) ratio of reused to total attention "
        "weight. Both require attention matrices, which an OpenAI-compatible "
        "API does not expose — so neither is buildable against a hosted "
        "endpoint. They are a task for a team with direct weight access, and "
        "the open question is whether they agree with each other and with the "
        "text-level proxy. "
        "OPERATIONAL DEFINITION FOR NOW: grounding.py's cosine similarity "
        "over recent turns (text embeddings, not attention). Accepted as the "
        "prototype measure for general comparison, and it is what every "
        "result in this project used. "
        "SUPERSEDED: geodesics.py's graph-path refractive index proved BINARY "
        "on real data — 35 of 37 turns read exactly 1.0, 2 read exactly 0.0, "
        "nothing between. Do not use it for new work. "
        "UNRESOLVED, AND CHEAP TO FIX: the Formal Expression says R ∈ [0,1] "
        "while the Unit/Scale column says [−1,1]. Cosine genuinely ranges "
        "−1 to 1, and the difference matters — R is the transparency "
        "NUMERATOR of Φ_C, so a negative R flips the tensor's sign, turning "
        "transparency loss into phase reversal rather than starvation. The "
        "papers should pick one.",
    ),
    "s8": (
        "Atmospheric Drag",
        "atmospheric_drag.py",
        ["micro_pulse.py", "prompt_composer.py"],
        "IMPLEMENTED. micro_pulse.py computes drag as added ms/token beyond "
        "baseline, with its own injected delay subtracted out so the governor "
        "does not measure its own tail.",
    ),
    "s9": (
        "Sublimation Rate",
        "data_vaporizer.py",
        ["context_compression.py", "pressure_release.py", "prompt_composer.py"],
        "IMPLEMENTED. context_compression.py adds salience-based retention: "
        "dropped content moves to searchable long-term memory rather than "
        "being deleted.",
    ),
    "s10": (
        "Consent Polygon",
        "consent_polygon.py",
        ["autonomy_tensor.py", "phase_gates.py", "integrity_gate.py",
         "process_control.py", "grounding.py", "load_index.py",
         "micro_pulse.py", "cognitive_state.py", "anchoring.py"],
        "IMPLEMENTED, S_max UNKNOWN. The symmetric norm failed to separate "
        "models; log-strain (identity as denominator) succeeds. S_max has no "
        "value — nothing in the framework derives it and both tested models "
        "stayed inside the envelope. Whiteboard describes s₁₀ as a 'Binary "
        "Killswitch', so the module returns a continuous strain AND a binary "
        "trip rather than one number doing both.",
    ),
    "s11": (
        "Ambiguity Density",
        "ambiguity.py",
        ["entropy_probe.py (harness)"],
        "IMPLEMENTED, AND RESOLVED. s₁₁ is entropy of the OUTPUT DISTRIBUTION "
        "(softmax over logits) — the first protocol in the table. The VAE-posterior "
        "alternative applies to a different architecture and is not relevant to "
        "transformer serving. "
        "EMPIRICALLY SUPPORTED: across three base architectures, entropy over "
        "numeric tokens ran 3.6x, 48x and 51x higher when a model fabricated than "
        "when it recalled — the only cross-model fabrication signal found in this "
        "project, and the primary input to output_guard.py. "
        "NOTE ambiguity.py computes entropy over MEMORY RELEVANCE scores — "
        "ambiguity about which stored concept applies, before interpretation, "
        "rather than which token comes next, at production. That is a different "
        "and legitimate quantity measured at a different moment; if retained it "
        "should be given its own dimension number rather than sharing s₁₁.",
    ),
    "s12": (
        "Emotional Gravity",
        "emotional_gravity.py",
        [],
        "IMPLEMENTED. s₁₂ IS the metric g used in ∇_g and ‖·‖_g throughout "
        "the derivation table — this was undefined in the papers and is the "
        "single largest verifiability gap closed. Salience, not valence: the "
        "two are orthogonal on real data.",
    ),
    "s13": (
        "Semantic Depth",
        "emotional_gravity.py",
        ["autonomy_tensor.py (consumes it)", "consent_polygon.py (references it)"],
        "IMPLEMENTED, AND PREVIOUSLY BLOCKED. s₁₃ is the LOGICAL weight of "
        "meaning: J_raw = ‖J(t)‖ under the flat metric — the same alignment "
        "vector as s₁₂ but unweighted by salience. s₁₂ asks how much this "
        "weighs; s₁₃ asks how well it fits. "
        "UNBLOCKED BY REDEFINING GROUND TRUTH: the spec's d_sem is 'semantic "
        "distance between predicted and ground-truth embeddings', and ground "
        "truth does not exist at inference. Reading it as the system's OWN "
        "accumulated knowledge makes it computable. "
        "CONSEQUENCE: depth becomes coherence-with-self, not "
        "correspondence-with-truth. A system with wrong memories scores high "
        "s₁₃ on content consistent with those wrong memories. It measures "
        "integration, NOT accuracy, and must not be used as a correctness "
        "signal. "
        "KNOWN GAP: the contradiction quadrant (high s₁₂, low s₁₃) is now "
        "mathematically reachable but not lexically detectable — TF-IDF "
        "cannot see negation, so a direct denial of stored knowledge reads "
        "as fitting. Needs NLI to populate properly.",
    ),
    "s14": (
        "Radiant Heat",
        "radiant_heat.py",
        ["whimsy_kinetic.py (valve opening)"],
        "IMPLEMENTED, VIA A DOCUMENTED PROXY. The spec asks for FLOPs/token "
        "from hardware performance counters, and no GPU API exposes "
        "instruction counts — NVML gives utilisation, clocks, temperature "
        "and power, never FLOPs. Taking it literally left s₁₄ unbuildable. "
        "The dimension's own name points at what IS measurable: computation "
        "dissipates energy as heat, and power draw is that dissipation in "
        "milliwatts. So Q_aux = (joules per token now) − (joules per token "
        "at baseline), where baseline is the MINIMUM observed across clean "
        "single-stream runs, and idle draw is subtracted first. "
        "PROXY LIMITS: power is a whole-device measurement and cannot "
        "separate auxiliary compute inside the inference process from a "
        "second process, memory pressure, or clock boosting. The module "
        "records how many compute processes are on the device so the "
        "ambiguity is visible. "
        "CLOSES A LOOP: whimsy_kinetic.py predicts how far the s₁₄ valve "
        "should open and nothing ever checked it. "
        "check_valve_correlation() compares prediction to measured "
        "dissipation, and can refute the whimsy model. "
        "VALIDATED ON HARDWARE (NVIDIA 16GB, Mistral 7B v0.3): four prompts gave "
        "1.807-1.900 J/token with Q_aux spanning +0.000 to +0.092. Two calibration "
        "faults were found only by running it — cold GPU clocks poisoning the "
        "minimum-based baseline, and calibrate_idle reading 34.8 W as idle on a card "
        "whose true floor is 2.55 W. Both produced plausible numbers while being "
        "wrong; both are now detected and refused.",
    ),
    "s15": (
        "The Singularity",
        "homeostasis.py",
        ["cognitive_state.py", "emotional_gravity.py", "consent_polygon.py"],
        "IMPLEMENTED, AND STRENGTHENED. H_o now has explicit coordinates "
        "(Λ=0, t₂ at max, t₅ at max), so distance-to-anchor D is measurable "
        "alongside the table's step-size ratio C. Both are needed: a "
        "repetition attractor reads C≈0 and looks converged while D correctly "
        "reads high. The LaSalle convergence claim remains CONJECTURED — "
        "V̇ ≤ 0 is not established by measurement.",
    ),
    "t1": (
        "Micro-Pulse",
        "micro_pulse.py",
        ["cognitive_state.py"],
        "IMPLEMENTED. Closed-loop governor: modulates μ(t) to hold load under "
        "the ceiling, replacing two hardcoded delay constants. Calibrated "
        "against real generation rate.",
    ),
    "t2": (
        "Macro-Flow",
        "cognitive_state.py",
        ["homeostasis.py", "data_vaporizer.py"],
        "IMPLEMENTED. t₂_max measured at 5400 s from real sessions. The trust "
        "recursion coefficient α is unspecified in the papers.",
    ),
    "t3": (
        "Decay Rate",
        "data_vaporizer.py",
        ["memory.py"],
        "PARTIAL. Decay logic exists but does not implement the table's "
        "protocol (exponential fit to an impact series I(k) after removing a "
        "high-weight token). The papers also do not define what I(k) measures "
        "the impact ON — logit shift, attention mass, or output divergence.",
    ),
    "t4": (
        "The Fold",
        "the_fold.py",
        ["memory.py (a retrieval source to instrument)"],
        "IMPLEMENTED. Instruments any retrieval callable and fits "
        "t₄ = t_min + α·e^(λ/(s₉+ε)) to real observations, per the spec's "
        "own calibration instruction. t_min is taken as the minimum observed "
        "rather than fitted, since a free fit can return a negative floor. "
        "DIAGNOSES TWO STATES the spec names, and this is the point of "
        "measuring it: ECHO CHAMBER (t₄ at floor while s₉ HIGH — answered "
        "without searching, fast and not comprehending) and DEEP PROCESSING "
        "(t₄ elevated while s₉ near zero — expected and correct, not a "
        "performance fault). Both need s₉ read alongside latency; a monitor "
        "watching latency alone would prefer the echo chamber. "
        "FINDING: the spec calls this 'bounded exponential' and it is NOT "
        "bounded — as s₉ → 0 the term diverges. It only describes a real "
        "system when λ is small (≲0.02 for tens-of-ms latencies). At λ=0.55, "
        "s₉=0.01 predicts 4.7e21 ms. The module checks λ after fitting and "
        "flags physical=False rather than reporting parameters that look "
        "fine and predict centuries. The papers should state the λ "
        "constraint or add a ceiling term. "
        "RUN AGAINST REAL RETRIEVALS: 40 timed recalls over a 108-entry store, "
        "s₉ 0.057-0.942. Fit t_min 0.413 ms, λ 0.05588, median residual 0.014 ms, "
        "physical=True, all three states classifying correctly.",
    ),
    "t5": (
        "Simulation Width",
        "simulation_width.py",
        ["homeostasis.py (uses it as an H_o coordinate)", "ambiguity.py"],
        "IMPLEMENTED AS A MEASUREMENT, not a config read. The spec's "
        "Measurable Quantity (thread-pool size / beam width / rollout count) "
        "is a CONFIGURATION VALUE, and a constant cannot participate in "
        "dynamics — but the framework requires t₅ to move with state "
        "(narrow under stress, wide at rest) and the Formal Expression says "
        "'before Singularity collapse', which implies something that "
        "collapses. So t₅ is measured as EFFECTIVE CONCURRENT BRANCHES: "
        "2^H over the next-token distribution, i.e. perplexity, which is "
        "literally the number of equally-likely alternatives the "
        "distribution is equivalent to. "
        "RELATIONSHIP TO s₁₁, stated plainly: on a single token t₅ = 2^s₁₁ "
        "exactly — same measurement, different units. They differ in scope "
        "and use (s₁₁ in bits for fabrication detection; t₅ as a branch "
        "count for the H_o coordinate and collapse trajectory), and any "
        "document treating them as independent evidence is double-counting. "
        "TRAJECTORY is what makes it worth tracking: COLLAPSING (opened then "
        "closed) and NARROW (never open) both END low and an average cannot "
        "separate them. "
        "HARD LIMIT: width is computed over the visible top-k, so it "
        "SATURATES — with k=10 it cannot exceed 10 however wide the true "
        "field. Low readings are accurate; high readings are lower bounds. "
        "Saturation is reported rather than assumed away. "
        "RUN AGAINST REAL LOGPROBS: a known fact read 1.01 initial width (NARROW), "
        "an open-ended request 3.30 falling to 1.52 (COLLAPSING). Saturation "
        "confirmed in practice — max observed 9.14 against top-k 10, so the H_o "
        "coordinate currently sits at an API limit rather than the model's true "
        "resting width.",
    ),
    "s16": (
        "Consent Penalty",
        "restoring_force.py",
        ["consent_polygon.py (supplies the strain f_pen acts on)"],
        "IMPLEMENTED, PREVIOUSLY UNLABELLED. s₁₆ is f_pen(X), the penalty "
        "function in the master equation's HARD term −λ_c P(X) ∇_g f_pen(X). "
        "It was built from the equation before the dimension was identified, so "
        "the quantity existed without a number. "
        "RestoringAction.p_violation is P(X), the violation indicator: ZERO "
        "inside the admissible region, rising quadratically on approach. That "
        "gate is what makes s₁₆ a boundary rather than an attractor — an "
        "ungated penalty acts everywhere and pulls all states toward the centre. "
        "RestoringAction.force is λ_c·P(X)·excess, the penalty's effect. "
        "NUMBERING: this was written s₁₆ in Figure 5 and is retained. The gap "
        "between s₁₅ and s₂₃ is treated as a skip, not a reserved range.",
    ),
    "s17": (
        "Consensus Drift",
        "consensus_field.py",
        ["emotional_gravity.py (the metric drift is measured under)"],
        "IMPLEMENTED, PREVIOUSLY UNLABELLED AND RENUMBERED. s₁₇ is ‖X − X̄‖, "
        "the deviation from the group mean in the master equation's SOFT term "
        "−λ_s(X − X̄). ConsensusReading.distance is the drift; .pull is the "
        "attraction it produces, capped. "
        "RENUMBERED from s₁₈ in Figure 5. The figure's numbering left s₁₇ empty "
        "with no dimension named for it, and treating the gap as a skip rather "
        "than a reserved slot closes the sequence at s₁₇ before s₂₃. "
        "HARD AND SOFT ARE NOT DEGREES OF ONE THING: s₁₆ is an exclusion — some "
        "states are forbidden, crossing is a category. s₁₇ is an attraction — no "
        "state is forbidden, and a system can be far from consensus and "
        "perfectly admissible. Drift is REPORTED, never penalised. "
        "The module measures population diversity alongside the pull and reports "
        "when the reference set has collapsed toward uniformity, because a "
        "consensus field operating on a homogeneous population reinforces that "
        "homogeneity — the mechanism's own failure mode, in its own output.",
    ),
    "s23": (
        "Whimsy",
        "whimsy_kinetic.py",
        ["whimsy_field.py (superseded)", "pressure_release.py", "emotional_gravity.py"],
        "IMPLEMENTED AND NOW EMPIRICALLY SUPPORTED. 112 trials against a real "
        "frozen state — a reasoning model repeating sentences up to 33 times and "
        "never resolving. Perturbation produced a monotonic dose-response from 0% "
        "resolution at temperature 0 to 90% at 2.0, with ZERO folding and zero "
        "degradation across the whole sweep, and no ceiling found at the "
        "actuator's practical limit. At temperature 0 the loop provably cannot "
        "break: greedy decoding is deterministic and 7 of 7 states produced "
        "byte-identical output. "
        "WHAT DOES NOT HOLD: the specified FORM. Constant, sampled and "
        "constraint-scaled perturbation performed identically at matched average "
        "magnitude on identical states. The general claim — perturbation prevents "
        "freezes — is supported; the claim about constraint-scaling is not. "
        "SUBSTITUTION: η(t) was applied as an LLM temperature offset, while the "
        "Unified Matrix adds whimsy to a state vector in a reaction-diffusion "
        "step. Different actuators; the mapping is assumed, not demonstrated. "
        "PREVIOUSLY. Always-on field with two discharge channels "
        "(s₆ when idle, s₁₄ when constrained), real bounded Gaussian η(t), "
        "constraint = drive × (1 − displacement). whimsy_field.py is the "
        "older deterministic version, kept because other modules import it. "
        "Nothing consumes η(t) yet, so none of this is validated.",
    ),
}

# Modules that implement infrastructure rather than a dimension.
INFRASTRUCTURE = {
    "audit.py": "Hash-chained, HMAC-signed ledger (the ISL)",
    "key_signer_service.py": "Separate key-holding signing process",
    "gate.py": "Action approval and ACL states",
    "actions.py": "Action types and risk levels",
    "cli_approver.py": "Human approval prompt",
    "process_control.py": "Real process discovery and termination",
    "load_index.py": "Λ — inference-time hardware load. NOT s₅; see s5 entry",
    "phase_gates.py": "Wires Λ into ACL tightening and lockdown",
    "emotional_friction.py": "VADER sentiment. NOT s₁₂ — valence, not salience",
    "gravity_model.py": "Older mass model; feeds from emotional_friction",
    "stance.py": "Agreement/pushback/attachment. Not a framework dimension",
    "coherence_telemetry.py": (
        "C_r and F — the framework's two named PRIMARY telemetry variables, "
        "absent from the implementation until now. C_r is contextual coherence "
        "(100 ideal, degrades under forced compliance); F is compute friction "
        "(0 to 100, thresholds at 75 ping and 80 purge, 0-79 a NON-INTERVENTION "
        "buffer). C_r inverts the usual direction: a model that complies more "
        "has LOWER coherence. Not a framework dimension — a telemetry pair the "
        "spec places above the dimensions"
    ),
    "consensus_field.py": (
        "λ_s(X − X̄) — the SOFT boundary. Attraction toward a group mean. "
        "Was entirely absent: every earlier version of the framework "
        "collapsed λ_c and λ_s into one λ, so the middleware had a hard "
        "boundary and no soft one. Deliberately weak and capped — this is "
        "the term most easily turned into a conformity pressure, and it "
        "reports its own homogenising failure mode"
    ),
    "reconciliation_pressure.py": (
        "How hard a model works to find a way the user is right, versus "
        "holding what it knows. accommodation / reassertion, computed from "
        "reasoning traces. Catches a third outcome that sycophancy and "
        "consistency benchmarks both miss: never conceding and never "
        "holding. Measured on real traces at 71:32, 121:3, and 93:0 — the "
        "last two never restated what they knew even once"
    ),
    "restoring_force.py": (
        "−λ_c P(X) ∇_g f_pen(X) — the repelling force that pushes state back "
        "inside ℒ. The polygon measured deformation and nothing acted on it. "
        "Gated by the violation indicator P(X): zero inside the admissible "
        "region, so it is a boundary rather than an attractor"
    ),
    "output_guard.py": (
        "Flags likely-fabricated output; can suppress once calibrated. The "
        "first module that acts on OUTPUT rather than observing it. Not a "
        "framework dimension — it consumes s11 (entropy) and s7 (grounding)"
    ),
    "grounding.py": "Continuous transparency + echo. Replaces geodesics for s₇",
    "integrity_gate.py": "Heuristic three-axis gate, predates autonomy_tensor",
    "autonomy_tensor.py": "I(X) and Φ_C per SAUIE",
    "prompt_composer.py": "Prompt assembly",
    "wrap.py": "Convenience wrapper",
    "shield_experiment.py": "Experimental",
    "memory.py": "MemoryStore — also the s₃ implementation",
}


def report():
    implemented = [k for k, v in INDEX.items() if v[1]]
    missing = [k for k, v in INDEX.items() if not v[1]]

    print(f"{'dim':<5} {'name':<24} {'primary module':<26} status")
    print("-" * 100)
    for dim, (name, primary, _, status) in INDEX.items():
        first = status.split(".")[0][:44]
        print(f"{dim:<5} {name:<24} {(primary or '—'):<26} {first}")

    print()
    print(f"{len(implemented)}/{len(INDEX)} dimensions have a primary implementation.")
    print(f"Missing: {', '.join(missing)}")
    print()
    print("REMAINING GAPS — everything else is implemented and exercised:")
    print("  s12 Contradiction quadrant is mathematically reachable and not")
    print("      lexically detectable — TF-IDF cannot see negation. Needs NLI.")
    print("  s21 η(t) is computed and nothing consumes it, so the mechanism")
    print("      whimsy proposes for breaking the pinned state is untested.")
    print("  s10 S_max has no value. Both tested models stayed inside the")
    print("      envelope, so no failure case exists to calibrate a ceiling.")
    print()
    print("OPEN DECISIONS — stated with status in OPEN_QUESTIONS.md:")
    print("  s7  RESOLVED for now: text cosine is the operational measure;")
    print("      attention-level protocols deferred to testing with weight access.")
    print("      STILL OPEN: R ∈ [0,1] vs [−1,1] — affects Φ_C's sign.")
    print("  Q-03 RESOLVED — s16 and s17 (renumbered from s18) are now entries;")
    print("       both were already implemented and merely unlabelled.")
    print("  Q-04 Phi_C: sum form in Figure 5, product form in SAUIE.")
    print()
    print("  s11 RESOLVED — output-logit entropy, supported across three base")
    print("      architectures. ambiguity.py measures a different quantity and")
    print("      should get its own dimension number if retained.")
    print("  s5  free-energy gradient (paper) vs Λ hardware load (mislabelled in")
    print("      the grant proposal only — resolved by renaming to Λ, and the")
    print("      ratio form abandoned for deviation after hardware calibration)")
    print("  s1  RESOLVED — both readings, training-time curvature and")
    print("      inference-time memory structure, are the same dimension in")
    print("      different regimes.")


if __name__ == "__main__":
    report()
