# Scientific Domains Drawn On

Two lists, and the distinction between them matters more than either one alone.

**Part 1** is domains the measurements are *accountable to* — a specialist in that field could check the work against their own standards and find it right or wrong.

**Part 2** is domains that supplied *structure* — where a concept came from, without the measurement inheriting that field's verification standards.

Conflating the two is the most common way interdisciplinary work overclaims. A framework that borrows a word from thermodynamics has not thereby become physics.

\---

## Part 1 — Domains the measurements answer to

Each of these has an established method that could confirm or refute what was built.

### Information theory

Shannon entropy, used in three distinct places: s₁₁ ambiguity density (entropy of the output distribution), t₅ simulation width (2^H as effective branch count — the same quantity in different units), and s₉ sublimation rate. Perplexity as an interpretable branch count is standard here.

### Riemannian geometry

The metric tensor g, the norms ‖·‖\_g and gradients ∇\_g throughout, and the resolution that s₁₂ *is* g. The property that absolute scale is arbitrary while relative weighting carries meaning is what licensed normalising G to mean 1.0 — a geometry fact, not a convenience.

### Graph theory

Eigenvector centrality by power iteration (Bonacich), used for s₁₂'s salience weighting. Concept-graph construction and shortest-path traversal in `geodesics.py`. The substitution of concept-graph centrality for attention-graph centrality is documented as a substitution.

### Control theory and Lyapunov stability

s₁₅ convergence, the candidate function V(X) = ‖X − H\_o‖²\_g, LaSalle's invariance principle and its preconditions. This domain is why the convergence claim is marked **conjectured** rather than confirmed: LaSalle requires V̇ ≤ 0 along trajectories, which measurement cannot establish. Also supplies bang-bang control for the I(X) phase gate, and hysteresis for gate thresholds.

### Dynamical systems

Attractors, basins, fixed points, phase space. The distinction between *stopped* and *at rest* — a repetition attractor has step size near zero while distance from the anchor stays high — comes from here and is the reason s₁₅ needs two measures rather than one.

### Statistical mechanics and thermodynamics

Not as metaphor: s₁₄ radiant heat is measured as joules per token from real GPU power draw, because computation dissipates energy and that dissipation is instrumentable. Also the free-energy formulation F(X) inherited from predictive processing.

**Validated since this list was written.** Four generations on a real card gave 1.807–1.900 J/token with Q\_aux spanning +0.000 to +0.092. The measurement also exposed two calibration errors that no amount of reasoning would have found: a GPU that had not boosted its clocks drew half the power of subsequent runs and, because the baseline is a minimum, became the baseline; and a "settled idle" reading of 34.8 W on a card whose true floor is 2.55 W, which would have understated every energy figure by roughly half. Both produced plausible numbers with no error raised.

### Power and thermal instrumentation

Distinct enough from the above to name separately. NVML power sampling, clock-state settling behaviour, the difference between core and board power domains, and the fact that a discrete GPU takes materially longer than a minute to return to deep idle after activity. This domain is why s₁₄ is buildable at all — no GPU API exposes FLOP counts, and power is the instrumentable proxy for the same dissipation.

### Least-squares fitting and model adequacy

Beyond the mechanics of fitting: the question of whether a fitted model can describe a real system. t₄'s exponential is unbounded as s₉ → 0, so a fit returning a large λ produces a curve predicting latencies of 10²¹ ms. The module checks λ after fitting and reports `physical=False` rather than returning parameters that look well-conditioned and predict centuries. Also the choice of median over mean for residuals when a curve is steep near a singularity — one outlier there would otherwise dominate and report a fit far worse than the one obtained.

### Numerical analysis

Least-squares fitting on the linearised t₄ curve, overflow handling, the choice of median over mean for residuals when a curve is steep near a singularity, and the physicality check on λ. This domain is why t₄'s "bounded exponential" was caught as unbounded.

### Systems and performance engineering

t₁ micro-pulse (tokens/ms via `perf\\\\\\\_counter`), s₈ atmospheric drag (added latency per token), t₄ retrieval round-trip, Λ load index from GPU/CPU/thermal telemetry. Windowing to handle per-token sampling noise. NVML power and thermal instrumentation.

### Cryptography

SHA-256 hash chaining, HMAC signing, the separation of key custody from key use, and the specific demonstration that a hash chain alone is tamper-*evident* but not tamper-*resistant* — an attacker who recomputes the chain defeats it, and only a keyed signature does not.

### Operating systems

Process discovery by listening port and by name, signal handling, the SIGTERM-then-SIGKILL escalation, and the correction that terminating a process does not require having spawned it.

### Information retrieval

TF-IDF vectorisation, cosine similarity, s₇ refractive index and s₁₃ semantic depth. Also the domain in which several failures occurred — lexical measures missing semantic phenomena, four separate times.

### Bayesian statistics

Posterior entropy as the formal reading of s₁₁, and posterior *thickness* measured behaviourally as answer scatter across repeated samples when the posterior itself is inaccessible.

### Experimental design and psychometrics

This one is doing more work than its position in the list suggests. Pre-registration, manipulation checks, blinded scoring, control conditions, effect-size floors, small-sample guards, ceiling and floor effects, inter-rater reliability, and the distinction between a null result and an opposite result. The telemetry-injection experiment's validity rests entirely on its manipulation check having passed.

### Machine learning and NLP

Transformer inference, logprobs and top-k truncation, temperature and sampling, RLHF and instruction tuning, context windows, beam search. Also the source of the caution that comparing models across vendors varies many things at once — which is why the Llama/Hermes matched pair carries weight that the four-model comparison does not.

\---

### Multi-agent systems and consensus dynamics

The λ\_s term — attraction toward a group mean X̄ — is a mean-field coordination mechanism, and the literature on it is where the failure mode is documented: a consensus field operating on a homogeneous population reinforces that homogeneity. `consensus\\\\\\\_field.py` measures population diversity alongside the pull and reports when it drops below a floor, so the mechanism's own pathology is visible in its own output rather than requiring an external audit.

**Note this is the one domain where the framework's stated purpose and the mathematics pull against each other.** λ\_s pulling toward a mean IS a conformity pressure; the safeguards against it being used as one are structural — small default gain, a hard cap on the pull, and distance reported but never penalised.

\---

## Part 2 — Domains that supplied structure, not verification

These generated the framework's concepts. **The measurements do not answer to these fields**, and results here are not evidence about them.

### Cognitive science and psychology

The origin of nearly every dimension's *shape*: memory consolidation by salience, the idea that what is remembered is what connected to what was already held, contradiction driving further study, somatic regulation under stress, and performative compliance itself. None of the measurements test a psychological claim, and none of the results say anything about human cognition.

### Neuroscience

Predictive processing and free energy, which supply F(X)'s meaning as prediction error. Borrowed as formalism. Nothing here is evidence about brains.

### Physics — general relativity

The structural move where mass curves a metric and geodesics follow the curvature, reused as: salience curves g, and alignment is measured under it. Genuinely the same *mathematical* operation, and genuinely not a claim that anything physical is happening. Whether the analogy is apt is not testable by anything in this package.

### Astrogeology, planetary science, thermodynamic metaphor

Fossil floor, orbital vacuum, atmospheric drag, radiant heat, topography. Naming that made an abstract space thinkable. s₁₄ later turned out to be measurable *as* thermal dissipation, which is a coincidence worth noting and not a vindication of the metaphor.

### Materials science

Elastic limit and plastic deformation, which is what the consent polygon is: deformation that springs back versus deformation that does not. Structural, not literal.

### Electromagnetism

The always-on field with state-dependent discharge channels, which is what fixed the whimsy contradiction — one field, two outlets, routed by constraint. The code implements the *structure* the analogy implies. Whether it holds as physics is not something the code can check, and nothing depends on it.

\---

## Why the split is the important part

**Part 1 is where the work can be wrong.** A control theorist can check the Lyapunov claim. A cryptographer can check the ledger. A statistician can check the effect-size handling. That exposure is what makes the findings worth something.

**Part 2 is where the work came from.** The concepts are load-bearing for the *design* and carry no evidential weight. Emotional Gravity is a metric tensor built from recurrence and centrality; the name explains why anyone thought to build it, and contributes nothing to whether it measures what it claims.

An interdisciplinary project fails in a predictable way: borrowing vocabulary from a field and then implying the borrowed authority. Keeping these lists separate is the defence against that, and it is why every dimension in the derivation table carries a domain label — so a reviewer can go straight to the rows in their own field rather than judging the whole table by whichever row they understand least.

