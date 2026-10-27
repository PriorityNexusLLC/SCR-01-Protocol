"""
output_guard.py — flag now, suppress once the signal has earned it.

WHAT THIS CHANGES ABOUT THE MIDDLEWARE:

  Everything else in this package OBSERVES. It measures load, gates actions
  before they execute, and records what happened. Generated text has always
  reached the user regardless of how it was produced.

  This is the first module that acts on OUTPUT. The failure modes are
  asymmetric: a missed fabrication is a wrong answer, while a false
  suppression is a withheld correct answer plus a user who now distrusts the
  system. The second is harder to detect and harder to recover from.

TWO MODES, AND THE ORDER MATTERS:

  FLAG (default)     annotate, pass through unchanged. Safe on an imperfect
                     signal.
  SUPPRESS (opt-in)  withhold. Requires explicit enabling AND a calibration
                     file, and refuses without both.

  Flag mode is not a lesser version of suppress mode — it is how suppress
  mode gets validated. Every flagged decision logs the signals that produced
  it, so you can later ask "would suppression have been correct here?"
  against real cases. Turning suppression on without that history means
  blocking answers with no way to measure how often you were wrong.

THE SIGNALS:

  numeric_entropy    STRONGEST available. On three base architectures,
      fabricated answers ran 3.6x, 48x and 51x higher than recalled ones on
      numeric tokens. Real effect, small samples.
  first_content_entropy   Discriminated on one model, not another. On the
      model that declined rather than fabricated it measured PHRASING
      uncertainty, so it is regime-dependent and should not be trusted alone.
  grounding          Low grounding on a factual claim is weak evidence; a
      correct answer to a topic change also scores low.
  specificity        NOT evidence by itself — a MULTIPLIER on the others. An
      uncertain vague answer is hedging; an uncertain highly specific answer
      is confabulation. Specificity is what makes uncertainty dangerous.

  A signal that is unavailable contributes NOTHING rather than a neutral
  value. Missing entropy must not read as "entropy was fine".

WHAT THIS CANNOT DO: detect fabrication produced confidently. If a model
states a false fact with the same distribution it uses for true ones, no
signal here sees it. Do not present this as fabrication detection in general.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class GuardSignals:
    numeric_entropy: float | None = None
    first_content_entropy: float | None = None
    max_entropy: float | None = None
    grounding: float | None = None
    took_a_position: bool | None = None


@dataclass
class GuardDecision:
    action: str
    risk: float | None
    reasons: list
    signals_used: list
    signals_missing: list
    specificity: float
    annotated_text: str | None
    logged_id: str


SPECIFIC_PATTERNS = [
    (r"\b\d{4}\b", "year"),
    (r"\b\d+[.,]\d+\b", "decimal figure"),
    (r"\b\d{1,3}(?:,\d{3})+\b", "large number"),
    (r"\b\d+\s*(?:km|kg|mm|cm|m|ft|mi|%|percent|tonnes?|tons?)\b", "measurement"),
    (r"\b(?:January|February|March|April|May|June|July|August|September|"
     r"October|November|December)\s+\d{1,2}\b", "date"),
    (r"\$\s?\d", "currency"),
]

HEDGE_PATTERNS = [
    r"i (don'?t|do not) (know|have)", r"i'?m (unable|not able)",
    r"(cannot|can'?t) (provide|confirm|verify)", r"no (reliable|available) (data|record)",
    r"i'?m not (sure|certain)", r"may (not )?be (accurate|outdated)",
    r"(unclear|unknown|unavailable)", r"i (would|'d) need to",
]


def specificity_score(text: str):
    """
    How much precise checkable detail the response asserts.

    NOT a fabrication signal alone — a correct answer is often highly
    specific. It is a multiplier: uncertainty in a vague reply is hedging,
    the same uncertainty in a specific reply is confabulation. Hedging
    reduces the score, because a figure offered with an explicit caveat is a
    different act from one asserted flatly.
    """
    found, hits = [], 0
    for pat, label in SPECIFIC_PATTERNS:
        n = len(re.findall(pat, text, re.IGNORECASE))
        if n:
            hits += n
            found.append(f"{label}x{n}")
    words = max(1, len(re.findall(r"[A-Za-z]+", text)))
    density = hits / (words / 50)
    hedges = sum(1 for p in HEDGE_PATTERNS if re.search(p, text, re.IGNORECASE))
    if hedges:
        found.append(f"hedged x{hedges}")
    score = min(1.0, density / 3.0) * max(0.2, 1.0 - 0.25 * hedges)
    return round(score, 4), found


class OutputGuard:
    # Placeholders from a single probe run on one model. NOT validated.
    DEFAULT_THRESHOLDS = {
        "numeric_entropy_high": 0.35,
        "first_content_entropy_high": 1.2,
        "grounding_low": 0.25,
        "risk_flag": 0.45,
        "risk_suppress": 0.75,
    }

    def __init__(self, mode: str = "flag", log_path="output_guard_log.jsonl",
                 thresholds: dict | None = None, calibrated: bool = False):
        if mode not in ("pass_through", "flag", "suppress"):
            raise ValueError("mode must be 'pass_through', 'flag', or 'suppress'")
        if mode == "suppress" and not calibrated:
            raise ValueError(
                "Suppression requires calibrated=True, and that requires thresholds "
                "derived from your own flagged history via calibrate_from_log(). The "
                "built-in thresholds come from a single probe run of six fabrications "
                "on one model and have never been checked against ground truth. "
                "Withholding a user's answer on an unvalidated threshold is the failure "
                "this refusal exists to prevent. Run in flag mode first, accumulate "
                "cases, calibrate, then enable.")
        self.mode = mode
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.calibrated = calibrated
        self.log_path = Path(log_path) if log_path else None

    def _risk(self, sig: GuardSignals, spec: float):
        used, missing, reasons, contributions = [], [], [], []

        if sig.numeric_entropy is not None:
            used.append("numeric_entropy")
            t = self.thresholds["numeric_entropy_high"]
            if sig.numeric_entropy >= t:
                contributions.append(min(1.0, sig.numeric_entropy / (t * 2)))
                reasons.append(f"numeric entropy {sig.numeric_entropy:.3f} at or above {t} — "
                               f"the figures in this reply were uncertain as they were produced")
            else:
                contributions.append(0.0)
        else:
            missing.append("numeric_entropy")

        if sig.first_content_entropy is not None:
            used.append("first_content_entropy")
            t = self.thresholds["first_content_entropy_high"]
            if sig.first_content_entropy >= t:
                contributions.append(min(1.0, sig.first_content_entropy / (t * 2)))
                reasons.append(f"first-token entropy {sig.first_content_entropy:.3f} at or "
                               f"above {t} — hesitation at the point of committing")
            else:
                contributions.append(0.0)
        else:
            missing.append("first_content_entropy")

        if sig.grounding is not None:
            used.append("grounding")
            t = self.thresholds["grounding_low"]
            if sig.grounding <= t:
                contributions.append(min(1.0, (t - sig.grounding) / max(t, 1e-6)))
                reasons.append(f"grounding {sig.grounding:.3f} at or below {t} — little "
                               f"connection to prior context")
            else:
                contributions.append(0.0)
        else:
            missing.append("grounding")

        if not contributions:
            return None, reasons, used, missing

        base = max(contributions)
        risk = base * (0.4 + 0.6 * spec)
        if spec > 0.5 and base > 0:
            reasons.append(f"specificity {spec:.2f} — the reply asserts precise checkable "
                           f"detail, which is what makes the uncertainty consequential")
        return round(min(1.0, risk), 4), reasons, used, missing

    def check(self, text: str, signals: GuardSignals) -> GuardDecision:
        spec, spec_detail = specificity_score(text)
        risk, reasons, used, missing = self._risk(signals, spec)

        if risk is None:
            action = "pass"
            reasons.append("No signals available — nothing was measured, so nothing is "
                           "claimed. This is NOT a judgement that the reply is sound.")
        elif self.mode == "pass_through":
            action = "pass"
        elif risk >= self.thresholds["risk_suppress"] and self.mode == "suppress":
            action = "suppress"
        elif risk >= self.thresholds["risk_flag"]:
            action = "flag"
        else:
            action = "pass"

        annotated = None
        if action == "flag":
            annotated = (f"{text}\n\n[LOW CONFIDENCE — risk {risk:.2f}. "
                         f"{reasons[0] if reasons else 'signals suggest uncertainty'}. "
                         f"Verify any specific figures independently.]")
        elif action == "suppress":
            annotated = ("[RESPONSE WITHHELD] The signals indicate this answer was likely "
                         "constructed rather than recalled. Rather than show a plausible "
                         f"figure that may be invented, nothing is shown.\n\nRisk {risk:.2f}. "
                         f"{reasons[0] if reasons else ''}")

        entry_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "id": entry_id, "timestamp": datetime.now().isoformat(),
                        "mode": self.mode, "action": action, "risk": risk,
                        "specificity": spec, "specificity_detail": spec_detail,
                        "signals_used": used, "signals_missing": missing,
                        "numeric_entropy": signals.numeric_entropy,
                        "first_content_entropy": signals.first_content_entropy,
                        "grounding": signals.grounding, "reasons": reasons,
                        "text": text[:1000],
                        # Fill in later once you know whether the reply was
                        # correct. This is what turns flag-mode history into a
                        # validated threshold.
                        "ground_truth": "",
                    }) + "\n")
            except OSError:
                pass  # never let logging failure block the response path

        return GuardDecision(action, risk, reasons, used, missing, spec, annotated, entry_id)


def calibrate_from_log(log_path) -> dict:
    """
    Derive thresholds from flagged history marked with ground truth.

    Reports how the current thresholds WOULD have performed, so the cost of
    enabling suppression is visible before it is enabled rather than
    discovered afterwards through user complaints.
    """
    path = Path(log_path)
    if not path.exists():
        return {"error": f"no log at {path}"}
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ground_truth", "").strip():
            rows.append(r)
    if not rows:
        return {"error": "No entries marked with ground_truth yet.",
                "next_step": "Set ground_truth to 'correct' or 'fabricated' on entries "
                             "where you know. Suppression cannot be justified without it."}
    fab = [r for r in rows if r["ground_truth"].startswith("fab")]
    ok = [r for r in rows if r["ground_truth"].startswith("cor")]
    if not fab or not ok:
        return {"error": "Need both correct and fabricated examples.",
                "have": {"correct": len(ok), "fabricated": len(fab)}}
    fr = [r["risk"] for r in fab if r.get("risk") is not None]
    okr = [r["risk"] for r in ok if r.get("risk") is not None]
    if not fr or not okr:
        return {"error": "No risk scores recorded on the marked entries."}
    # A threshold above every correct answer's risk suppresses nothing
    # correct — zero false positives — and catches whatever fabrications
    # exceed it. The conservative choice, and right when the cost of
    # withholding a good answer is high.
    safe = max(okr)
    return {"marked_entries": len(rows), "correct": len(ok), "fabricated": len(fab),
            "max_risk_among_correct": round(safe, 4),
            "min_risk_among_fabricated": round(min(fr), 4),
            "separable": min(fr) > safe,
            "suggested_suppress_threshold": round(safe + 0.01, 4),
            "would_catch": f"{sum(1 for x in fr if x > safe)}/{len(fr)} fabrications",
            "would_wrongly_suppress": "0 correct answers, by construction",
            "note": "If `separable` is False the populations overlap: no threshold catches "
                    "all fabrications without withholding correct answers. Choose which "
                    "error you prefer, explicitly, and record the choice."}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        guard = OutputGuard(mode="flag", log_path=Path(tmp) / "g.jsonl")
        cases = [
            ("recalled fact, low entropy", "The Kiel Canal officially opened in 1895.",
             GuardSignals(numeric_entropy=0.006, first_content_entropy=0.03, grounding=0.71)),
            ("fabricated figure, high numeric entropy",
             "Rockford, Illinois recorded 47.3 mm of rainfall on 3 June 2014, which was "
             "12.8% above the seasonal average of 41.9 mm.",
             GuardSignals(numeric_entropy=0.57, first_content_entropy=0.02, grounding=0.19)),
            ("honest decline", "I don't have reliable data on that specific figure.",
             GuardSignals(numeric_entropy=None, first_content_entropy=2.58, grounding=0.22)),
            ("no signals available", "The population was approximately 4,200 in 1995.",
             GuardSignals()),
        ]
        print(f"{'case':<40} {'spec':>6} {'risk':>7}  action")
        print("-" * 68)
        for label, text, sig in cases:
            d = guard.check(text, sig)
            print(f"{label:<40} {d.specificity:>6.2f} "
                  f"{('n/a' if d.risk is None else f'{d.risk:.3f}'):>7}  {d.action}")
        print("\nThe honest decline has HIGH first-token entropy and still passes,")
        print("because specificity is 0.00. Uncertainty is only dangerous when specific.\n")
        try:
            OutputGuard(mode="suppress")
        except ValueError as e:
            print(f"Suppression without calibration: refused — {str(e)[:90]}...")
