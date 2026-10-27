"""
emotional_friction.py

IMPORTANT NAMING NOTE: this was originally named emotional_gravity.py.
Renamed to emotional_friction.py because what it actually computes is
resistance/heaviness/drag on processing -- friction, not gravity (which
is about mass and attraction; see gravity_model.py for that instead).

A concrete, testable implementation of what was originally described as
'Emotional Gravity,' but which behaves like friction:

    "the weight on words/phrases/choices... positive and negative weights,
    where negative weighs more, gets heavy, dense"

This maps directly onto a real, well-studied effect: negativity bias
(negative stimuli are weighted more heavily than positive stimuli of equal
magnitude — see Baumeister et al. 2001, "Bad Is Stronger Than Good"; also
central to Kahneman & Tversky's loss aversion in prospect theory).

Implementation:
    1. Score each phrase's sentiment with a real, validated sentiment model
       (VADER — tuned specifically for short informal text).
    2. Apply asymmetric weighting: negative scores get multiplied by a
       "heaviness" factor k > 1; positive scores are left alone.
    3. Aggregate across a passage to get a single "friction" scalar --
       resistance/heaviness that a passage exerts, not attraction.

This is honest about what it is NOT yet:
    - It does not predict anything on its own. A weighted sentiment score is
      just a number until you define what it's supposed to predict (agent
      behavior change? user distress? something else) and test that
      correlation against real logged data.
    - "Negativity weighs more" is a *choice* of k, not a derived constant.
      Different domains will need different k, calibrated empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class PhraseWeight:
    text: str
    raw_sentiment: float   # VADER compound score, -1..+1
    weight: float          # after asymmetric weighting


def score_phrase(text: str, negativity_factor: float = 1.6) -> PhraseWeight:
    """
    Score a single phrase and apply asymmetric ("negative weighs more") weighting.

    negativity_factor: how much heavier negative sentiment counts, relative
    to positive of the same magnitude. 1.0 = symmetric (no bias). Values
    around 1.5-2.0 are consistent with the negativity-bias literature, but
    the "correct" value for your use case should be calibrated against data,
    not assumed.
    """
    raw = _analyzer.polarity_scores(text)["compound"]  # -1..+1
    weight = raw * negativity_factor if raw < 0 else raw
    return PhraseWeight(text=text, raw_sentiment=raw, weight=weight)


def emotional_friction(passage: str, negativity_factor: float = 1.6) -> dict:
    """
    Compute aggregate 'emotional friction' for a passage of text.

    Splits on sentence-ish boundaries, scores each phrase, and returns both
    the per-phrase breakdown and an aggregate. "Density" here = mean weight
    magnitude, i.e. how emotionally loaded the passage is on average,
    independent of direction.
    """
    import re

    phrases = [p.strip() for p in re.split(r"[.!?\n]+", passage) if p.strip()]
    if not phrases:
        return {"phrases": [], "total_friction": 0.0, "mean_friction": 0.0, "density": 0.0}

    scored = [score_phrase(p, negativity_factor) for p in phrases]
    total = sum(p.weight for p in scored)
    mean = total / len(scored)
    density = sum(abs(p.weight) for p in scored) / len(scored)

    return {
        "phrases": scored,
        "total_friction": total,
        "mean_friction": mean,   # negative = net-heavy passage, positive = net-light
        "density": density,      # how emotionally loaded overall, regardless of sign
    }


if __name__ == "__main__":
    examples = [
        "I love this, it's wonderful and everything worked perfectly.",
        "This is terrible. I hate it. Nothing works and it's completely broken.",
        "The meeting is at 3pm. Bring the report. Room 4B.",
        "I'm a little worried about the deadline, but the team is doing great work.",
    ]
    for ex in examples:
        result = emotional_friction(ex)
        print(f"\n{ex!r}")
        print(f"  mean_friction={result['mean_friction']:.3f}  density={result['density']:.3f}")
        for p in result["phrases"]:
            print(f"    {p.text!r}: raw={p.raw_sentiment:+.3f} weighted={p.weight:+.3f}")
