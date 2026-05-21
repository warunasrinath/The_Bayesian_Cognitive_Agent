# agents/belief_state.py
"""
Bayesian Belief State Engine
Core mathematical component: sequential belief updating.
Implements: p(θ_t | z_1:t) ∝ p(z_t | θ_t) × p(θ_t | z_1:t-1)
"""

from agents.cpip import (
    BELIEF_CLASSES, compute_entropy,
    normalized_entropy, uniform_prior
)


class BayesianBeliefState:
    """
    Maintains and updates probabilistic belief state
    across conversation turns.

    This is the cognitive memory of the framework —
    what the agent believes about user intent at every turn.
    """

    def __init__(self, initial_prior: dict,
                  temperature: float = 1.5):
        self.belief        = dict(initial_prior)
        self.initial_prior = dict(initial_prior)
        self.temperature   = temperature
        self.turn_count    = 0
        self.history       = []

    def _compute_likelihood(self, evidence: dict) -> dict:
        """
        p(z_t | θ): likelihood of observing this evidence
        given each possible belief state.

        Uses soft weights to preserve prior influence
        across multiple turns (temperature tempering).
        """
        slots      = evidence.get("slots",      {})
        intent     = evidence.get("intent",     "other")
        sentiment  = evidence.get("sentiment",  "neutral")
        confidence = evidence.get("confidence", 0.5)

        # Soft base likelihood
        L = {cls: 0.08 for cls in BELIEF_CLASSES}

        # Slot-based likelihood signals
        slot_signals = {
            "location":        ["location_request"],
            "time":            ["time_request"],
            "guests":          ["guests_request"],
            "seating":         ["seating_request"],
            "restaurant_name": ["name_request"],
        }
        for slot, val in slots.items():
            if val:
                for target in slot_signals.get(slot, []):
                    if target in L:
                        L[target] += 0.25

        # Intent signal weighted by confidence
        if intent in L and intent != "other":
            L[intent] += 0.30 * confidence

        # Sentiment signals
        if sentiment == "positive":
            for cls in ["confirmation", "confirm_booking"]:
                if cls in L:
                    L[cls] += 0.08
        if sentiment == "negative":
            if "rejection" in L:
                L["rejection"] += 0.10

        # Normalize to [0,1]
        max_l = max(L.values())
        if max_l > 0:
            L = {k: v/max_l for k, v in L.items()}

        return L

    def update(self, evidence: dict) -> dict:
        """
        Core Bayes update with temperature tempering.

        posterior ∝ likelihood^(1/τ) × prior

        Temperature τ > 1 softens likelihood to preserve
        prior influence — critical for CPIP to matter.

        Returns updated belief distribution.
        """
        self.turn_count += 1
        prior      = dict(self.belief)
        likelihood = self._compute_likelihood(evidence)

        # Temperature tempering
        tempered = {
            cls: prob ** (1.0 / self.temperature)
            for cls, prob in likelihood.items()
        }

        # Bayes rule
        posterior = {
            cls: tempered[cls] * prior[cls]
            for cls in BELIEF_CLASSES
        }

        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {
                k: round(v/total, 6)
                for k, v in posterior.items()
            }
        else:
            posterior = uniform_prior()

        self.belief = posterior

        # Record for trajectory visualization
        self.history.append({
            "turn":      self.turn_count,
            "evidence":  evidence,
            "prior":     prior,
            "posterior": dict(posterior),
            "entropy":   compute_entropy(posterior),
            "h_norm":    normalized_entropy(posterior),
            "dominant":  max(posterior, key=posterior.get),
            "confidence": max(posterior.values()),
        })

        return posterior

    def get_dominant(self) -> tuple:
        """Returns (dominant_intent, confidence)."""
        dom  = max(self.belief, key=self.belief.get)
        conf = self.belief[dom]
        return dom, conf

    def get_top_n(self, n: int = 3) -> list:
        """Returns top N beliefs as sorted list."""
        return sorted(
            self.belief.items(),
            key=lambda x: x[1], reverse=True
        )[:n]

    def get_belief_shift(self) -> dict:
        """Returns belief change from previous turn."""
        if len(self.history) < 2:
            return {}
        prev = self.history[-2]["posterior"]
        curr = self.history[-1]["posterior"]
        return {
            cls: round(curr[cls] - prev.get(cls, 0), 4)
            for cls in curr
            if abs(curr[cls] - prev.get(cls, 0)) > 0.005
        }

    def reset(self, new_prior: dict):
        """Reset for new conversation."""
        self.belief        = dict(new_prior)
        self.initial_prior = dict(new_prior)
        self.turn_count    = 0
        self.history       = []