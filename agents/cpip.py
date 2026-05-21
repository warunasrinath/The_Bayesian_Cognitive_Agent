# agents/cpip.py
"""
Cognitive Prior Initialization Protocol (CPIP)
Core research contribution: LLM-bootstrapped Bayesian priors
for cold-start conversational deployment.
"""

import json
import math
from agents import client

BELIEF_CLASSES = [
    "location_request",
    "time_request",
    "guests_request",
    "seating_request",
    "name_request",
    "confirm_booking",
    "confirmation",
    "rejection",
    "other"
]

DOMAIN_DESCRIPTION = (
    "Restaurant table reservation assistant helping diners book "
    "tables at restaurants in New York City. Collects location "
    "preference, date and time, number of guests, seating "
    "preference, and restaurant name."
)


def uniform_prior() -> dict:
    """Maximum entropy prior — complete ignorance."""
    n = len(BELIEF_CLASSES)
    return {cls: round(1/n, 6) for cls in BELIEF_CLASSES}


def validate_prior(prior: dict, epsilon: float = 0.01) -> dict:
    """
    Ensure mathematically valid probability distribution.
    Prevents zero-probability collapse in Bayesian updates.
    """
    fixed = {
        cls: max(prior.get(cls, epsilon), epsilon)
        for cls in BELIEF_CLASSES
    }
    total = sum(fixed.values())
    return {k: round(v/total, 6) for k, v in fixed.items()}


def generate_cpip_prior() -> tuple:
    """
    CPIP Stage 2: LLM-bootstrapped prior generation.

    Uses LLM world knowledge to estimate domain-informed
    initial belief distribution before any conversation starts.

    Returns:
        (prior_dict, reasoning_str, entropy_reduction_pct)
    """
    classes_str = "\n".join(
        [f"  - {cls}" for cls in BELIEF_CLASSES]
    )

    prompt = f"""You are a Bayesian prior estimation system.

A restaurant booking assistant is starting a NEW conversation.
No user message has been received yet.

The possible user intent states are:
{classes_str}

Based on how restaurant booking conversations typically BEGIN,
estimate the probability of each intent appearing in the
FIRST user message.

Consider:
- Users typically open with location, time, or general intent
- Confirmation/rejection rarely appear at conversation start
- "other" covers greetings and vague openers

Rules:
- ALL probabilities must be > 0 (never assign 0)
- Probabilities must sum to exactly 1.0

Return ONLY this JSON:
{{
  "probabilities": {{
    "location_request": 0.0,
    "time_request": 0.0,
    "guests_request": 0.0,
    "seating_request": 0.0,
    "name_request": 0.0,
    "confirm_booking": 0.0,
    "confirmation": 0.0,
    "rejection": 0.0,
    "other": 0.0
  }},
  "reasoning": "one sentence explaining the dominant choices"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system",
                 "content": "Return ONLY valid JSON. "
                             "All values > 0. Sum = 1.0."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        content = content.replace(
            "```json", ""
        ).replace("```", "").strip()
        result    = json.loads(content)
        probs     = result.get("probabilities", {})
        reasoning = result.get(
            "reasoning", "LLM-estimated domain prior"
        )
        prior     = validate_prior(probs)

        # Compute entropy reduction vs uniform
        uni      = uniform_prior()
        uni_h    = compute_entropy(uni)
        cpip_h   = compute_entropy(prior)
        reduction = round((1 - cpip_h/uni_h) * 100, 1)

        return prior, reasoning, reduction

    except Exception as e:
        print(f"CPIP generation failed: {e} — using uniform")
        uni = uniform_prior()
        return uni, "Uniform fallback (API error)", 0.0


def compute_entropy(belief: dict) -> float:
    """Shannon entropy H(θ) = -Σ p log₂ p"""
    h = 0.0
    for p in belief.values():
        if p > 1e-10:
            h -= p * math.log2(p)
    return round(h, 4)


def normalized_entropy(belief: dict) -> float:
    """Normalized entropy ∈ [0,1]. 1.0 = maximum uncertainty."""
    n     = len(belief)
    max_h = math.log2(n) if n > 1 else 1.0
    return round(compute_entropy(belief) / max_h, 4)