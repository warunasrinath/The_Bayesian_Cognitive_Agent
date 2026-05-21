# agents/decision_layer.py
"""
Decision-Theoretic Action Selection Layer
Selects optimal action using expected utility maximisation.
Score(a) = Σ_θ p(θ) × I(a targets θ) × InfoGain(a)
"""

REQUIRED_SLOTS = ["location", "time", "guests"]

ACTIONS = [
    {
        "action_id":   "ask_location",
        "label":       "Ask for location",
        "response":    "Which area or neighbourhood are you "
                       "looking for?",
        "targets":     ["location_request"],
        "info_gain":   0.9,
        "slot":        "location"
    },
    {
        "action_id":   "ask_time",
        "label":       "Ask for time",
        "response":    "What date and time would you like "
                       "the reservation?",
        "targets":     ["time_request"],
        "info_gain":   0.9,
        "slot":        "time"
    },
    {
        "action_id":   "ask_guests",
        "label":       "Ask for party size",
        "response":    "How many people will be dining?",
        "targets":     ["guests_request"],
        "info_gain":   0.8,
        "slot":        "guests"
    },
    {
        "action_id":   "ask_seating",
        "label":       "Ask seating preference",
        "response":    "Do you have any seating preferences — "
                       "indoor, outdoor, or bar area?",
        "targets":     ["seating_request"],
        "info_gain":   0.6,
        "slot":        "seating"
    },
    {
        "action_id":   "ask_restaurant",
        "label":       "Ask restaurant name",
        "response":    "Do you have a particular restaurant "
                       "in mind?",
        "targets":     ["name_request"],
        "info_gain":   0.7,
        "slot":        "restaurant_name"
    },
    {
        "action_id":   "confirm_booking",
        "label":       "Confirm reservation",
        "response":    "I have all the details needed. "
                       "Shall I confirm your reservation now?",
        "targets":     ["confirm_booking", "confirmation"],
        "info_gain":   1.0,
        "slot":        None
    },
    {
        "action_id":   "offer_alternatives",
        "label":       "Offer alternatives",
        "response":    "I understand that does not work. "
                       "Would you like me to suggest "
                       "some alternatives?",
        "targets":     ["rejection"],
        "info_gain":   0.7,
        "slot":        None
    },
    {
        "action_id":   "clarify",
        "label":       "Request clarification",
        "response":    "Could you tell me a bit more about "
                       "what you are looking for?",
        "targets":     ["other"],
        "info_gain":   0.5,
        "slot":        None
    },
]


def select_action(belief: dict,
                   confirmed_slots: dict) -> dict:
    """
    Decision-theoretic action selection with required slot gate.

    Two rules applied in order:
    1. GATE: confirm_booking blocked until required slots filled
    2. SCORE: select action maximising expected utility

    Score(a) = Σ_θ p(θ) × 1[θ ∈ targets(a)] × InfoGain(a)
    """
    missing_required = [
        s for s in REQUIRED_SLOTS
        if s not in confirmed_slots
    ]

    all_scored = []

    for action in ACTIONS:
        action_id = action["action_id"]

        # Skip already confirmed slots
        slot = action.get("slot")
        if slot and slot in confirmed_slots:
            continue

        # GATE: block confirmation if required slots missing
        if action_id == "confirm_booking" and missing_required:
            continue

        # Compute expected utility score
        belief_mass = sum(
            belief.get(t, 0.0) for t in action["targets"]
        )
        score = belief_mass * action["info_gain"]

        all_scored.append({
            **action,
            "score":           round(score, 4),
            "belief_mass":     round(belief_mass, 4),
            "missing_required": missing_required,
            "reason": (
                f"Belief mass on {action['targets']}: "
                f"{belief_mass:.3f} × "
                f"InfoGain {action['info_gain']} = {score:.4f}"
                + (
                    f" | Still missing: {missing_required}"
                    if missing_required else
                    " | All required slots confirmed"
                )
            )
        })

    if not all_scored:
        # All slots confirmed — confirm booking
        return {
            "action_id":        "confirm_booking",
            "label":            "Confirm reservation",
            "response":         "I have everything I need. "
                                "Shall I confirm your reservation?",
            "score":            1.0,
            "targets":          ["confirm_booking", "confirmation"],
            "slot":             None,
            "missing_required": [],
            "reason":           "All required slots confirmed — "
                                "proceeding to confirmation"
        }

    return max(all_scored, key=lambda x: x["score"])


def score_all_actions(belief: dict,
                       confirmed_slots: dict) -> list:
    """
    Returns all actions with scores for visualization.
    Used in explainability panel.
    """
    results = []
    for action in ACTIONS:
        belief_mass = sum(
            belief.get(t, 0.0) for t in action["targets"]
        )
        score = belief_mass * action["info_gain"]
        slot  = action.get("slot")
        results.append({
            "action_id":  action["action_id"],
            "label":      action["label"],
            "score":      round(score, 4),
            "blocked":    (
                action["action_id"] == "confirm_booking" and
                any(s not in confirmed_slots
                    for s in REQUIRED_SLOTS)
            ) or (slot and slot in confirmed_slots),
        })
    return sorted(results, key=lambda x: x["score"],
                  reverse=True)