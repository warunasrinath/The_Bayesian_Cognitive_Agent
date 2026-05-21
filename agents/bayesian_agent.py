# agents/bayesian_agent.py
"""
Bayesian-Enhanced Cognitive Agent
Full framework: CPIP → Evidence → Belief → Decision → Response
"""

import json
from agents import client
from agents.cpip import (
    generate_cpip_prior, compute_entropy,
    normalized_entropy, DOMAIN_DESCRIPTION
)
from agents.extractor import extract_evidence
from agents.belief_state import BayesianBeliefState
from agents.decision_layer import select_action, score_all_actions


class BayesianAgent:
    """
    Full Bayesian-Enhanced Cognitive Framework Agent.

    Pipeline per turn:
    1. LLM extracts structured evidence z_t
    2. Bayesian update: p(θ_t|z_1:t) ∝ p(z_t|θ_t) × p(θ_t|z_1:t-1)
    3. Decision-theoretic action selection
    4. LLM generates response conditioned on belief + action
    5. Full reasoning trace logged for explainability
    """

    def __init__(self):
        self.belief_state      = None
        self.confirmed_slots   = {}
        self.conversation_hist = []
        self.reasoning_log     = []
        self.cpip_prior        = None
        self.cpip_reasoning    = ""
        self.cpip_reduction    = 0.0
        self.initialized       = False

    def initialize(self) -> dict:
        """
        CPIP initialization — call once per conversation.
        Returns initialization metadata for display.
        """
        prior, reasoning, reduction = generate_cpip_prior()
        self.cpip_prior     = prior
        self.cpip_reasoning = reasoning
        self.cpip_reduction = reduction
        self.belief_state   = BayesianBeliefState(prior)
        self.confirmed_slots   = {}
        self.conversation_hist = []
        self.reasoning_log     = []
        self.initialized       = True

        return {
            "prior":     prior,
            "reasoning": reasoning,
            "reduction": reduction,
            "entropy":   compute_entropy(prior),
            "h_norm":    normalized_entropy(prior),
        }

    def process_turn(self, user_text: str) -> dict:
        """
        Full pipeline for one conversation turn.
        Returns complete result including reasoning trace.
        """
        if not self.initialized:
            self.initialize()

        # Build conversation summary for context
        summary = " | ".join([
            f"{m['role']}: {m['content'][:40]}"
            for m in self.conversation_hist[-4:]
        ]) or "First turn"

        # STEP 1: LLM Evidence Extraction
        evidence = extract_evidence(user_text, summary)

        # STEP 2: Update confirmed slots from evidence
        slot_updates = {}
        for slot, val in evidence.get("slots", {}).items():
            if val and slot not in self.confirmed_slots:
                self.confirmed_slots[slot] = val
                slot_updates[slot] = val

        # STEP 3: Bayesian belief update
        prev_belief = dict(self.belief_state.belief)
        new_belief  = self.belief_state.update(evidence)
        belief_shift = self.belief_state.get_belief_shift()

        # STEP 4: Decision-theoretic action selection
        action        = select_action(new_belief,
                                       self.confirmed_slots)
        all_scores    = score_all_actions(new_belief,
                                           self.confirmed_slots)

        # Mark slot as being asked
        if action.get("slot"):
            slot = action["slot"]
            if slot not in self.confirmed_slots:
                self.confirmed_slots[slot] = "pending"

        # STEP 5: Generate response conditioned on belief + action
        response = self._generate_response(
            user_text, evidence, new_belief, action
        )

        # STEP 6: Update conversation history
        self.conversation_hist.extend([
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": response}
        ])

        # STEP 7: Build full reasoning trace
        dominant, confidence = self.belief_state.get_dominant()
        entropy  = compute_entropy(new_belief)
        h_norm   = normalized_entropy(new_belief)

        trace = {
            "turn":            self.belief_state.turn_count,
            "user_said":       user_text,
            "evidence":        evidence,
            "slot_updates":    slot_updates,
            "belief_before":   prev_belief,
            "belief_after":    new_belief,
            "belief_shift":    belief_shift,
            "action":          action,
            "all_scores":      all_scores,
            "response":        response,
            "confirmed_slots": dict(self.confirmed_slots),
            "dominant":        dominant,
            "confidence":      confidence,
            "entropy":         entropy,
            "h_norm":          h_norm,
            "top3":            self.belief_state.get_top_n(3),
        }
        self.reasoning_log.append(trace)

        return trace

    def _generate_response(self, user_text: str,
                             evidence: dict,
                             belief: dict,
                             action: dict) -> str:
        """
        Generate natural language response conditioned on
        current belief state and selected action.
        The LLM generates surface form — decision is pre-made.
        """
        dominant, confidence = max(
            belief.items(), key=lambda x: x[1]
        ), None
        if isinstance(dominant, tuple):
            dominant, confidence = dominant
        else:
            dominant   = max(belief, key=belief.get)
            confidence = belief[dominant]

        missing = action.get("missing_required", [])

        system_prompt = f"""You are a restaurant booking assistant.

COGNITIVE STATE:
- Current dominant intent: {dominant} ({confidence:.1%} confidence)
- Required action: {action['action_id']}
- Action reasoning: {action['reason']}
- Confirmed information: {json.dumps(self.confirmed_slots)}
- Still needed: {missing}

Generate a natural, warm, conversational response (1-2 sentences).
The action to take is already decided — just express it naturally.
Do NOT mention beliefs, probabilities, or system internals."""

        messages = self.conversation_hist[-4:] + [
            {"role": "user", "content": user_text}
        ]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=120,
                messages=[
                    {"role": "system",
                     "content": system_prompt}
                ] + messages
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return action["response"]

    def get_last_trace(self) -> dict:
        return self.reasoning_log[-1] if self.reasoning_log else {}

    def reset(self):
        self.__init__()