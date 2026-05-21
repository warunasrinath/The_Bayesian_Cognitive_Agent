# agents/pure_llm_agent.py
"""
Pure LLM Baseline Agent
Direct response generation — no Bayesian reasoning.
Research baseline for comparison with Bayesian framework.
"""

from agents import client

SYSTEM_PROMPT = """You are a helpful restaurant booking assistant 
in New York City. Help users book restaurant tables by collecting 
their preferences naturally. Ask for location, time, party size, 
and any other relevant details. Be warm and conversational."""


class PureLLMAgent:
    """
    Pure LLM agent — research baseline.

    Characteristics (intentional limitations for comparison):
    - No belief state maintained
    - No uncertainty quantification
    - No structured evidence extraction
    - No decision-theoretic action selection
    - No explainability — black box by nature
    """

    def __init__(self):
        self.conversation_hist = []
        self.turn_count        = 0
        self.initialized       = True

    def process_turn(self, user_text: str) -> dict:
        """Generate response directly from conversation history."""
        self.turn_count += 1
        self.conversation_hist.append(
            {"role": "user", "content": user_text}
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=150,
                messages=[
                    {"role": "system",
                     "content": SYSTEM_PROMPT}
                ] + self.conversation_hist[-8:]
            )
            response = resp.choices[0].message.content.strip()
        except Exception as e:
            response = f"I apologise, something went wrong: {e}"

        self.conversation_hist.append(
            {"role": "assistant", "content": response}
        )

        return {
            "turn":              self.turn_count,
            "user_said":         user_text,
            "response":          response,
            "belief_state":      None,
            "evidence":          None,
            "action":            None,
            "reasoning":         None,
            "can_explain":       False,
            "explainability_msg": (
                "The Pure LLM cannot explain its reasoning. "
                "No belief state is maintained. No probability "
                "distribution is tracked. Response is generated "
                "directly from conversation context using neural "
                "network pattern matching — not auditable reasoning."
            )
        }

    def reset(self):
        self.__init__()