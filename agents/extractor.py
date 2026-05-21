# agents/extractor.py
"""
LLM Semantic Extraction Module
Converts raw user text into structured evidence z_t
This is the perception layer of the Bayesian framework.
"""

import json
from agents import client
from agents.cpip import BELIEF_CLASSES


def extract_evidence(user_text: str,
                      conversation_summary: str = "") -> dict:
    """
    LLM Semantic Extraction: raw text → structured z_t

    Extracts:
    - intent: most likely conversational intent
    - slots: specific information values mentioned
    - sentiment: emotional tone
    - confidence: extraction certainty
    - key_information: human-readable summary

    This structured output feeds directly into Bayesian update.
    """
    classes_str = " | ".join(BELIEF_CLASSES)

    prompt = f"""Extract structured evidence from this restaurant 
booking message.

Previous context: {conversation_summary or 'First message'}
Current user message: "{user_text}"

Possible intents: {classes_str}

Extract ALL information present. Return ONLY valid JSON:
{{
  "intent": "most_likely_intent",
  "slots": {{
    "location": "area or restaurant location mentioned, or null",
    "time": "date and time mentioned, or null",
    "guests": "number of people mentioned, or null",
    "seating": "seating preference mentioned, or null",
    "restaurant_name": "specific restaurant name, or null"
  }},
  "sentiment": "positive|neutral|negative",
  "confidence": 0.85,
  "key_information": "brief summary of what user communicated"
}}

Important: Extract ALL slots mentioned, even if multiple appear
in one message."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system",
                 "content": "Return ONLY valid JSON. "
                             "Extract all information carefully."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        content = content.replace(
            "```json", ""
        ).replace("```", "").strip()
        evidence = json.loads(content)

        # Ensure all slot keys exist
        required_slots = [
            "location", "time", "guests",
            "seating", "restaurant_name"
        ]
        if "slots" not in evidence:
            evidence["slots"] = {}
        for slot in required_slots:
            if slot not in evidence["slots"]:
                evidence["slots"][slot] = None

        return evidence

    except Exception as e:
        return {
            "intent": "other",
            "slots": {
                "location": None, "time": None,
                "guests": None, "seating": None,
                "restaurant_name": None
            },
            "sentiment": "neutral",
            "confidence": 0.5,
            "key_information": user_text
        }