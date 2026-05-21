# agents/__init__.py
"""
Bayesian Cognitive Agent Package
Loads environment variables and creates shared OpenAI client.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (one level up from agents/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# Fallback: read from Streamlit Cloud secrets if env var not set
if not os.getenv("OPENAI_API_KEY"):
    try:
        import streamlit as st
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

# Shared OpenAI client — used by all agent modules
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
