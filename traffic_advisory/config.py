"""config.py - all settings in one place.

Secrets (the API key) come from environment variables / a local .env file.
They are never written in source code and never put in the output.
"""
import os
from pathlib import Path

try:  # python-dotenv reads the .env file into environment variables
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # module still works if dotenv is not installed
    pass

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Business rule from the project brief: congestion level -> priority.
PRIORITY_BY_CONGESTION = {"Free": "Low", "Moderate": "Medium", "Heavy": "High"}


# Functions (not constants) so tests can change environment variables.
def get_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_model():
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def get_timeout():
    try:
        return float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    except ValueError:
        return 20.0
