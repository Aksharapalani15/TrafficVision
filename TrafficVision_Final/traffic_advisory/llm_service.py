"""llm_service.py - the ONLY file that knows about the LLM provider (Google Gemini).

To switch provider later, rewrite generate_text(prompt) -> str and keep the
same error class. Nothing else in the project needs to change.
Uses only the standard library (urllib), so there is no extra SDK to install.
"""
import json
import socket
import urllib.error
import urllib.request

try:
    from . import config
except ImportError:
    import config


class LLMServiceError(Exception):
    """Any problem talking to the LLM (no key, network, HTTP error, bad reply).
    Messages never contain the API key."""


def generate_text(prompt):
    """Send a prompt to Gemini and return the reply text."""
    api_key = config.get_api_key()
    if not api_key or api_key == "your_api_key_here":
        raise LLMServiceError("GEMINI_API_KEY is not set.")

    url = f"{config.GEMINI_BASE_URL}/{config.get_model()}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2,
                             "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": api_key},  # key goes in a header, not the URL
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.get_timeout()) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMServiceError(f"LLM API returned HTTP {exc.code}.") from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
        raise LLMServiceError("Could not connect to the LLM API.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise LLMServiceError("LLM API returned an unreadable response.") from None

    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMServiceError("LLM API response had an unexpected structure.") from None
    if not text:
        raise LLMServiceError("LLM API returned empty text.")
    return text
