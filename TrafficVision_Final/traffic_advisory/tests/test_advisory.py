import json
import urllib.error

import pytest

import config
import llm_service
from advisory import generate_advisory, parse_llm_response, InvalidLLMResponse
from scenarios import SCENARIOS, check, run_scenario, GOOD_REPLY
from validation import validate_traffic_data, ValidationError

HEAVY = {"cars": 120, "buses": 20, "trucks": 10, "two_wheelers": 80, "congestion": "Heavy"}


# ---- the 13 required scenarios (parametrised) ----
@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: f"{s['id']:02d}-{s['name']}")
def test_scenario(sc):
    result, llm_calls = run_scenario(sc)
    assert check(sc, result) == []
    if result["status"] == "error":
        assert llm_calls == 0, "LLM must not be called for invalid input"


# ---- extra behaviour tests ----
def test_duplicate_actions_removed():
    r = generate_advisory(HEAVY, llm_call=lambda p: GOOD_REPLY)
    assert len(r["recommended_actions"]) == 2


def test_llm_reply_in_markdown_fence_is_accepted():
    fenced = "```json\n" + GOOD_REPLY + "\n```"
    assert generate_advisory(HEAVY, llm_call=lambda p: fenced)["source"] == "llm"


def test_llm_missing_keys_triggers_fallback():
    r = generate_advisory(HEAVY, llm_call=lambda p: json.dumps({"advisory": "x"}))
    assert r["source"] == "fallback" and r["status"] == "success"


def test_llm_invented_accident_is_rejected():
    bad = json.dumps({"traffic_observation": "An accident has blocked the road.",
                      "advisory": "Authorities may review.", "recommended_actions": ["Review timings."],
                      "prediction_summary": None})
    assert generate_advisory(HEAVY, llm_call=lambda p: bad)["source"] == "fallback"


def test_unexpected_exception_in_llm_still_returns_report():
    def boom(prompt):
        raise RuntimeError("boom")
    r = generate_advisory(HEAVY, llm_call=boom)
    assert r["status"] == "success" and r["source"] == "fallback"


def test_missing_api_key_uses_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    r = generate_advisory(HEAVY)
    assert r["source"] == "fallback" and "not set" in r["note"]


def test_api_key_never_in_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    def fake_urlopen(*a, **k):
        raise urllib.error.URLError("network down")
    monkeypatch.setattr(llm_service.urllib.request, "urlopen", fake_urlopen)
    r = generate_advisory(HEAVY)
    assert "SECRET-KEY-123" not in json.dumps(r)
    assert r["source"] == "fallback"


def test_fallback_mentions_significant_increase():
    r = generate_advisory({"cars": 5, "buses": 0, "trucks": 0, "two_wheelers": 3, "congestion": "Free"},
                          {"predicted_congestion": "Heavy"}, use_llm=False)
    assert "significant" in r["prediction_summary"]
    assert r["priority"] == "Low"  # priority still follows CURRENT congestion


def test_congestion_is_case_insensitive():
    assert validate_traffic_data({**HEAVY, "congestion": " heavy "})["congestion"] == "Heavy"


def test_string_count_rejected():
    r = generate_advisory({**HEAVY, "cars": "120"}, use_llm=False)
    assert r["status"] == "error" and "cars" in r["note"]


def test_missing_field_reported():
    bad = {k: v for k, v in HEAVY.items() if k != "buses"}
    assert "Missing required field 'buses'" in generate_advisory(bad, use_llm=False)["note"]


def test_prediction_mismatch_rejected():
    r = generate_advisory(HEAVY, {"current_congestion": "Free", "predicted_congestion": "Heavy"}, use_llm=False)
    assert r["status"] == "error" and "does not match" in r["note"]


def test_empty_prediction_dict_means_no_prediction():
    assert generate_advisory(HEAVY, {}, use_llm=False)["prediction_summary"] is None


def test_priority_mapping():
    assert [config.PRIORITY_BY_CONGESTION[k] for k in ("Free", "Moderate", "Heavy")] == ["Low", "Medium", "High"]


def test_parse_rejects_empty_text():
    with pytest.raises(InvalidLLMResponse):
        parse_llm_response("", has_prediction=False)


# ---- llm_service error mapping (network mocked, no real calls) ----
def test_llm_service_no_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(llm_service.LLMServiceError):
        llm_service.generate_text("hi")


def test_llm_service_http_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    def fake(*a, **k):
        raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
    monkeypatch.setattr(llm_service.urllib.request, "urlopen", fake)
    with pytest.raises(llm_service.LLMServiceError, match="429"):
        llm_service.generate_text("hi")


def test_llm_service_parses_good_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}).encode()

    class FakeResp:
        def read(self): return body
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(llm_service.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    assert llm_service.generate_text("hi") == "hello"


def test_llm_service_bad_structure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    class FakeResp:
        def read(self): return b'{"unexpected": 1}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(llm_service.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    with pytest.raises(llm_service.LLMServiceError, match="unexpected structure"):
        llm_service.generate_text("hi")
