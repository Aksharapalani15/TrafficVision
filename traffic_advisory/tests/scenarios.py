"""The 13 required test scenarios, shared by pytest and run_scenarios.py.

The real Gemini API is NOT called in tests (tests must be fast, free and repeatable).
Instead a 'fake LLM' returns a prepared reply, so we test OUR code's behaviour.
"""
import json

GOOD_REPLY = json.dumps({
    "traffic_observation": "The supplied counts show the reported vehicle volume for this junction.",
    "advisory": "Authorities may review the situation based on the supplied congestion level.",
    "recommended_actions": ["Review traffic signal timings.",
                            "Consider monitoring the junction with traffic personnel.",
                            "review traffic signal timings"],   # duplicate on purpose
    "prediction_summary": "The forecast suggests a change in congestion level.",
})


def fake_good_llm(prompt):
    return GOOD_REPLY


def fake_failing_llm(prompt):
    from llm_service import LLMServiceError
    raise LLMServiceError("Could not connect to the LLM API.")


def fake_invalid_llm(prompt):
    return "Sorry, I think traffic is bad. <not json>"


LLMS = {"good": fake_good_llm, "failing": fake_failing_llm, "invalid": fake_invalid_llm}


def t(cars, buses, trucks, tw, congestion):
    return {"cars": cars, "buses": buses, "trucks": trucks,
            "two_wheelers": tw, "congestion": congestion}


# expect keys: status, congestion, priority, source, prediction (True=not null, False=null),
#              error_contains, observation_contains
SCENARIOS = [
    dict(id=1, name="Free congestion", traffic=t(15, 1, 0, 10, "Free"), prediction=None, llm="good",
         expect=dict(status="success", congestion="Free", priority="Low", source="llm", prediction=False)),
    dict(id=2, name="Moderate congestion", traffic=t(60, 8, 4, 40, "Moderate"), prediction=None, llm="good",
         expect=dict(status="success", congestion="Moderate", priority="Medium", source="llm", prediction=False)),
    dict(id=3, name="Heavy congestion", traffic=t(120, 20, 10, 80, "Heavy"), prediction=None, llm="good",
         expect=dict(status="success", congestion="Heavy", priority="High", source="llm", prediction=False)),
    dict(id=4, name="Zero vehicles (offline fallback)", traffic=t(0, 0, 0, 0, "Free"), prediction=None, llm="offline",
         expect=dict(status="success", congestion="Free", priority="Low", source="fallback",
                     prediction=False, observation_contains="No vehicles")),
    dict(id=5, name="Very high vehicle counts", traffic=t(5000, 800, 600, 4000, "Heavy"), prediction=None, llm="good",
         expect=dict(status="success", congestion="Heavy", priority="High", source="llm", prediction=False)),
    dict(id=6, name="Moderate now, Heavy predicted", traffic=t(60, 8, 4, 40, "Moderate"),
         prediction={"current_congestion": "Moderate", "predicted_congestion": "Heavy", "predicted_vehicle_count": 280},
         llm="good", expect=dict(status="success", congestion="Moderate", priority="Medium", source="llm", prediction=True)),
    dict(id=7, name="Heavy now, Moderate predicted", traffic=t(120, 20, 10, 80, "Heavy"),
         prediction={"current_congestion": "Heavy", "predicted_congestion": "Moderate", "predicted_vehicle_count": 90},
         llm="good", expect=dict(status="success", congestion="Heavy", priority="High", source="llm", prediction=True)),
    dict(id=8, name="Missing prediction (LLM wrongly returns a summary)", traffic=t(120, 20, 10, 80, "Heavy"),
         prediction=None, llm="good",
         expect=dict(status="success", congestion="Heavy", priority="High", source="llm", prediction=False)),
    dict(id=9, name="Invalid congestion level", traffic=t(10, 1, 1, 5, "Jammed"), prediction=None, llm="good",
         expect=dict(status="error", error_contains="Invalid congestion level")),
    dict(id=10, name="Negative vehicle counts", traffic=t(-5, 1, 1, 5, "Heavy"), prediction=None, llm="good",
         expect=dict(status="error", error_contains="cannot be negative")),
    dict(id=11, name="Missing input (None)", traffic=None, prediction=None, llm="good",
         expect=dict(status="error", error_contains="No traffic data")),
    dict(id=12, name="LLM API failure -> fallback", traffic=t(120, 20, 10, 80, "Heavy"), prediction=None, llm="failing",
         expect=dict(status="success", congestion="Heavy", priority="High", source="fallback", prediction=False)),
    dict(id=13, name="Invalid LLM response -> fallback", traffic=t(60, 8, 4, 40, "Moderate"), prediction=None, llm="invalid",
         expect=dict(status="success", congestion="Moderate", priority="Medium", source="fallback", prediction=False)),
]


def run_scenario(sc):
    """Run one scenario; return (result, llm_call_count)."""
    from advisory import generate_advisory
    calls = {"n": 0}
    if sc["llm"] == "offline":
        return generate_advisory(sc["traffic"], sc["prediction"], use_llm=False), 0

    def counting(prompt):
        calls["n"] += 1
        return LLMS[sc["llm"]](prompt)
    return generate_advisory(sc["traffic"], sc["prediction"], llm_call=counting), calls["n"]


def check(sc, result):
    """Return a list of problems (empty list = PASS)."""
    e, problems = sc["expect"], []
    for key in ("status", "congestion", "priority", "source"):
        if key in e and result.get(key) != e[key]:
            problems.append(f"{key}: expected {e[key]!r}, got {result.get(key)!r}")
    if "prediction" in e:
        has = result.get("prediction_summary") is not None
        if has != e["prediction"]:
            problems.append(f"prediction_summary not-null expected {e['prediction']}, got {has}")
    if "error_contains" in e and e["error_contains"] not in (result.get("note") or ""):
        problems.append(f"error text should contain {e['error_contains']!r}")
    if "observation_contains" in e and e["observation_contains"] not in (result.get("traffic_observation") or ""):
        problems.append(f"observation should contain {e['observation_contains']!r}")
    if result.get("status") == "success":
        acts = result["recommended_actions"]
        if not acts or len({a.lower() for a in acts}) != len(acts):
            problems.append("actions empty or duplicated")
    return problems
