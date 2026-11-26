"""advisory.py - the brain of the module.  Public function: generate_advisory().

Steps: validate -> build prompt -> call LLM -> check LLM reply -> (fallback if bad)
-> assemble the final JSON-friendly dictionary.

Design choice: congestion and priority are decided by CODE, not the LLM.
The LLM only writes the wording (observation, advisory, actions, prediction summary).
"""
import json
import re

import config
import llm_service
from validation import ValidationError, validate_traffic_data

LEVEL_RANK = {"Free": 0, "Moderate": 1, "Heavy": 2}
MAX_ACTIONS = 6

# If the LLM mentions events that are NOT in our input, we reject its reply.
FORBIDDEN_PATTERN = re.compile(
    r"\b(accidents?|crash\w*|collisions?|injur\w*|ambulances?|fires?|flood\w*|"
    r"rain\w*|storms?|weather|roadworks?|construction|breakdowns?|emergenc\w*)\b",
    re.IGNORECASE)


class InvalidLLMResponse(Exception):
    """The LLM replied, but not in the format/content we accept."""


# ----------------------------------------------------------------- prediction
def describe_prediction(clean):
    """Work out trend facts from validated data (done in code, not by the LLM)."""
    pred = clean["prediction"]
    if pred is None:
        return None
    diff = LEVEL_RANK[pred["predicted_congestion"]] - LEVEL_RANK[clean["congestion"]]
    return {
        "current": clean["congestion"],
        "predicted": pred["predicted_congestion"],
        "predicted_vehicle_count": pred["predicted_vehicle_count"],
        "trend": "increase" if diff > 0 else "decrease" if diff < 0 else "no_change",
        "significant_increase": diff >= 2,  # Free -> Heavy
    }


# --------------------------------------------------------------------- prompt
def build_prompt(clean, priority):
    c = clean["vehicle_counts"]
    lines = [
        "You are an assistant that helps traffic authorities by writing short "
        "advisory reports from traffic monitoring data.",
        "",
        "STRICT RULES",
        "1. Use ONLY the data in the sections below. Never invent accidents, weather, "
        "road closures, emergencies, locations, times or causes.",
        "2. Never give automatic signal-control commands. Every action is a SUGGESTION "
        "for human traffic authorities to review (use wording like 'Consider', "
        "'Review', 'Authorities may').",
        "3. Use simple, professional English.",
        "4. Choose 2 to 5 relevant actions. No duplicates, no irrelevant actions.",
        "5. Do not change the congestion level or priority given below.",
        "6. Mention predictions only if a PREDICTION section exists; otherwise set "
        "prediction_summary to null.",
        "",
        "TRAFFIC DATA",
        f"- cars: {c['cars']}",
        f"- buses: {c['buses']}",
        f"- trucks: {c['trucks']}",
        f"- two_wheelers: {c['two_wheelers']}",
        f"- total vehicles: {clean['total_vehicles']}",
        f"- current congestion: {clean['congestion']}",
        f"- priority: {priority}",
    ]
    p = describe_prediction(clean)
    if p:
        count = p["predicted_vehicle_count"]
        lines += ["", "PREDICTION",
                  f"- predicted congestion: {p['predicted']}",
                  f"- trend compared with current: {p['trend']}",
                  f"- predicted vehicle count: {count if count is not None else 'not provided'}"]
        if p["significant_increase"]:
            lines.append("- note: this is a significant increase in congestion level.")
    lines += [
        "", "OUTPUT FORMAT",
        "Return ONLY a JSON object (no markdown, no extra text) with these keys:",
        '{"traffic_observation": "1-2 sentences summarising the supplied counts and congestion",',
        ' "advisory": "2-3 sentences explaining the situation for authorities",',
        ' "recommended_actions": ["action 1", "action 2"],',
        ' "prediction_summary": "1-2 sentences, or null if there is no PREDICTION section"}',
    ]
    return "\n".join(lines)


# ----------------------------------------------------------- LLM reply checks
def _extract_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise InvalidLLMResponse("LLM reply was not valid JSON.")


def _text_field(data, key, max_len, required=True):
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidLLMResponse(f"LLM field '{key}' is missing or not text.")
    if len(value) > max_len:
        raise InvalidLLMResponse(f"LLM field '{key}' is too long.")
    return value.strip()


def parse_llm_response(raw_text, has_prediction):
    """Turn raw LLM text into a checked dict, or raise InvalidLLMResponse."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise InvalidLLMResponse("LLM reply was empty.")
    data = _extract_json(raw_text)
    if not isinstance(data, dict):
        raise InvalidLLMResponse("LLM reply was not a JSON object.")

    observation = _text_field(data, "traffic_observation", 500)
    advisory = _text_field(data, "advisory", 800)
    summary = _text_field(data, "prediction_summary", 500, required=has_prediction)
    if not has_prediction:
        summary = None  # never keep a prediction the input did not contain

    actions = data.get("recommended_actions")
    if not isinstance(actions, list) or not actions:
        raise InvalidLLMResponse("LLM 'recommended_actions' must be a non-empty list.")
    unique, seen = [], set()
    for item in actions:
        if not isinstance(item, str) or not item.strip() or len(item) > 250:
            raise InvalidLLMResponse("LLM returned an invalid action item.")
        key = re.sub(r"[^a-z0-9 ]", "", item.lower()).strip()
        if key not in seen:  # remove duplicates
            seen.add(key)
            unique.append(item.strip())
    unique = unique[:MAX_ACTIONS]

    for text in [observation, advisory, summary or ""] + unique:
        if FORBIDDEN_PATTERN.search(text):
            raise InvalidLLMResponse("LLM mentioned an event that is not in the input data.")

    return {"traffic_observation": observation, "advisory": advisory,
            "recommended_actions": unique, "prediction_summary": summary}


# ------------------------------------------------------------------- fallback
FALLBACK_ADVISORY = {
    "Free": "Traffic is currently classified as Free. Continue normal monitoring and "
            "watch for sudden changes in vehicle volume.",
    "Moderate": "Moderate congestion has been detected. Traffic authorities may review "
                "signal timings and continue monitoring the junction.",
    "Heavy": "Heavy congestion has been detected. Traffic authorities should review the "
             "situation and consider suitable traffic management measures.",
}
FALLBACK_ACTIONS = {
    "Free": ["Continue normal traffic monitoring.",
             "Maintain existing traffic management arrangements.",
             "Monitor for sudden increases in vehicle counts."],
    "Moderate": ["Review whether signal timing adjustments are appropriate.",
                 "Monitor vehicle accumulation at the junction.",
                 "Consider communicating alternative routes if congestion increases."],
    "Heavy": ["Review current traffic signal timings.",
              "Consider deploying traffic personnel to monitor the junction.",
              "Consider communicating suitable alternative routes to commuters.",
              "Review subsequent traffic observations to check whether congestion is improving."],
}


def build_fallback(clean):
    """Rule-based report used when the LLM is unavailable or unreliable."""
    c, level, total = clean["vehicle_counts"], clean["congestion"], clean["total_vehicles"]
    if total == 0:
        observation = f"No vehicles were reported in the supplied counts. Current congestion is classified as {level}."
        if level != "Free":
            observation += " The congestion level and the zero count appear inconsistent and may need checking."
    else:
        observation = (f"{total} vehicles were reported in total (cars: {c['cars']}, buses: {c['buses']}, "
                       f"trucks: {c['trucks']}, two-wheelers: {c['two_wheelers']}). "
                       f"Current congestion is classified as {level}.")
    advisory = FALLBACK_ADVISORY[level]
    actions = list(FALLBACK_ACTIONS[level])
    summary = None

    p = describe_prediction(clean)
    if p:
        count_text = (f" The predicted vehicle count is {p['predicted_vehicle_count']}."
                      if p["predicted_vehicle_count"] is not None else "")
        if p["trend"] == "increase":
            summary = (f"Congestion is predicted to change from {p['current']} to {p['predicted']}; "
                       f"traffic volume may increase.{count_text}")
            if p["significant_increase"]:
                summary += " This is a significant predicted increase."
            advisory += " Traffic volume may increase, so authorities should consider preparing additional monitoring arrangements."
            actions.append("Consider preparing additional monitoring arrangements ahead of the predicted increase.")
        elif p["trend"] == "decrease":
            summary = (f"Congestion is predicted to change from {p['current']} to {p['predicted']}; "
                       f"traffic volume may decrease.{count_text}")
            advisory += " Traffic volume may decrease; continue monitoring to confirm whether congestion is improving."
            actions.append("Review subsequent observations to confirm whether congestion is improving.")
        else:
            summary = f"Congestion is predicted to remain {p['predicted']}.{count_text}"
    return {"traffic_observation": observation, "advisory": advisory,
            "recommended_actions": actions[:MAX_ACTIONS], "prediction_summary": summary}


# ------------------------------------------------------------- public function
def _error_result(message):
    return {"congestion": None, "priority": None, "traffic_observation": None,
            "advisory": None, "recommended_actions": [], "prediction_summary": None,
            "status": "error", "source": None, "note": message}


def generate_advisory(traffic_data, prediction_data=None, use_llm=True, llm_call=None):
    """Create a traffic advisory report.

    Args:
        traffic_data: dict with cars, buses, trucks, two_wheelers, congestion.
        prediction_data: optional dict (predicted_congestion, predicted_vehicle_count,
            current_congestion). None means "no prediction".
        use_llm: False forces the offline fallback.
        llm_call: optional function(prompt)->str, used by tests to replace the real LLM.

    Returns a dict with: congestion, priority, traffic_observation, advisory,
    recommended_actions, prediction_summary, status ("success" | "error"),
    source ("llm" | "fallback"), note (reason for fallback / error text, else None).
    """
    try:
        clean = validate_traffic_data(traffic_data, prediction_data)
    except ValidationError as exc:
        return _error_result(str(exc))

    congestion = clean["congestion"]
    priority = config.PRIORITY_BY_CONGESTION[congestion]
    content, note = None, None

    if use_llm:
        caller = llm_call or llm_service.generate_text
        try:
            raw = caller(build_prompt(clean, priority))
            content = parse_llm_response(raw, has_prediction=clean["prediction"] is not None)
        except (llm_service.LLMServiceError, InvalidLLMResponse) as exc:
            note = f"Fallback used: {exc}"
        except Exception as exc:  # last safety net: never crash the dashboard
            note = f"Fallback used: unexpected LLM error ({type(exc).__name__})."
    else:
        note = "Fallback used: LLM disabled (offline mode)."

    source = "llm"
    if content is None:
        content, source = build_fallback(clean), "fallback"

    return {"congestion": congestion, "priority": priority, **content,
            "status": "success", "source": source, "note": note}
