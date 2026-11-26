# LLM Prompt Documentation

The prompt is built by `build_prompt()` in `advisory.py`. The LLM receives only validated data.

## Structure
1. **Role** – assistant helping traffic authorities write short advisories.
2. **Strict rules** – (1) use only supplied data; never invent accidents, weather, closures, emergencies, locations, times, causes; (2) no automatic signal-control commands, suggestions only ("Consider", "Review"); (3) simple professional English; (4) 2–5 relevant, non-duplicate actions; (5) don't change congestion/priority; (6) predictions only if a PREDICTION section exists, else `prediction_summary: null`.
3. **TRAFFIC DATA** – four counts, total, congestion, priority (computed by code).
4. **PREDICTION** (only when supplied) – predicted level, trend (increase/decrease/no_change, computed by code), predicted count or "not provided", and a note if Free→Heavy.
5. **OUTPUT FORMAT** – JSON object with `traffic_observation`, `advisory`, `recommended_actions`, `prediction_summary`.

## Settings
Temperature 0.2 (consistent output); `responseMimeType: application/json` asks Gemini for JSON.

## Example prompt (Heavy, no prediction)
```
TRAFFIC DATA
- cars: 120
- buses: 20
- trucks: 10
- two_wheelers: 80
- total vehicles: 230
- current congestion: Heavy
- priority: High
```
(preceded by the role and rules, followed by the output format.)

## Why these choices
- Code, not LLM, decides congestion/priority → exact and testable.
- Trend computed in code → LLM cannot misread which direction congestion is moving.
- Reply validated after the call (JSON shape, lengths, duplicates removed, banned event words, prediction forced null if absent) → bad replies trigger the fallback.

## Known prompt limits
Prompt rules reduce but do not eliminate wrong or off-topic output. Prompt wording was not tuned against live Gemini output in my environment; test it with your key and adjust if needed.
