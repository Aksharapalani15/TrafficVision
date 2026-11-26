# Short Explanation for the Final Presentation (≈2 minutes)

"My module is the AI Advisory part of TrafficVision. It takes the vehicle counts and congestion level from the other modules, plus an optional prediction, and produces a report for traffic authorities: an observation, an advisory, a list of recommended actions, a prediction summary and a priority level.

First, **validation** rejects missing, negative or non-numeric counts and invalid congestion levels. Then I build a **prompt** containing only the supplied data and send it to the **Gemini LLM**. The reply must be valid JSON; I check its format, remove duplicate actions, and reject text that mentions events not in the data, like accidents or weather.

Two design choices matter. **Priority is calculated by code**, not the AI: Free is Low, Moderate is Medium, Heavy is High. And if the AI is unavailable or gives a bad answer, a **rule-based fallback** still produces a complete report, so the dashboard never breaks.

All actions are **suggestions for human review**, not automatic signal control. I tested it with 13 scenarios and 32 automated pytest tests, all passing with a simulated LLM. Teammates integrate it with one call: `generate_advisory(traffic_data, prediction_data)`."

## Likely questions
- *Why not let the LLM decide priority?* Consistency and testability.
- *What if the API fails?* Fallback report, `source = "fallback"`.
- *How do you prevent hallucination?* Constrained prompt, only supplied data, output checks, fallback; it reduces but cannot fully remove the risk.
- *Why Gemini?* Free tier, simple REST/JSON; replaceable via `llm_service.py`.
- *Was the real API tested?* Be honest: automated tests use a simulated LLM; live-API checking is done on your machine with your key.
