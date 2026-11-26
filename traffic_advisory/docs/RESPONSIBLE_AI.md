# Limitations and Responsible AI Report

| Requirement | How it is handled |
|---|---|
| Advice only from supplied data | Prompt rule + banned-term check on LLM output (accident, weather, emergency, etc.) → fallback if triggered |
| No invented conditions/emergencies | Same checks; priority is fixed by congestion level, never "Critical/Emergency" |
| No automatic signal control | Prompt forbids it; all actions are phrased as suggestions; dashboard caption states human decision-making |
| Predictions not invented | `prediction_summary` is forced to `null` in code when no prediction is supplied |
| No personal data | Module accepts only five numeric/categorical fields; no images, faces or number plates are ever received or stored; nothing is logged to disk |
| API key safety | Read from `.env`/environment; sent in an HTTP header; error messages never include it; tested (`test_api_key_never_in_output`); `.env` is git-ignored |
| Human review | Output wording ("Consider", "Authorities may"); examples show a review caption |
| LLM limitations | Documented below |

## Limitations
- LLMs can produce plausible but wrong or generic text.
- Counts come from upstream detection (YOLO), which can miscount; this module trusts its input (it only flags obviously inconsistent cases such as zero vehicles with non-Free congestion in fallback mode).
- Word-based safety checks are crude (false positives and misses possible).
- Data sent to the LLM provider leaves the system: only aggregate counts, but deployments should confirm provider terms.
- Not validated with real traffic authorities or real traffic data.

## Recommended practice
Show the AI/fallback source on the dashboard, keep a human in the loop, review outputs periodically, and never connect this output directly to signal controllers.
