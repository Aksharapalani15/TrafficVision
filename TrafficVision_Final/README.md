# TrafficVision

An integrated, interactive traffic-monitoring and analytics Streamlit application combining:
1. **Member 1 (`vehicle_detection`)**: YOLO11n vehicle detection (`car`, `motorcycle`, `bus`, `truck`).
2. **Member 2 (`vehicle_counting`)**: Multi-frame vehicle tracking, unique ID assignment, and category-wise counting.
3. **Member 3 (`traffic_analysis`)**: Congestion classification (`LOW`, `MODERATE`, `HIGH`), interval timelines, and moving-average forecasting.
4. **Member 5 (`traffic_advisory`)**: Rule-based traffic advisory report (with optional Gemini module preserved).

---

## Dashboard Sections

- **🏠 Overview**: KPI metric cards for total unique vehicles, cars, buses, trucks, motorcycles, current traffic condition, average vehicles/frame, and most frequent vehicle category.
- **🎬 Video Analysis**: Upload a traffic video (or select the built-in sample `traffic.mp4`), track live frame-by-frame progress, play the annotated output video, and download it.
- **📊 Traffic Analytics**: Interactive vehicle-category filter, distribution donut chart, category-wise bar chart, interval volume timeline, condition summary, and simple moving-average next-interval forecast.
- **🗺️ Congestion Map**: Illustrative top-view road-junction congestion map with Green (`LOW`), Yellow (`MODERATE`), Orange (`HIGH`), and Red (`VERY HIGH`) zones, legend, and interval intensity heatmap.
- **🚦 Traffic Advisory**: Current condition summary, observation, advisory, and suggested management actions for human review (using Member 5's rule-based fallback).
- **📄 Reports**: Downloadable CSV vehicle-count report and JSON analysis report built strictly from actual pipeline outputs.

---

## Windows Setup (PowerShell)

1. Open PowerShell in `TrafficVision_Final`:
   ```powershell
   cd path\to\TrafficVision_Final
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks script activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)*

3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Run the Streamlit Dashboard (Windows)

```powershell
streamlit run app.py
```

---

## Traffic Condition Thresholds (Member 3)

$$\text{avg\_vehicles\_per\_frame} = \frac{\text{total\_frame\_detections}}{\text{frames\_processed}}$$

| Average Vehicles / Frame | Traffic Condition | Heatmap Zone | Member 5 Advisory Input |
|---|---|---|---|
| `< 5.0` | `LOW` | Green (`LOW`) | `Free` |
| `5.0` to `< 12.0` | `MODERATE` | Yellow (`MODERATE`) | `Moderate` |
| `12.0` to `< 20.0` | `HIGH` | Orange (`HIGH`) | `Heavy` |
| `>= 20.0` | `HIGH` | Red (`VERY HIGH`) | `Heavy` |

---

## Run Automated Tests

```powershell
python -m pytest tests vehicle_detection\tests traffic_advisory\tests -v
```
