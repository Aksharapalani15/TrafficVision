"""Simple, explainable congestion analysis based on vehicle detection and tracking results.

Thresholds (average vehicles detected per frame):
- LOW      : avg_vehicles_per_frame < 5.0          (maps to advisory congestion "Free")
- MODERATE : 5.0 <= avg_vehicles_per_frame < 12.0  (maps to advisory congestion "Moderate")
- HIGH     : avg_vehicles_per_frame >= 12.0        (maps to advisory congestion "Heavy")

For the visual congestion map, HIGH is further distinguished into:
- HIGH      : 12.0 <= avg_vehicles_per_frame < 20.0 (Orange)
- VERY HIGH : avg_vehicles_per_frame >= 20.0        (Red)

Only measured counts and per-frame density are used; no speed, GPS, or unmeasured
traffic metrics are invented.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

LOW_THRESHOLD = 5.0
HIGH_THRESHOLD = 12.0
VERY_HIGH_THRESHOLD = 20.0

CONDITION_TO_ADVISORY = {
    "LOW": "Free",
    "MODERATE": "Moderate",
    "HIGH": "Heavy",
}

THRESHOLD_DOC = (
    f"Based on average detected vehicles per frame: "
    f"< {LOW_THRESHOLD:g} = LOW, "
    f"{LOW_THRESHOLD:g} to < {HIGH_THRESHOLD:g} = MODERATE, "
    f">= {HIGH_THRESHOLD:g} = HIGH."
)


def classify_traffic_condition(avg_vehicles_per_frame: float) -> Tuple[str, str]:
    """Return (dashboard_condition, advisory_congestion_label)."""
    if avg_vehicles_per_frame < LOW_THRESHOLD:
        condition = "LOW"
    elif avg_vehicles_per_frame < HIGH_THRESHOLD:
        condition = "MODERATE"
    else:
        condition = "HIGH"
    return condition, CONDITION_TO_ADVISORY[condition]


def get_congestion_zone(avg_vehicles_per_frame: float) -> Dict[str, str]:
    """Return 4-tier visual zone info (LOW, MODERATE, HIGH, VERY HIGH) for the heatmap."""
    if avg_vehicles_per_frame < LOW_THRESHOLD:
        return {"zone": "LOW", "color": "#2ECC71", "label": "LOW (Free Flow)"}
    if avg_vehicles_per_frame < HIGH_THRESHOLD:
        return {"zone": "MODERATE", "color": "#F4C430", "label": "MODERATE"}
    if avg_vehicles_per_frame < VERY_HIGH_THRESHOLD:
        return {"zone": "HIGH", "color": "#FF7A45", "label": "HIGH"}
    return {"zone": "VERY HIGH", "color": "#E63946", "label": "VERY HIGH"}


def get_most_frequent_category(class_counts: Dict[str, int]) -> Dict[str, object]:
    """Return the most frequently detected unique vehicle category."""
    if not class_counts or sum(class_counts.values()) == 0:
        return {"category": "None", "count": 0}
    top_cat = max(class_counts.items(), key=lambda item: item[1])
    return {"category": top_cat[0].capitalize(), "count": int(top_cat[1])}


def _choose_interval_seconds(duration_seconds: float) -> float:
    """Pick a clean interval width so videos yield ~5 to 15 intervals when possible."""
    for candidate in (1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0):
        if duration_seconds / candidate <= 15.0:
            return candidate
    return 60.0


def build_interval_timeline(
    frame_timeline: List[Dict],
    interval_seconds: Optional[float] = None,
) -> List[Dict]:
    """Aggregate per-frame detection stats into clean time intervals for charts & forecasting."""
    if not frame_timeline:
        return []

    max_ts = max(float(f.get("timestamp", 0.0)) for f in frame_timeline)
    min_ts = min(float(f.get("timestamp", 0.0)) for f in frame_timeline)
    duration = max(0.0, max_ts - min_ts)

    # If video is very short (< 2 seconds), split by frame chunks if >= 6 frames so intervals remain informative.
    if duration < 2.0 and len(frame_timeline) >= 6 and interval_seconds is None:
        chunk_size = max(2, len(frame_timeline) // 4)
        intervals: List[Dict] = []
        prev_unique = 0
        for idx in range(0, len(frame_timeline), chunk_size):
            bucket = frame_timeline[idx : idx + chunk_size]
            n_frames = len(bucket)
            avg_det = round(sum(b["detections"] for b in bucket) / n_frames, 2)
            peak_det = max(b["detections"] for b in bucket)
            end_unique = int(bucket[-1].get("unique_cumulative", prev_unique))
            new_unique = max(0, end_unique - prev_unique)
            prev_unique = end_unique
            cond, _ = classify_traffic_condition(avg_det)
            zone = get_congestion_zone(avg_det)
            intervals.append(
                {
                    "interval_index": len(intervals) + 1,
                    "label": f"Frames {bucket[0]['frame_id']}–{bucket[-1]['frame_id']}",
                    "start_sec": round(float(bucket[0]["timestamp"]), 2),
                    "end_sec": round(float(bucket[-1]["timestamp"]), 2),
                    "frames": n_frames,
                    "avg_vehicles": avg_det,
                    "peak_vehicles": int(peak_det),
                    "new_unique_vehicles": new_unique,
                    "car": round(sum(b.get("car", 0) for b in bucket) / n_frames, 2),
                    "bus": round(sum(b.get("bus", 0) for b in bucket) / n_frames, 2),
                    "truck": round(sum(b.get("truck", 0) for b in bucket) / n_frames, 2),
                    "motorcycle": round(sum(b.get("motorcycle", 0) for b in bucket) / n_frames, 2),
                    "condition": cond,
                    "zone": zone["zone"],
                }
            )
        return intervals

    step = interval_seconds if interval_seconds and interval_seconds > 0 else _choose_interval_seconds(max(duration, 1.0))
    buckets: Dict[int, List[Dict]] = {}
    for item in frame_timeline:
        b_idx = int(float(item.get("timestamp", 0.0)) // step)
        buckets.setdefault(b_idx, []).append(item)

    intervals = []
    prev_unique = 0
    for b_idx in sorted(buckets):
        bucket = buckets[b_idx]
        n_frames = len(bucket)
        start_s = round(b_idx * step, 1)
        end_s = round((b_idx + 1) * step, 1)
        avg_det = round(sum(b["detections"] for b in bucket) / n_frames, 2)
        peak_det = max(b["detections"] for b in bucket)
        end_unique = int(bucket[-1].get("unique_cumulative", prev_unique))
        new_unique = max(0, end_unique - prev_unique)
        prev_unique = end_unique
        cond, _ = classify_traffic_condition(avg_det)
        zone = get_congestion_zone(avg_det)
        intervals.append(
            {
                "interval_index": len(intervals) + 1,
                "label": f"{start_s:g}s–{end_s:g}s",
                "start_sec": start_s,
                "end_sec": end_s,
                "frames": n_frames,
                "avg_vehicles": avg_det,
                "peak_vehicles": int(peak_det),
                "new_unique_vehicles": new_unique,
                "car": round(sum(b.get("car", 0) for b in bucket) / n_frames, 2),
                "bus": round(sum(b.get("bus", 0) for b in bucket) / n_frames, 2),
                "truck": round(sum(b.get("truck", 0) for b in bucket) / n_frames, 2),
                "motorcycle": round(sum(b.get("motorcycle", 0) for b in bucket) / n_frames, 2),
                "condition": cond,
                "zone": zone["zone"],
            }
        )
    return intervals


def forecast_next_interval(intervals: List[Dict], window: int = 3, min_intervals: int = 3) -> Dict:
    """Estimate next-interval traffic volume using a simple moving average over recent intervals."""
    if len(intervals) < min_intervals:
        return {
            "available": False,
            "reason": (
                f"Insufficient interval observations for forecasting ({len(intervals)} recorded; "
                f"at least {min_intervals} video time intervals are required)."
            ),
            "window": window,
            "observed_intervals": len(intervals),
            "estimated_avg_vehicles": None,
            "estimated_condition": None,
            "explanation": None,
        }

    used_window = min(window, len(intervals))
    recent = intervals[-used_window:]
    recent_values = [float(item["avg_vehicles"]) for item in recent]
    estimated = round(sum(recent_values) / used_window, 2)
    est_condition, _ = classify_traffic_condition(estimated)

    last_val = recent_values[-1]
    if estimated > last_val + 0.5:
        trend_word = "slightly higher than"
    elif estimated < last_val - 0.5:
        trend_word = "slightly lower than"
    else:
        trend_word = "consistent with"

    explanation = (
        f"Computed using a simple {used_window}-interval moving average over the most recent "
        f"observed averages ({', '.join(f'{v:.2f}' for v in recent_values)} vehicles/frame). "
        f"The estimated next-interval volume ({estimated:.2f} avg vehicles/frame, {est_condition}) "
        f"is {trend_word} the latest observed interval ({last_val:.2f}). "
        "This is a statistical smoothing estimate from the video timeline, not a guaranteed prediction."
    )

    return {
        "available": True,
        "reason": None,
        "window": used_window,
        "observed_intervals": len(intervals),
        "estimated_avg_vehicles": estimated,
        "estimated_condition": est_condition,
        "next_label": "Next Interval (Est.)",
        "explanation": explanation,
    }


def analyze_traffic(tracking_summary: Dict, frame_timeline: Optional[List[Dict]] = None) -> Dict:
    """Compute Member 3 traffic analysis from Member 1+2 tracking summary."""
    frames_processed = int(tracking_summary.get("frames_processed", 0))
    total_detections = int(tracking_summary.get("total_frame_detections", 0))
    total_unique = int(tracking_summary.get("total_unique_vehicles", 0))

    raw_classes = tracking_summary.get("class_counts", {})
    class_counts = {
        "car": int(raw_classes.get("car", 0)),
        "bus": int(raw_classes.get("bus", 0)),
        "truck": int(raw_classes.get("truck", 0)),
        "motorcycle": int(raw_classes.get("motorcycle", 0)),
    }

    raw_categories = tracking_summary.get("category_counts", {})
    category_counts = {
        "cars": int(raw_categories.get("cars", class_counts["car"])),
        "buses": int(raw_categories.get("buses", class_counts["bus"])),
        "trucks": int(raw_categories.get("trucks", class_counts["truck"])),
        "two_wheelers": int(raw_categories.get("two_wheelers", class_counts["motorcycle"])),
    }

    avg_per_frame = round(total_detections / frames_processed, 2) if frames_processed > 0 else 0.0
    traffic_condition, advisory_congestion = classify_traffic_condition(avg_per_frame)
    congestion_zone = get_congestion_zone(avg_per_frame)
    most_frequent = get_most_frequent_category(class_counts)
    intervals = build_interval_timeline(frame_timeline or [])
    forecast = forecast_next_interval(intervals)

    return {
        "total_unique_vehicles": total_unique,
        "class_counts": class_counts,
        "category_counts": category_counts,
        "traffic_condition": traffic_condition,
        "advisory_congestion": advisory_congestion,
        "congestion_zone": congestion_zone,
        "most_frequent_category": most_frequent,
        "avg_vehicles_per_frame": avg_per_frame,
        "frames_processed": frames_processed,
        "threshold_description": THRESHOLD_DOC,
        "intervals": intervals,
        "forecast": forecast,
    }
