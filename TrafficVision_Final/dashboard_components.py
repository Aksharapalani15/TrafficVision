"""UI styling, interactive Altair charts, junction heatmap SVG, and report builders."""

from __future__ import annotations

import csv
import html
import io
import json
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

CATEGORY_LABELS = {
    "car": "Cars",
    "bus": "Buses",
    "truck": "Trucks",
    "motorcycle": "Motorcycles",
}

CATEGORY_COLORS = {
    "Cars": "#FF6B6B",        # Coral pink
    "Buses": "#9B87F5",       # Soft lavender
    "Trucks": "#FF9F43",      # Warm peach/orange
    "Motorcycles": "#38C19B", # Mint green
}

CONDITION_STYLES = {
    "LOW": {"bg": "#DFF8EE", "border": "#58C99B", "text": "#166547", "dot": "#2ECC71", "icon": "🟢"},
    "MODERATE": {"bg": "#FFF4CC", "border": "#F2C94C", "text": "#7A5B00", "dot": "#F4C430", "icon": "🟡"},
    "HIGH": {"bg": "#FFE4D6", "border": "#FF8C61", "text": "#8C2D0B", "dot": "#FF7A45", "icon": "🟠"},
    "VERY HIGH": {"bg": "#FFD9DE", "border": "#E63946", "text": "#8B1120", "dot": "#E63946", "icon": "🔴"},
}


def inject_custom_css() -> None:
    """Apply the warm cream, coral, peach, butter yellow, lavender & mint theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #2D2121;
        }

        .stApp {
            background: linear-gradient(160deg, #FFFDF7 0%, #FFF8EE 52%, #FBF4EA 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FBF4EA 0%, #FFF6EC 60%, #F7ECFF 100%);
            border-right: 1px solid #F0E2D0;
        }

        /* Hero Header */
        .tv-hero {
            background: linear-gradient(120deg, #FFE2D6 0%, #FFF1C9 48%, #EFE6FF 100%);
            border: 1px solid rgba(255, 139, 107, 0.25);
            border-radius: 20px;
            padding: 1.45rem 1.8rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 10px 28px rgba(155, 105, 80, 0.10);
        }
        .tv-hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.78);
            color: #C44D34;
            font-weight: 700;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 107, 107, 0.30);
            margin-bottom: 0.5rem;
        }
        .tv-hero h1 {
            margin: 0;
            font-size: 2.05rem;
            font-weight: 800;
            color: #2D2121;
            line-height: 1.15;
        }
        .tv-hero p {
            margin: 0.45rem 0 0 0;
            color: #5C4B4B;
            font-size: 0.98rem;
            font-weight: 500;
        }

        /* Metric Cards */
        .tv-card {
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(210, 180, 155, 0.32);
            box-shadow: 0 8px 20px rgba(130, 88, 60, 0.07);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            min-height: 122px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .tv-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 26px rgba(130, 88, 60, 0.13);
        }
        .tv-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .tv-card-label {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #5C4B4B;
        }
        .tv-card-icon {
            font-size: 1.3rem;
            background: rgba(255, 255, 255, 0.72);
            border-radius: 11px;
            padding: 0.25rem 0.48rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .tv-card-value {
            font-size: 2rem;
            font-weight: 800;
            color: #2D2121;
            line-height: 1.1;
            margin-top: 0.45rem;
        }
        .tv-card-sub {
            font-size: 0.80rem;
            color: #6B5959;
            margin-top: 0.25rem;
        }

        /* Pastel card gradients */
        .bg-coral   { background: linear-gradient(135deg, #FFF0EB 0%, #FFDCD1 100%); }
        .bg-peach   { background: linear-gradient(135deg, #FFF4E6 0%, #FFE2C4 100%); }
        .bg-butter  { background: linear-gradient(135deg, #FFFBE6 0%, #FFF0B3 100%); }
        .bg-lavender{ background: linear-gradient(135deg, #F6F1FF 0%, #E7DCFF 100%); }
        .bg-mint    { background: linear-gradient(135deg, #ECFCF5 0%, #CCF5E4 100%); }
        .bg-ivory   { background: linear-gradient(135deg, #FFFFFF 0%, #FAF0E2 100%); }

        /* Status Pill / Panel */
        .tv-status-banner {
            border-radius: 16px;
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin: 0.4rem 0 1rem 0;
            box-shadow: 0 6px 16px rgba(130, 88, 60, 0.06);
        }
        .tv-section-box {
            background: #FFFBF4;
            border: 1px solid #EFE0CE;
            border-radius: 18px;
            padding: 1.15rem 1.35rem;
            box-shadow: 0 6px 18px rgba(130, 88, 60, 0.05);
            margin-bottom: 1rem;
        }

        /* Primary buttons */
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF9F43 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.55rem 1.25rem;
            box-shadow: 0 6px 16px rgba(255, 107, 107, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(255, 107, 107, 0.35);
            color: #FFFFFF;
        }

        /* Navigation tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #F7ECDC;
            padding: 6px;
            border-radius: 16px;
            border: 1px solid #EAD8C0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 8px 16px;
            font-weight: 600;
            color: #5C4B4B;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FF7E6B 0%, #FFA652 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: object,
    icon: str,
    subtitle: str = "",
    theme_class: str = "bg-ivory",
) -> None:
    """Render a rounded, gradient-backed KPI card."""
    safe_label = html.escape(str(label))
    safe_val = html.escape(str(value))
    safe_icon = html.escape(str(icon))
    safe_sub = html.escape(str(subtitle))
    sub_html = f'<div class="tv-card-sub">{safe_sub}</div>' if safe_sub else ""
    st.markdown(
        f"""
        <div class="tv-card {theme_class}">
            <div class="tv-card-top">
                <span class="tv-card-label">{safe_label}</span>
                <span class="tv-card-icon">{safe_icon}</span>
            </div>
            <div class="tv-card-value">{safe_val}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_condition_banner(condition: str, detail_text: str) -> None:
    """Render a pastel status banner for LOW / MODERATE / HIGH / VERY HIGH."""
    style = CONDITION_STYLES.get(condition, CONDITION_STYLES["MODERATE"])
    safe_cond = html.escape(condition)
    safe_detail = html.escape(detail_text)
    st.markdown(
        f"""
        <div class="tv-status-banner" style="background:{style['bg']}; border:1.5px solid {style['border']}; color:{style['text']};">
            <div style="font-size:1.08rem; font-weight:800;">
                {style['icon']} Traffic Condition: {safe_cond}
            </div>
            <div style="font-size:0.9rem; font-weight:600;">
                {safe_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_df_from_counts(class_counts: Dict[str, int], selected_keys: List[str]) -> pd.DataFrame:
    rows = []
    total = sum(int(class_counts.get(k, 0)) for k in selected_keys)
    for k in selected_keys:
        cnt = int(class_counts.get(k, 0))
        label = CATEGORY_LABELS.get(k, k.capitalize())
        pct = round((cnt / total) * 100.0, 1) if total > 0 else 0.0
        rows.append({"Vehicle Type": label, "Count": cnt, "Share (%)": pct, "Key": k})
    return pd.DataFrame(rows)


def build_donut_chart(class_counts: Dict[str, int], selected_keys: List[str]) -> alt.Chart:
    """Colourful interactive donut chart of unique vehicle distribution."""
    df = _chart_df_from_counts(class_counts, selected_keys)
    domain = [CATEGORY_LABELS[k] for k in selected_keys]
    range_colors = [CATEGORY_COLORS[CATEGORY_LABELS[k]] for k in selected_keys]

    return (
        alt.Chart(df)
        .mark_arc(innerRadius=62, outerRadius=112, stroke="#FFFDF7", strokeWidth=2.5)
        .encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color(
                "Vehicle Type:N",
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=alt.Legend(title="Vehicle Type", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Vehicle Type:N", title="Category"),
                alt.Tooltip("Count:Q", title="Unique Count"),
                alt.Tooltip("Share (%):Q", title="Share (%)", format=".1f"),
            ],
        )
        .properties(height=310, title="Vehicle Category Distribution")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=15, color="#2D2121", anchor="start")
    )


def build_bar_chart(class_counts: Dict[str, int], selected_keys: List[str]) -> alt.Chart:
    """Colourful rounded-top bar chart of category-wise unique counts."""
    df = _chart_df_from_counts(class_counts, selected_keys)
    domain = [CATEGORY_LABELS[k] for k in selected_keys]
    range_colors = [CATEGORY_COLORS[CATEGORY_LABELS[k]] for k in selected_keys]

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=9, cornerRadiusTopRight=9, size=44)
        .encode(
            x=alt.X("Vehicle Type:N", sort=domain, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Count:Q", title="Unique Vehicles Counted"),
            color=alt.Color(
                "Vehicle Type:N",
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Vehicle Type:N", title="Category"),
                alt.Tooltip("Count:Q", title="Unique Count"),
                alt.Tooltip("Share (%):Q", title="Share (%)", format=".1f"),
            ],
        )
    )
    labels = bars.mark_text(dy=-8, fontWeight="bold", fontSize=13, color="#2D2121").encode(
        text="Count:Q"
    )
    return (
        (bars + labels)
        .properties(height=310, title="Category-wise Unique Vehicle Counts")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#F0E4D4", labelColor="#4A3B3B", titleColor="#4A3B3B")
        .configure_title(fontSize=15, color="#2D2121", anchor="start")
    )


def build_timeline_chart(intervals: List[Dict], selected_keys: List[str]) -> alt.Chart:
    """Interactive timeline of average vehicles detected per frame across video intervals."""
    rows = []
    for item in intervals:
        filtered_avg = round(sum(float(item.get(k, 0.0)) for k in selected_keys), 2)
        rows.append(
            {
                "IntervalIndex": item["interval_index"],
                "Interval": item["label"],
                "Avg Vehicles / Frame": filtered_avg,
                "All Vehicles Avg": item["avg_vehicles"],
                "Peak Frame Load": item["peak_vehicles"],
                "New Unique Vehicles": item["new_unique_vehicles"],
                "Condition": item["condition"],
            }
        )
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("Interval:N", sort=alt.SortField("IntervalIndex"), title="Video Time Interval", axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("Avg Vehicles / Frame:Q", title="Avg Vehicles / Frame"),
    )
    area = base.mark_area(
        opacity=0.28,
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#FF6B6B", offset=0),
                alt.GradientStop(color="#FFE680", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    )
    line = base.mark_line(color="#FF6B6B", strokeWidth=3)
    points = base.mark_circle(size=85, color="#FF7A45", stroke="#FFFFFF", strokeWidth=1.5).encode(
        tooltip=[
            alt.Tooltip("Interval:N", title="Interval"),
            alt.Tooltip("Avg Vehicles / Frame:Q", title="Selected Avg/Frame", format=".2f"),
            alt.Tooltip("All Vehicles Avg:Q", title="Total Avg/Frame", format=".2f"),
            alt.Tooltip("Peak Frame Load:Q", title="Peak Frame Detections"),
            alt.Tooltip("New Unique Vehicles:Q", title="New Unique Vehicles"),
            alt.Tooltip("Condition:N", title="Interval Condition"),
        ]
    )
    return (
        (area + line + points)
        .properties(height=300, title="Traffic Volume Timeline Across Video Intervals")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#F0E4D4", labelColor="#4A3B3B", titleColor="#4A3B3B")
        .configure_title(fontSize=15, color="#2D2121", anchor="start")
    )


def build_interval_heatmap(intervals: List[Dict]) -> alt.Chart:
    """Colourful interval-by-interval congestion strip heatmap."""
    df = pd.DataFrame(intervals)
    zone_domain = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    zone_colors = ["#2ECC71", "#F4C430", "#FF7A45", "#E63946"]

    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=8, stroke="#FFFDF7", strokeWidth=2)
        .encode(
            x=alt.X("label:N", sort=alt.SortField("interval_index"), title="Video Time Interval", axis=alt.Axis(labelAngle=-20)),
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(domain=zone_domain, range=zone_colors),
                legend=alt.Legend(title="Zone Intensity", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Interval"),
                alt.Tooltip("avg_vehicles:Q", title="Avg Vehicles/Frame", format=".2f"),
                alt.Tooltip("peak_vehicles:Q", title="Peak Vehicles"),
                alt.Tooltip("zone:N", title="Zone"),
            ],
        )
        .properties(height=115, title="Observed Congestion Intensity Across Video Intervals")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=14, color="#2D2121", anchor="start")
    )


def build_forecast_chart(intervals: List[Dict], forecast: Dict) -> alt.Chart:
    """Observed timeline + dashed moving-average estimated next interval."""
    rows = []
    for item in intervals:
        rows.append(
            {
                "Step": item["interval_index"],
                "Label": item["label"],
                "Avg Vehicles / Frame": float(item["avg_vehicles"]),
                "Series": "Observed",
            }
        )
    last_item = intervals[-1]
    next_step = last_item["interval_index"] + 1
    next_label = forecast["next_label"]
    est_val = float(forecast["estimated_avg_vehicles"])

    # Connector row from last observed point to estimated next point
    connector_rows = [
        {
            "Step": last_item["interval_index"],
            "Label": last_item["label"],
            "Avg Vehicles / Frame": float(last_item["avg_vehicles"]),
            "Series": "Estimated (Moving Avg)",
        },
        {
            "Step": next_step,
            "Label": next_label,
            "Avg Vehicles / Frame": est_val,
            "Series": "Estimated (Moving Avg)",
        },
    ]

    obs_df = pd.DataFrame(rows)
    est_df = pd.DataFrame(connector_rows)
    all_labels = [r["Label"] for r in rows] + [next_label]

    obs_line = (
        alt.Chart(obs_df)
        .mark_line(strokeWidth=3, color="#FF6B6B")
        .encode(
            x=alt.X("Label:N", sort=all_labels, title="Time Interval", axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("Avg Vehicles / Frame:Q", title="Avg Vehicles / Frame"),
        )
    )
    obs_pts = (
        alt.Chart(obs_df)
        .mark_circle(size=85, color="#FF6B6B", stroke="#FFFFFF", strokeWidth=1.5)
        .encode(
            x=alt.X("Label:N", sort=all_labels),
            y="Avg Vehicles / Frame:Q",
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Observed", "Estimated (Moving Avg)"],
                    range=["#FF6B6B", "#8B6FF0"],
                ),
                legend=alt.Legend(title="Data Type", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Label:N", title="Interval"),
                alt.Tooltip("Series:N", title="Type"),
                alt.Tooltip("Avg Vehicles / Frame:Q", format=".2f"),
            ],
        )
    )
    est_line = (
        alt.Chart(est_df)
        .mark_line(strokeWidth=3, strokeDash=[7, 5], color="#8B6FF0")
        .encode(
            x=alt.X("Label:N", sort=all_labels),
            y="Avg Vehicles / Frame:Q",
        )
    )
    est_pt = (
        alt.Chart(est_df.iloc[[1]])
        .mark_point(shape="diamond", size=190, filled=True, color="#8B6FF0", stroke="#FFFFFF", strokeWidth=1.8)
        .encode(
            x=alt.X("Label:N", sort=all_labels),
            y="Avg Vehicles / Frame:Q",
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Observed", "Estimated (Moving Avg)"],
                    range=["#FF6B6B", "#8B6FF0"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Label:N", title="Interval"),
                alt.Tooltip("Series:N", title="Type"),
                alt.Tooltip("Avg Vehicles / Frame:Q", title="Estimated Avg/Frame", format=".2f"),
            ],
        )
    )

    return (
        (obs_line + est_line + obs_pts + est_pt)
        .properties(height=300, title="Observed Traffic Volume & Moving-Average Next-Interval Estimate")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#F0E4D4", labelColor="#4A3B3B", titleColor="#4A3B3B")
        .configure_title(fontSize=15, color="#2D2121", anchor="start")
    )


def render_junction_heatmap_svg(analysis: Dict) -> str:
    """Return a polished HTML/SVG top-view junction heatmap reflecting actual video-level congestion."""
    zone = analysis["congestion_zone"]
    zone_name = zone["zone"]
    zone_color = zone["color"]
    zone_label = zone["label"]
    avg_density = analysis["avg_vehicles_per_frame"]
    total_unique = analysis["total_unique_vehicles"]

    glow_map = {
        "LOW": "rgba(46, 204, 113, 0.55)",
        "MODERATE": "rgba(244, 196, 48, 0.62)",
        "HIGH": "rgba(255, 122, 69, 0.68)",
        "VERY HIGH": "rgba(230, 57, 70, 0.72)",
    }
    glow_color = glow_map.get(zone_name, "rgba(255, 122, 69, 0.65)")

    return f"""
    <div style="font-family:'Plus Jakarta Sans', system-ui, sans-serif; background:linear-gradient(145deg,#FFFDF9,#FBF1E3);
                border:1px solid #ECD8C3; border-radius:20px; padding:16px; box-shadow:0 8px 22px rgba(130,88,60,0.08);">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
        <div>
          <div style="font-size:15px; font-weight:800; color:#2D2121;">Illustrative Junction Congestion Map (Video-Level Density)</div>
          <div style="font-size:12px; color:#6A5757;">Single-camera aggregate intensity — not GPS or lane-separated measurements</div>
        </div>
        <div style="background:{zone_color}; color:#FFFFFF; font-weight:800; font-size:12px; padding:6px 14px; border-radius:999px; box-shadow:0 3px 10px {glow_color};">
          STATUS: {html.escape(zone_label)}
        </div>
      </div>

      <svg viewBox="0 0 760 340" width="100%" height="310" xmlns="http://www.w3.org/2000/svg" style="display:block; border-radius:16px; background:#F5EAD8;">
        <defs>
          <radialGradient id="junctionGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="{zone_color}" stop-opacity="0.88"/>
            <stop offset="60%" stop-color="{zone_color}" stop-opacity="0.45"/>
            <stop offset="100%" stop-color="{zone_color}" stop-opacity="0.06"/>
          </radialGradient>
          <linearGradient id="roadV" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#3E3A42"/>
            <stop offset="50%" stop-color="#4A454F"/>
            <stop offset="100%" stop-color="#3E3A42"/>
          </linearGradient>
          <linearGradient id="roadH" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#3E3A42"/>
            <stop offset="50%" stop-color="#4A454F"/>
            <stop offset="100%" stop-color="#3E3A42"/>
          </linearGradient>
          <style>
            @keyframes pulseZone {{
              0% {{ transform: scale(0.95); opacity: 0.78; }}
              50% {{ transform: scale(1.04); opacity: 1.0; }}
              100% {{ transform: scale(0.95); opacity: 0.78; }}
            }}
            .pulse-ring {{
              transform-origin: 380px 170px;
              animation: pulseZone 2.6s ease-in-out infinite;
            }}
          </style>
        </defs>

        <!-- Pastel corner landscaping blocks -->
        <rect x="14" y="14" width="268" height="96" rx="18" fill="#DDF5E8" stroke="#BCE5CF" stroke-width="1.5"/>
        <rect x="478" y="14" width="268" height="96" rx="18" fill="#FFE8DC" stroke="#F8CBB6" stroke-width="1.5"/>
        <rect x="14" y="230" width="268" height="96" rx="18" fill="#EFE8FF" stroke="#D8C9FA" stroke-width="1.5"/>
        <rect x="478" y="230" width="268" height="96" rx="18" fill="#FFF3C8" stroke="#F3DD93" stroke-width="1.5"/>

        <!-- Decorative subtle trees / blocks -->
        <circle cx="52" cy="52" r="16" fill="#9CE0BE"/>
        <circle cx="88" cy="48" r="12" fill="#B6E9CF"/>
        <circle cx="708" cy="54" r="15" fill="#FFC3A8"/>
        <circle cx="670" cy="286" r="14" fill="#FFE082"/>
        <circle cx="64" cy="284" r="15" fill="#D3BFFF"/>

        <!-- Roads -->
        <rect x="300" y="0" width="160" height="340" fill="url(#roadV)"/>
        <rect x="0" y="122" width="760" height="96" fill="url(#roadH)"/>

        <!-- Road shoulders -->
        <line x1="300" y1="0" x2="300" y2="122" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="460" y1="0" x2="460" y2="122" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="300" y1="218" x2="300" y2="340" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="460" y1="218" x2="460" y2="340" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="0" y1="122" x2="300" y2="122" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="460" y1="122" x2="760" y2="122" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="0" y1="218" x2="300" y2="218" stroke="#FFE6A3" stroke-width="3"/>
        <line x1="460" y1="218" x2="760" y2="218" stroke="#FFE6A3" stroke-width="3"/>

        <!-- Center dashed lane dividers -->
        <line x1="380" y1="10" x2="380" y2="104" stroke="#FFFDF7" stroke-width="2.5" stroke-dasharray="10 10" opacity="0.75"/>
        <line x1="380" y1="236" x2="380" y2="330" stroke="#FFFDF7" stroke-width="2.5" stroke-dasharray="10 10" opacity="0.75"/>
        <line x1="12" y1="170" x2="278" y2="170" stroke="#FFFDF7" stroke-width="2.5" stroke-dasharray="10 10" opacity="0.75"/>
        <line x1="482" y1="170" x2="748" y2="170" stroke="#FFFDF7" stroke-width="2.5" stroke-dasharray="10 10" opacity="0.75"/>

        <!-- Zebra crosswalk stripes -->
        <g fill="#FFFDF7" opacity="0.72">
          <rect x="312" y="108" width="14" height="6" rx="2"/><rect x="336" y="108" width="14" height="6" rx="2"/>
          <rect x="360" y="108" width="14" height="6" rx="2"/><rect x="386" y="108" width="14" height="6" rx="2"/>
          <rect x="410" y="108" width="14" height="6" rx="2"/><rect x="434" y="108" width="14" height="6" rx="2"/>

          <rect x="312" y="226" width="14" height="6" rx="2"/><rect x="336" y="226" width="14" height="6" rx="2"/>
          <rect x="360" y="226" width="14" height="6" rx="2"/><rect x="386" y="226" width="14" height="6" rx="2"/>
          <rect x="410" y="226" width="14" height="6" rx="2"/><rect x="434" y="226" width="14" height="6" rx="2"/>
        </g>

        <!-- Congestion Heatmap Zones across corridors (overall video-level intensity) -->
        <rect x="312" y="14" width="136" height="86" rx="14" fill="{zone_color}" opacity="0.34"/>
        <rect x="312" y="240" width="136" height="86" rx="14" fill="{zone_color}" opacity="0.34"/>
        <rect x="18" y="132" width="262" height="76" rx="14" fill="{zone_color}" opacity="0.34"/>
        <rect x="480" y="132" width="262" height="76" rx="14" fill="{zone_color}" opacity="0.34"/>

        <!-- Pulsing junction core heatmap -->
        <circle class="pulse-ring" cx="380" cy="170" r="88" fill="url(#junctionGlow)"/>
        <rect x="298" y="120" width="164" height="100" rx="16" fill="{zone_color}" opacity="0.30" stroke="{zone_color}" stroke-width="2.5"/>

        <!-- Center readouts -->
        <rect x="313" y="136" width="134" height="68" rx="14" fill="#FFFDF9" opacity="0.95" stroke="{zone_color}" stroke-width="2"/>
        <text x="380" y="159" text-anchor="middle" font-size="11" font-weight="700" fill="#6B5959">OVERALL DENSITY</text>
        <text x="380" y="180" text-anchor="middle" font-size="18" font-weight="800" fill="#2D2121">{avg_density:.2f} / frame</text>
        <text x="380" y="196" text-anchor="middle" font-size="11" font-weight="700" fill="{zone_color}">{total_unique} unique vehicles</text>

        <!-- Labels inside corner blocks -->
        <text x="148" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#2F6649">Camera Monitored Zone</text>
        <text x="612" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#8C3E22">Aggregate Video Level</text>
      </svg>

      <!-- Legend -->
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-top:12px; font-size:12px; color:#4A3B3B;">
        <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center;">
          <span><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#2ECC71;margin-right:5px;"></span><b>Green:</b> LOW (&lt; 5.0)</span>
          <span><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#F4C430;margin-right:5px;"></span><b>Yellow:</b> MODERATE (5.0–11.99)</span>
          <span><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#FF7A45;margin-right:5px;"></span><b>Orange:</b> HIGH (12.0–19.99)</span>
          <span><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#E63946;margin-right:5px;"></span><b>Red:</b> VERY HIGH (&ge; 20.0)</span>
        </div>
      </div>
    </div>
    """


def build_csv_report(result: Dict) -> str:
    """Create a clean CSV report containing only actual pipeline outputs."""
    analysis = result["analysis"]
    video_info = result["video_info"]
    cls_counts = analysis["class_counts"]

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Section", "Metric", "Value"])
    writer.writerow(["Video Summary", "Source Video", video_info.get("path", "")])
    writer.writerow(["Video Summary", "Resolution", f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"])
    writer.writerow(["Video Summary", "FPS", round(float(video_info.get("fps", 0.0)), 2)])
    writer.writerow(["Video Summary", "Frames Processed", result.get("frames_processed", 0)])
    writer.writerow(["Video Summary", "Processing Time (s)", result.get("processing_seconds", 0.0)])
    writer.writerow(["Traffic Analysis", "Total Unique Vehicles", analysis["total_unique_vehicles"]])
    writer.writerow(["Traffic Analysis", "Average Vehicles Per Frame", analysis["avg_vehicles_per_frame"]])
    writer.writerow(["Traffic Analysis", "Traffic Condition", analysis["traffic_condition"]])
    writer.writerow(["Traffic Analysis", "Most Frequent Category", analysis["most_frequent_category"]["category"]])

    writer.writerow([])
    writer.writerow(["Vehicle Category", "Unique Count"])
    for key in ("car", "bus", "truck", "motorcycle"):
        writer.writerow([CATEGORY_LABELS[key], cls_counts.get(key, 0)])

    intervals = analysis.get("intervals", [])
    if intervals:
        writer.writerow([])
        writer.writerow([
            "Interval Index",
            "Interval Label",
            "Frames",
            "Avg Vehicles / Frame",
            "Peak Vehicles",
            "New Unique Vehicles",
            "Condition",
        ])
        for item in intervals:
            writer.writerow([
                item["interval_index"],
                item["label"],
                item["frames"],
                item["avg_vehicles"],
                item["peak_vehicles"],
                item["new_unique_vehicles"],
                item["condition"],
            ])

    return buf.getvalue()


def build_json_report(result: Dict) -> str:
    """Create a clean JSON report containing only actual pipeline outputs."""
    payload = {
        "video_info": result.get("video_info", {}),
        "frames_processed": result.get("frames_processed", 0),
        "processing_seconds": result.get("processing_seconds", 0.0),
        "output_video_path": result.get("output_video_path", ""),
        "analysis": result.get("analysis", {}),
        "advisory": result.get("advisory", {}),
    }
    return json.dumps(payload, indent=2)
