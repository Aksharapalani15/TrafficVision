"""TrafficVision - Enhanced Interactive Streamlit Dashboard."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard_components import (
    CATEGORY_LABELS,
    build_bar_chart,
    build_csv_report,
    build_donut_chart,
    build_forecast_chart,
    build_interval_heatmap,
    build_json_report,
    build_timeline_chart,
    inject_custom_css,
    render_condition_banner,
    render_junction_heatmap_svg,
    render_metric_card,
)
from pipeline import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, process_traffic_video
from vehicle_detection import ModelLoadError, VideoError

UPLOADS_DIR = DEFAULT_OUTPUT_DIR / "uploads"
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "vehicle_detection" / "sample_data" / "traffic.mp4"


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    return cleaned or "uploaded_video.mp4"


def _save_uploaded_video(uploaded_file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = UPLOADS_DIR / f"{stamp}_{_safe_filename(uploaded_file.name)}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_empty_notice(section_name: str) -> None:
    st.info(
        f"No processed video results yet for **{section_name}**. "
        "Open the **🎬 Video Analysis** tab to upload a traffic video (or use the built-in sample) "
        "and click **Process Video**."
    )


def _render_sidebar(result: dict | None) -> tuple[float, int, bool]:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.4rem 0 0.8rem 0;">
                <div style="font-size:1.35rem; font-weight:800; color:#2D2121;">🚦 TrafficVision</div>
                <div style="font-size:0.83rem; color:#6A5757;">Smart Video Traffic Monitoring</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("⚙️ Processing Settings")
        conf_threshold = st.slider(
            "Detection Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum YOLO confidence score for vehicle detections.",
        )
        max_frames_val = st.number_input(
            "Max Frames (0 = full video)",
            min_value=0,
            value=0,
            step=25,
            help="Set a frame cap for a quick demonstration, or 0 to process the entire video.",
        )
        force_reprocess = st.checkbox(
            "Force re-process cached video",
            value=False,
            help="Unchecked reuses cached results when the same video and settings were already processed.",
        )

        st.divider()
        st.subheader("📌 Current Session")
        if result:
            info = result["video_info"]
            analysis = result["analysis"]
            st.markdown(f"**File:** `{Path(info.get('path', 'video')).name}`")
            st.markdown(f"**Frames:** `{result['frames_processed']}`")
            st.markdown(f"**Unique Vehicles:** `{analysis['total_unique_vehicles']}`")
            st.markdown(f"**Condition:** `{analysis['traffic_condition']}`")
        else:
            st.caption("Waiting for a traffic video to be processed.")

    return conf_threshold, int(max_frames_val), force_reprocess


def _render_overview_tab(result: dict | None) -> None:
    if not result:
        _render_empty_notice("Overview")
        return

    analysis = result["analysis"]
    cls_counts = analysis["class_counts"]
    top_cat = analysis["most_frequent_category"]
    condition = analysis["traffic_condition"]

    render_condition_banner(
        condition,
        f"Measured Density: {analysis['avg_vehicles_per_frame']} avg vehicles/frame • "
        f"{analysis['frames_processed']} frames analyzed",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card(
            "Total Unique Vehicles",
            analysis["total_unique_vehicles"],
            "🚘",
            "Tracked across video",
            "bg-coral",
        )
    with c2:
        render_metric_card("Cars", cls_counts["car"], "🚗", "Unique counted", "bg-peach")
    with c3:
        render_metric_card("Buses", cls_counts["bus"], "🚌", "Unique counted", "bg-lavender")
    with c4:
        render_metric_card("Trucks", cls_counts["truck"], "🚚", "Unique counted", "bg-butter")
    with c5:
        render_metric_card("Motorcycles", cls_counts["motorcycle"], "🏍️", "Unique counted", "bg-mint")

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        render_metric_card(
            "Avg Vehicles / Frame",
            f"{analysis['avg_vehicles_per_frame']:.2f}",
            "📐",
            analysis["threshold_description"],
            "bg-ivory",
        )
    with s2:
        render_metric_card(
            "Most Frequent Category",
            top_cat["category"],
            "🏆",
            f"{top_cat['count']} unique vehicles detected" if top_cat["count"] > 0 else "No vehicles detected",
            "bg-lavender",
        )
    with s3:
        render_metric_card(
            "Processing Speed",
            f"{result['processing_seconds']} s",
            "⏱️",
            f"{result['frames_processed']} frames @ {result['video_info'].get('width', 0)}×{result['video_info'].get('height', 0)}",
            "bg-mint",
        )


def _render_video_tab(conf_threshold: float, max_frames_val: int, force_reprocess: bool) -> None:
    st.markdown(
        '<div class="tv-section-box">'
        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.3rem;">Upload & Process Traffic Video</div>'
        '<div style="font-size:0.88rem; color:#5C4B4B;">'
        'Upload a traffic clip (MP4, AVI, MOV, MKV, WEBM) or test with the built-in sample video. '
        'Detection, tracking, counting, congestion analysis, and advisory run in one pass.'
        "</div></div>",
        unsafe_allow_html=True,
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Traffic Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM",
        )
    with col_sample:
        use_sample = False
        if SAMPLE_VIDEO_PATH.exists():
            use_sample = st.checkbox(
                "Use sample video",
                value=(uploaded_file is None and st.session_state.get("tv_prefer_sample", False)),
                help="Use bundled vehicle_detection/sample_data/traffic.mp4",
            )
            st.session_state["tv_prefer_sample"] = use_sample

    source_ready = (uploaded_file is not None) or use_sample
    process_clicked = st.button("Process Video", type="primary", disabled=not source_ready)

    if process_clicked:
        if not source_ready:
            st.warning("Please upload a traffic video or select the sample video first.")
            return

        if uploaded_file is not None and uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please choose a valid traffic video.")
            return

        cache_store = st.session_state.setdefault("tv_cache", {})
        if uploaded_file is not None:
            cache_key = ("upload", uploaded_file.name, uploaded_file.size, conf_threshold, max_frames_val)
        else:
            cache_key = ("sample", str(SAMPLE_VIDEO_PATH), SAMPLE_VIDEO_PATH.stat().st_size, conf_threshold, max_frames_val)

        if not force_reprocess and cache_key in cache_store:
            cached = cache_store[cache_key]
            if Path(cached["output_video_path"]).exists():
                st.session_state["tv_result"] = cached
                st.info("Loaded cached pipeline results for this video and settings (no duplicate processing needed).")
                return

        status_box = st.empty()
        progress_bar = st.progress(0.0)
        status_box.info("Preparing video and loading YOLO detection model...")

        try:
            input_path = _save_uploaded_video(uploaded_file) if uploaded_file is not None else SAMPLE_VIDEO_PATH

            def on_progress(done: int, total: int) -> None:
                if total > 0:
                    cap_total = min(total, max_frames_val) if max_frames_val > 0 else total
                    frac = min(1.0, done / float(max(1, cap_total)))
                    progress_bar.progress(frac)
                    status_box.info(f"Processing frame {done} / {cap_total}...")
                else:
                    status_box.info(f"Processing frame {done}...")

            result = process_traffic_video(
                video_path=input_path,
                output_dir=DEFAULT_OUTPUT_DIR,
                conf=conf_threshold,
                max_frames=(max_frames_val if max_frames_val > 0 else None),
                use_llm=False,
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0)
            status_box.success(
                f"Processing complete: {result['frames_processed']} frames in {result['processing_seconds']}s."
            )
            cache_store[cache_key] = result
            st.session_state["tv_result"] = result
        except VideoError as exc:
            progress_bar.empty()
            status_box.error(f"Video error: {exc}")
        except ModelLoadError as exc:
            progress_bar.empty()
            status_box.error(f"Model loading error: {exc}")
        except Exception as exc:
            progress_bar.empty()
            status_box.error(f"Processing failed: {exc}")

    result = st.session_state.get("tv_result")
    if not result:
        return

    st.subheader("Annotated Output Video")
    video_path = Path(result["output_video_path"])
    info = result["video_info"]

    if video_path.exists() and video_path.stat().st_size > 0:
        vid_col, info_col = st.columns([2.4, 1])
        with vid_col:
            st.video(str(video_path))
        with info_col:
            render_metric_card(
                "Frames Processed",
                result["frames_processed"],
                "🎞️",
                f"Source FPS: {round(float(info.get('fps', 0)), 2)}",
                "bg-peach",
            )
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            render_metric_card(
                "Processing Time",
                f"{result['processing_seconds']} s",
                "⚡",
                f"Resolution: {info.get('width', 0)}×{info.get('height', 0)}",
                "bg-mint",
            )
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            with open(video_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=vf,
                    file_name=video_path.name,
                    mime="video/mp4" if video_path.suffix == ".mp4" else "video/webm",
                    use_container_width=True,
                )
    else:
        st.error(f"Annotated output video file could not be found at: {video_path}")


def _render_analytics_tab(result: dict | None) -> None:
    if not result:
        _render_empty_notice("Traffic Analytics")
        return

    analysis = result["analysis"]
    cls_counts = analysis["class_counts"]
    intervals = analysis.get("intervals", [])
    forecast = analysis.get("forecast", {})

    # Interactive category filter
    all_keys = ["car", "bus", "truck", "motorcycle"]
    selected_labels = st.multiselect(
        "Filter Vehicle Categories",
        options=[CATEGORY_LABELS[k] for k in all_keys],
        default=[CATEGORY_LABELS[k] for k in all_keys],
        help="Filter the distribution, bar chart, and timeline below by vehicle class.",
    )
    selected_keys = [k for k in all_keys if CATEGORY_LABELS[k] in selected_labels]
    if not selected_keys:
        st.warning("Select at least one vehicle category to display charts.")
        return

    filtered_total = sum(int(cls_counts.get(k, 0)) for k in selected_keys)
    col_donut, col_bar = st.columns(2)
    with col_donut:
        if filtered_total > 0:
            st.altair_chart(build_donut_chart(cls_counts, selected_keys), use_container_width=True)
        else:
            st.info("No vehicles detected for the selected categories to plot a distribution chart.")
    with col_bar:
        st.altair_chart(build_bar_chart(cls_counts, selected_keys), use_container_width=True)

    st.divider()
    if intervals:
        st.altair_chart(build_timeline_chart(intervals, selected_keys), use_container_width=True)
    else:
        st.info("Interval timeline is unavailable because no frames were recorded.")

    # Traffic Condition Summary across intervals + explanation
    st.subheader("Traffic Condition Summary")
    cond_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    for item in intervals:
        cond = item.get("condition", "LOW")
        cond_counts[cond] = cond_counts.get(cond, 0) + 1

    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("LOW Intervals", cond_counts["LOW"], "🟢", "< 5.0 avg vehicles/frame", "bg-mint")
    with m2:
        render_metric_card("MODERATE Intervals", cond_counts["MODERATE"], "🟡", "5.0 to < 12.0 avg/frame", "bg-butter")
    with m3:
        render_metric_card("HIGH Intervals", cond_counts["HIGH"], "🟠", ">= 12.0 avg/frame", "bg-coral")

    top_cat = analysis["most_frequent_category"]
    st.markdown(
        f"""
        <div class="tv-section-box" style="margin-top:0.85rem;">
            <b>Analysis Explanation:</b> Across <b>{analysis['frames_processed']}</b> analyzed frames,
            the tracker recorded <b>{analysis['total_unique_vehicles']}</b> unique vehicles with an average
            density of <b>{analysis['avg_vehicles_per_frame']:.2f} vehicles/frame</b>, classifying overall traffic
            as <b>{analysis['traffic_condition']}</b>. The most common vehicle type was
            <b>{top_cat['category']}</b> ({top_cat['count']} unique).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simple Traffic Forecasting
    st.subheader("Simple Traffic Volume Forecast (Moving Average)")
    if forecast.get("available"):
        st.altair_chart(build_forecast_chart(intervals, forecast), use_container_width=True)
        f1, f2 = st.columns(2)
        with f1:
            render_metric_card(
                "Estimated Next Interval",
                f"{forecast['estimated_avg_vehicles']:.2f} / frame",
                "🔮",
                f"{forecast['window']}-interval simple moving average",
                "bg-lavender",
            )
        with f2:
            render_metric_card(
                "Estimated Condition",
                forecast["estimated_condition"],
                "📈",
                "Statistical estimate (not guaranteed)",
                "bg-peach",
            )
        st.caption(forecast["explanation"])
    else:
        st.info(forecast.get("reason") or "Insufficient interval data for moving-average forecasting.")


def _render_congestion_map_tab(result: dict | None) -> None:
    if not result:
        _render_empty_notice("Congestion Map")
        return

    analysis = result["analysis"]
    intervals = analysis.get("intervals", [])
    zone = analysis["congestion_zone"]

    render_condition_banner(
        zone["zone"],
        f"Overall Video Density: {analysis['avg_vehicles_per_frame']:.2f} vehicles/frame • "
        f"{analysis['total_unique_vehicles']} unique vehicles",
    )

    components.html(render_junction_heatmap_svg(analysis), height=445)

    if intervals:
        st.altair_chart(build_interval_heatmap(intervals), use_container_width=True)

        with st.expander("Inspect Interval Density Data", expanded=False):
            df_intervals = pd.DataFrame(intervals)[
                ["interval_index", "label", "frames", "avg_vehicles", "peak_vehicles", "new_unique_vehicles", "zone"]
            ].rename(
                columns={
                    "interval_index": "#",
                    "label": "Interval",
                    "frames": "Frames",
                    "avg_vehicles": "Avg Vehicles / Frame",
                    "peak_vehicles": "Peak Frame Load",
                    "new_unique_vehicles": "New Unique Vehicles",
                    "zone": "Zone",
                }
            )
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)


def _render_advisory_tab(result: dict | None) -> None:
    if not result:
        _render_empty_notice("Traffic Advisory")
        return

    advisory = result["advisory"]
    analysis = result["analysis"]

    if advisory["status"] != "success":
        st.error(f"Could not generate advisory: {advisory.get('note', 'Unknown error')}")
        return

    render_condition_banner(
        analysis["traffic_condition"],
        f"Advisory Priority: {advisory['priority']} • Based on actual video counts",
    )

    c_obs, c_adv = st.columns(2)
    with c_obs:
        st.markdown(
            f"""
            <div class="tv-section-box" style="background:linear-gradient(135deg,#FFF9F0,#FFE9DC); min-height:155px;">
                <div style="font-weight:800; font-size:1rem; margin-bottom:0.45rem;">📋 Traffic Observation</div>
                <div style="font-size:0.95rem; line-height:1.55;">{advisory['traffic_observation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_adv:
        st.markdown(
            f"""
            <div class="tv-section-box" style="background:linear-gradient(135deg,#F9F5FF,#ECE2FF); min-height:155px;">
                <div style="font-weight:800; font-size:1rem; margin-bottom:0.45rem;">🚦 Authority Advisory</div>
                <div style="font-size:0.95rem; line-height:1.55;">{advisory['advisory']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    actions = advisory.get("recommended_actions") or []
    if actions:
        actions_html = "".join(f"<li style='margin-bottom:0.4rem;'>{a}</li>" for a in actions)
        st.markdown(
            f"""
            <div class="tv-section-box" style="background:linear-gradient(135deg,#F4FCF8,#E1F7EC);">
                <div style="font-weight:800; font-size:1rem; margin-bottom:0.5rem;">✅ Recommended Actions (for Human Review)</div>
                <ul style="margin:0; padding-left:1.25rem; font-size:0.94rem; line-height:1.5;">
                    {actions_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Advisory generated via Member 5's rule-based fallback engine (Gemini API disabled as configured). "
        "Suggestions are for human traffic authorities only."
    )


def _render_reports_tab(result: dict | None) -> None:
    if not result:
        _render_empty_notice("Reports")
        return

    analysis = result["analysis"]
    info = result["video_info"]
    cls_counts = analysis["class_counts"]

    st.markdown(
        f"""
        <div class="tv-section-box">
            <div style="font-weight:800; font-size:1.05rem; margin-bottom:0.5rem;">📄 Export Actual Pipeline Reports</div>
            <div style="font-size:0.92rem; color:#4A3B3B;">
                <b>Video:</b> {Path(info.get('path', 'video')).name} ({info.get('width', 0)}×{info.get('height', 0)}, {round(float(info.get('fps', 0)), 2)} fps) &nbsp;|&nbsp;
                <b>Frames:</b> {result['frames_processed']} &nbsp;|&nbsp;
                <b>Unique Vehicles:</b> {analysis['total_unique_vehicles']} &nbsp;|&nbsp;
                <b>Condition:</b> {analysis['traffic_condition']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_data = build_csv_report(result)
    json_data = build_json_report(result)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download CSV Vehicle-Count Report",
            data=csv_data,
            file_name="trafficvision_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Download JSON Analysis Report",
            data=json_data,
            file_name="trafficvision_report.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Category Summary Table")
    summary_df = pd.DataFrame(
        [
            {"Vehicle Category": CATEGORY_LABELS[k], "Unique Count": int(cls_counts.get(k, 0))}
            for k in ("car", "bus", "truck", "motorcycle")
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("Preview JSON Report", expanded=False):
        st.code(json_data, language="json")


def main() -> None:
    st.set_page_config(page_title="TrafficVision Dashboard", page_icon="🚦", layout="wide")
    inject_custom_css()

    st.markdown(
        """
        <div class="tv-hero">
            <div class="tv-hero-badge">Intelligent Traffic Monitoring System</div>
            <h1>TrafficVision Dashboard</h1>
            <p>End-to-end YOLO vehicle detection, multi-frame tracking, congestion mapping, moving-average forecasting, and traffic advisory.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.get("tv_result")
    conf_threshold, max_frames_val, force_reprocess = _render_sidebar(result)

    tab_overview, tab_video, tab_analytics, tab_map, tab_advisory, tab_reports = st.tabs(
        [
            "🏠 Overview",
            "🎬 Video Analysis",
            "📊 Traffic Analytics",
            "🗺️ Congestion Map",
            "🚦 Traffic Advisory",
            "📄 Reports",
        ]
    )

    with tab_overview:
        _render_overview_tab(st.session_state.get("tv_result"))

    with tab_video:
        _render_video_tab(conf_threshold, max_frames_val, force_reprocess)

    with tab_analytics:
        _render_analytics_tab(st.session_state.get("tv_result"))

    with tab_map:
        _render_congestion_map_tab(st.session_state.get("tv_result"))

    with tab_advisory:
        _render_advisory_tab(st.session_state.get("tv_result"))

    with tab_reports:
        _render_reports_tab(st.session_state.get("tv_result"))


if __name__ == "__main__":
    main()
