"""SAMPLE ONLY - shows how the Streamlit dashboard can display the report.
Not part of the tested module; requires `pip install streamlit`.
Run with:  streamlit run examples/streamlit_example.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisory import generate_advisory  # noqa: E402

st.title("AI Traffic Advisory")

# In the real dashboard these values come from your teammates' modules.
cars = st.number_input("Cars", 0, 100000, 120)
buses = st.number_input("Buses", 0, 100000, 20)
trucks = st.number_input("Trucks", 0, 100000, 10)
two_wheelers = st.number_input("Two-wheelers", 0, 100000, 80)
congestion = st.selectbox("Congestion", ["Free", "Moderate", "Heavy"], index=2)

if st.button("Generate advisory"):
    report = generate_advisory({"cars": cars, "buses": buses, "trucks": trucks,
                                "two_wheelers": two_wheelers, "congestion": congestion})
    if report["status"] != "success":
        st.error(report["note"])
    else:
        {"Low": st.success, "Medium": st.warning, "High": st.error}[report["priority"]](
            f"Priority: {report['priority']}")
        st.subheader("Observation");  st.write(report["traffic_observation"])
        st.subheader("Advisory");     st.write(report["advisory"])
        st.subheader("Recommended actions (for human review)")
        for action in report["recommended_actions"]:
            st.markdown(f"- {action}")
        if report["prediction_summary"]:
            st.subheader("Prediction"); st.write(report["prediction_summary"])
        if report["source"] == "fallback":
            st.caption("AI service unavailable - showing rule-based advisory.")
        st.caption("AI-generated suggestions. Final decisions rest with traffic authorities.")
