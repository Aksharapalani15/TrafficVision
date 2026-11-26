"""How a teammate's module calls the advisory function.
Run from the project folder:  python examples/call_from_other_module.py
(Copy the traffic_advisory folder next to your code, or add it to sys.path as below.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisory import generate_advisory  # noqa: E402

# Pretend these came from the counting + forecasting modules.
traffic = {"cars": 120, "buses": 20, "trucks": 10, "two_wheelers": 80, "congestion": "Heavy"}
prediction = None  # or {"predicted_congestion": "Moderate", "predicted_vehicle_count": 90}

report = generate_advisory(traffic, prediction)

if report["status"] == "success":
    print("Priority:   ", report["priority"])
    print("Observation:", report["traffic_observation"])
    print("Advisory:   ", report["advisory"])
    for action in report["recommended_actions"]:
        print(" -", action)
    print("Source:     ", report["source"])
else:
    print("Input problem:", report["note"])
