"""sample_data.py - illustrative inputs (NOT real traffic measurements)."""

SAMPLES = {
    "free":     ({"cars": 15, "buses": 1, "trucks": 0, "two_wheelers": 10, "congestion": "Free"}, None),
    "moderate": ({"cars": 60, "buses": 8, "trucks": 4, "two_wheelers": 40, "congestion": "Moderate"}, None),
    "heavy":    ({"cars": 120, "buses": 20, "trucks": 10, "two_wheelers": 80, "congestion": "Heavy"}, None),
    "rising":   ({"cars": 60, "buses": 8, "trucks": 4, "two_wheelers": 40, "congestion": "Moderate"},
                 {"current_congestion": "Moderate", "predicted_congestion": "Heavy",
                  "predicted_vehicle_count": 280}),
    "easing":   ({"cars": 120, "buses": 20, "trucks": 10, "two_wheelers": 80, "congestion": "Heavy"},
                 {"current_congestion": "Heavy", "predicted_congestion": "Moderate",
                  "predicted_vehicle_count": 90}),
}
