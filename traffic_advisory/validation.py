"""validation.py - checks and cleans input BEFORE anything reaches the LLM."""

VEHICLE_FIELDS = ("cars", "buses", "trucks", "two_wheelers")
CONGESTION_LEVELS = ("Free", "Moderate", "Heavy")
MAX_COUNT = 100_000  # sanity limit for a single junction snapshot


class ValidationError(ValueError):
    """Raised when the supplied traffic data is unusable."""


def _clean_count(name, value, errors):
    """Return a non-negative int, or None (and record an error)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"'{name}' must be a number, got {type(value).__name__}.")
        return None
    if isinstance(value, float):
        if not value.is_integer():  # also rejects NaN and infinity
            errors.append(f"'{name}' must be a whole number, got {value}.")
            return None
        value = int(value)
    if value < 0:
        errors.append(f"'{name}' cannot be negative (got {value}).")
        return None
    if value > MAX_COUNT:
        errors.append(f"'{name}' is unrealistically large (got {value}).")
        return None
    return int(value)


def _clean_congestion(name, value, errors):
    """Accept 'heavy', ' HEAVY ' etc. and return 'Heavy'."""
    if not isinstance(value, str):
        errors.append(f"'{name}' must be one of {list(CONGESTION_LEVELS)}.")
        return None
    for level in CONGESTION_LEVELS:
        if value.strip().lower() == level.lower():
            return level
    errors.append(f"Invalid congestion level '{value}' for '{name}'. "
                  f"Allowed: {', '.join(CONGESTION_LEVELS)}.")
    return None


def validate_traffic_data(traffic_data, prediction_data=None):
    """Validate input and return a clean dictionary.

    Returns:
        {"vehicle_counts": {...}, "total_vehicles": int, "congestion": str,
         "prediction": None | {"predicted_congestion", "predicted_vehicle_count"}}
    Raises:
        ValidationError listing every problem found.
    """
    if not traffic_data:
        raise ValidationError("No traffic data was supplied.")
    if not isinstance(traffic_data, dict):
        raise ValidationError("Traffic data must be a dictionary (JSON object).")

    errors = []
    counts = {}
    for field in VEHICLE_FIELDS:
        if field not in traffic_data or traffic_data[field] is None:
            errors.append(f"Missing required field '{field}'.")
        else:
            counts[field] = _clean_count(field, traffic_data[field], errors)

    if "congestion" not in traffic_data or traffic_data["congestion"] is None:
        errors.append("Missing required field 'congestion'.")
        congestion = None
    else:
        congestion = _clean_congestion("congestion", traffic_data["congestion"], errors)

    # Optional prediction block (None or {} means "no prediction").
    prediction = None
    if prediction_data:
        if not isinstance(prediction_data, dict):
            errors.append("Prediction data must be a dictionary (JSON object).")
        else:
            predicted = None
            if prediction_data.get("predicted_congestion") is None:
                errors.append("Prediction data needs 'predicted_congestion'.")
            else:
                predicted = _clean_congestion("predicted_congestion",
                                              prediction_data["predicted_congestion"], errors)
            count = None
            if prediction_data.get("predicted_vehicle_count") is not None:
                count = _clean_count("predicted_vehicle_count",
                                     prediction_data["predicted_vehicle_count"], errors)
            if prediction_data.get("current_congestion") is not None:
                cur = _clean_congestion("current_congestion",
                                        prediction_data["current_congestion"], errors)
                if cur and congestion and cur != congestion:
                    errors.append(f"Prediction 'current_congestion' ({cur}) does not match "
                                  f"traffic 'congestion' ({congestion}).")
            if predicted:
                prediction = {"predicted_congestion": predicted,
                              "predicted_vehicle_count": count}

    if errors:
        raise ValidationError(" ".join(errors))

    return {"vehicle_counts": counts, "total_vehicles": sum(counts.values()),
            "congestion": congestion, "prediction": prediction}
