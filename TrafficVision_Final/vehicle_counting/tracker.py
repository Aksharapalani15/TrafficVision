import logging
import math

logger = logging.getLogger(__name__)


def get_center(bbox):
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2
    return cx, cy


def calculate_distance(center1, center2):
    x1, y1 = center1
    x2, y2 = center2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class VehicleTracker:
    def __init__(
        self,
        max_distance=None,
        max_missed=10,
        min_hits=3,
        counting_line_x=None,
        lane_boundaries=None,
        frame_width=None,
        frame_height=None,
    ):
        self.next_id = 1
        self.tracks = {}

        if max_distance is None:
            if frame_width and frame_height:
                self.max_distance = max(50, int(0.03 * math.hypot(frame_width, frame_height)))
            else:
                self.max_distance = 50
        else:
            self.max_distance = max_distance

        self.max_missed = max_missed
        self.min_hits = max(1, int(min_hits))

        # Line-crossing attributes preserved from Member 2's original tracker.
        self.counted_ids = set()
        self.vehicle_count = 0
        self.counting_line_x = (
            int(frame_width // 2) if counting_line_x is None and frame_width else (counting_line_x or 960)
        )

        self.minute_counts = {}

        if lane_boundaries is not None:
            self.lane_boundaries = lane_boundaries
        elif frame_height:
            self.lane_boundaries = [
                int(0.28 * frame_height),
                int(0.43 * frame_height),
                int(0.60 * frame_height),
            ]
        else:
            self.lane_boundaries = [300, 459, 647]

        self.vehicle_categories = {
            "cars": 0,
            "buses": 0,
            "trucks": 0,
            "two_wheelers": 0,
        }

        self.class_mapping = {
            "car": "cars",
            "bus": "buses",
            "truck": "trucks",
            "motorcycle": "two_wheelers",
            "motorbike": "two_wheelers",
            "two_wheeler": "two_wheelers",
        }

        # Unique-vehicle counting across any camera orientation or direction.
        self.unique_counted_ids = set()
        self.unique_category_counts = self.vehicle_categories.copy()
        self.unique_class_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }
        self._track_history = {}

    def count_crossing(self, vehicle_id, current_center, timestamp):
        current_x, current_y = current_center

        if vehicle_id not in self.tracks:
            return

        previous_x = self.tracks[vehicle_id]["previous_x"]

        if (
            previous_x > self.counting_line_x
            and current_x <= self.counting_line_x
            and vehicle_id not in self.counted_ids
        ):
            self.vehicle_count += 1
            self.counted_ids.add(vehicle_id)

            ts = timestamp if timestamp is not None else 0.0
            minute = int(ts // 60)

            lane = self.get_lane(current_y)

            vehicle_class = self.tracks[vehicle_id]["class"].lower()
            category = self.class_mapping.get(vehicle_class)

            if minute not in self.minute_counts:
                self.minute_counts[minute] = {
                    "lane_1": self.vehicle_categories.copy(),
                    "lane_2": self.vehicle_categories.copy(),
                    "lane_3": self.vehicle_categories.copy(),
                    "lane_4": self.vehicle_categories.copy(),
                }

            if category:
                self.minute_counts[minute][lane][category] += 1

            logger.debug(
                "Vehicle crossed: ID %s, %s, Total line count: %s",
                vehicle_id,
                lane,
                self.vehicle_count,
            )

    def _confirm_unique_vehicle(self, vehicle_id, vehicle_class):
        """Record a tracked vehicle once it meets the minimum hit threshold."""
        if vehicle_id in self.unique_counted_ids:
            return
        self.unique_counted_ids.add(vehicle_id)
        cls_name = vehicle_class.lower()
        if cls_name in self.unique_class_counts:
            self.unique_class_counts[cls_name] += 1
        else:
            self.unique_class_counts[cls_name] = 1
        category = self.class_mapping.get(cls_name)
        if category:
            self.unique_category_counts[category] += 1

    def update(self, detections, timestamp=0.0):
        tracked_vehicles = []
        matched_ids = set()

        for detection in detections:
            bbox = detection["bbox"]
            center = get_center(bbox)

            vehicle_id = None
            min_distance = self.max_distance

            for existing_id, track in self.tracks.items():
                if existing_id in matched_ids:
                    continue

                if track["class"] != detection["class"]:
                    continue

                distance = calculate_distance(center, track["center"])

                if distance < min_distance:
                    min_distance = distance
                    vehicle_id = existing_id

            if vehicle_id is None:
                vehicle_id = self.next_id
                self.next_id += 1
                previous_x = center[0]
                hits = 1
            else:
                previous_x = self.tracks[vehicle_id]["center"][0]
                hits = self.tracks[vehicle_id].get("hits", 1) + 1

            matched_ids.add(vehicle_id)

            self.tracks[vehicle_id] = {
                "center": center,
                "previous_x": previous_x,
                "class": detection["class"],
                "missed": 0,
                "hits": hits,
            }
            self._track_history[vehicle_id] = {
                "class": detection["class"],
                "hits": hits,
            }

            if hits >= self.min_hits:
                self._confirm_unique_vehicle(vehicle_id, detection["class"])

            self.count_crossing(vehicle_id, center, timestamp)

            tracked_vehicles.append({
                "id": vehicle_id,
                "class": detection["class"],
                "bbox": bbox,
            })

        for vehicle_id in self.tracks:
            if vehicle_id not in matched_ids:
                self.tracks[vehicle_id]["missed"] += 1

        expired_ids = []

        for vehicle_id, track in self.tracks.items():
            if track["missed"] > self.max_missed:
                expired_ids.append(vehicle_id)

        for vehicle_id in expired_ids:
            del self.tracks[vehicle_id]

        return tracked_vehicles

    def finalize(self, frames_processed=None):
        """Ensure very short videos (< min_hits frames) still count tracked vehicles."""
        if frames_processed is not None and frames_processed < self.min_hits:
            for vehicle_id, info in self._track_history.items():
                self._confirm_unique_vehicle(vehicle_id, info["class"])

    @property
    def total_unique_vehicles(self):
        return len(self.unique_counted_ids)

    def get_summary(self):
        return {
            "total_unique_vehicles": self.total_unique_vehicles,
            "category_counts": dict(self.unique_category_counts),
            "class_counts": dict(self.unique_class_counts),
            "line_crossing_count": self.vehicle_count,
            "minute_counts": self.minute_counts,
        }

    def get_lane(self, center_y):
        if center_y < self.lane_boundaries[0]:
            return "lane_1"
        elif center_y < self.lane_boundaries[1]:
            return "lane_2"
        elif center_y < self.lane_boundaries[2]:
            return "lane_3"
        else:
            return "lane_4"
