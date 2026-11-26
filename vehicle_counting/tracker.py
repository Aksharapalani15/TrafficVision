
import math


def get_center(bbox):
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2
    return cx, cy


def calculate_distance(center1, center2):
    x1, y1 = center1
    x2, y2 = center2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class VehicleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = 50
        self.max_missed = 10

        self.counted_ids = set()
        self.vehicle_count = 0
        self.counting_line_x = 960

        self.minute_counts = {}

        self.lane_boundaries = [300, 459, 647]

        self.vehicle_categories = {
            "cars": 0,
            "buses": 0,
            "trucks": 0,
            "two_wheelers": 0
        }

        self.class_mapping = {
            "car": "cars",
            "bus": "buses",
            "truck": "trucks",
            "motorcycle": "two_wheelers",
            "motorbike": "two_wheelers",
            "two_wheeler": "two_wheelers"
        }

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

            minute = int(timestamp // 60)

            lane = self.get_lane(current_y)

            vehicle_class = self.tracks[vehicle_id]["class"].lower()
            category = self.class_mapping.get(vehicle_class)

            if minute not in self.minute_counts:
                self.minute_counts[minute] = {
                    "lane_1": self.vehicle_categories.copy(),
                    "lane_2": self.vehicle_categories.copy(),
                    "lane_3": self.vehicle_categories.copy(),
                    "lane_4": self.vehicle_categories.copy()
                }

            if category:
                self.minute_counts[minute][lane][category] += 1

            print(
                f"Vehicle crossed: ID {vehicle_id}, "
                f"{lane}, Total: {self.vehicle_count}"
            )

    def update(self, detections, timestamp):
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
            else:
                previous_x = self.tracks[vehicle_id]["center"][0]

            matched_ids.add(vehicle_id)

            self.tracks[vehicle_id] = {
                "center": center,
                "previous_x": previous_x,
                "class": detection["class"],
                "missed": 0
            }

            self.count_crossing(vehicle_id, center, timestamp)

            tracked_vehicles.append({
                "id": vehicle_id,
                "class": detection["class"],
                "bbox": bbox
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

    def get_lane(self, center_y):
        if center_y < self.lane_boundaries[0]:
            return "lane_1"
        elif center_y < self.lane_boundaries[1]:
            return "lane_2"
        elif center_y < self.lane_boundaries[2]:
            return "lane_3"
        else:
            return "lane_4"
