import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import json

from vehicle_detection import detect_frame, load_model
from vehicle_counting.tracker import VehicleTracker


model = load_model()
tracker = VehicleTracker()

video_path = "sample_data/Traffic1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video width:", frame_width)
print("Video height:", frame_height)

frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id % 100 == 0:
        print("Processed frames:", frame_id)

    timestamp = (frame_id - 1) / fps

    result = detect_frame(
        frame,
        frame_id=frame_id,
        timestamp=timestamp,
        model=model
    )

    detections = result["detections"]

    tracked_vehicles = tracker.update(detections, timestamp)

    line_x = tracker.counting_line_x

    cv2.line(
        frame,
        (line_x, 0),
        (line_x, frame.shape[0]),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        f"Vehicle Count: {tracker.vehicle_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    for vehicle in tracked_vehicles:

        bbox = vehicle["bbox"]
        vehicle_id = vehicle["id"]
        vehicle_class = vehicle["class"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{vehicle_class} ID: {vehicle_id}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("TrafficVision - Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


print("\nMINUTE-WISE, LANE-WISE VEHICLE COUNT")

for minute, lanes in sorted(tracker.minute_counts.items()):

    print(f"\nMinute {minute}")

    for lane, counts in lanes.items():
        print(f"{lane}: {counts}")


print("\nTotal vehicles:", tracker.vehicle_count)


json_data = []

for minute, lanes in sorted(tracker.minute_counts.items()):

    record = {
        "minute": minute,
        "lanes": lanes
    }

    json_data.append(record)


with open("vehicle_counting/vehicle_counts.json", "w") as file:
    json.dump(json_data, file, indent=4)


print("\nLane-wise JSON file saved successfully!")
print("Total vehicles:", tracker.vehicle_count)
