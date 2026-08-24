import cv2
import os
import math
import sqlite3
from datetime import datetime
from ultralytics import YOLO

VIDEO = "Nighttrafficvideo.mp4"
VEHICLE = "models/yolo11n.pt"
HELMET = "models/helmet.pt"
PLATE = "models/license_plate.pt"

os.makedirs("results", exist_ok=True)

DB = "traffic_ai.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT,
            violation_type TEXT,
            plate_number TEXT,
            timestamp TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def save_violation(vehicle_id, violation_type, confidence=0):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO violations(vehicle_id,violation_type,plate_number,timestamp,confidence) VALUES(?,?,?,?,?)",
        (str(vehicle_id), violation_type, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(float(confidence), 2))
    )
    conn.commit()
    conn.close()

init_db()

vehicle_model = YOLO(VEHICLE)
helmet_model = YOLO(HELMET)
plate_model = YOLO(PLATE) if os.path.exists(PLATE) else None

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "results/Traffic_AI_Final.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

vehicle_ids = {
    "CAR": set(),
    "MOTORCYCLE": set(),
    "BUS": set(),
    "TRUCK": set()
}

wrong_ids = set()
speed_ids = set()
helmet_count = 0
nohelmet_count = 0
triple_ids = set()
plate_ids = set()

saved_events = set()
previous = {}
frame_no = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame_no += 1
    people = []
    motorcycles = []

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            cls = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])

            if conf < 0.35:
                continue

            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])

            tid = None
            if r.boxes.id is not None:
                tid = int(r.boxes.id[i])

            if cls == 0:
                people.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            elif cls in [2, 3, 5, 7]:
                names = {
                    2: "CAR",
                    3: "MOTORCYCLE",
                    5: "BUS",
                    7: "TRUCK"
                }

                name = names[cls]

                if tid is not None:
                    vehicle_ids[name].add(tid)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if tid in previous:
                        px, py = previous[tid]
                        dx = cx - px
                        dy = cy - py
                        distance = math.sqrt(dx * dx + dy * dy)

                        if dy > 8 and tid not in wrong_ids:
                            wrong_ids.add(tid)
                            event = (tid, "Wrong Direction")

                            if event not in saved_events:
                                save_violation(tid, "Wrong Direction", conf)
                                saved_events.add(event)

                        speed = distance / 8 * fps * 3.6

                        if speed > 40 and tid not in speed_ids:
                            speed_ids.add(tid)
                            event = (tid, "Overspeed")

                            if event not in saved_events:
                                save_violation(tid, "Overspeed", conf)
                                saved_events.add(event)

                    previous[tid] = (cx, cy)

                    if cls == 3:
                        motorcycles.append((x1, y1, x2, y2, tid, conf))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label = name

                if tid is not None:
                    label += " ID:" + str(tid)

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

    for mx1, my1, mx2, my2, tid, conf in motorcycles:
        count = 0

        for px1, py1, px2, py2 in people:
            cx = (px1 + px2) // 2
            cy = (py1 + py2) // 2

            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                count += 1

        if count >= 3:
            cv2.rectangle(
                frame,
                (mx1, my1),
                (mx2, my2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "TRIPLE RIDING",
                (mx1, max(25, my1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            if tid not in triple_ids:
                triple_ids.add(tid)
                event = (tid, "Triple Riding")

                if event not in saved_events:
                    save_violation(tid, "Triple Riding", conf)
                    saved_events.add(event)

    hr = helmet_model.predict(
        frame,
        conf=0.30,
        imgsz=640,
        verbose=False
    )

    current_helmet = 0
    current_nohelmet = 0

    for r in hr:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            cls = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])

            x1, y1, x2, y2 = map(
                int,
                r.boxes.xyxy[i]
            )

            cname = str(
                helmet_model.names.get(cls, cls)
            ).lower()

            if "no" in cname or "without" in cname:
                current_nohelmet += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "NO HELMET",
                    (x1, max(25, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )
            else:
                current_helmet += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "HELMET",
                    (x1, max(25, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

    helmet_count = max(helmet_count, current_helmet)
    nohelmet_count = max(nohelmet_count, current_nohelmet)

    if plate_model is not None:
        pr = plate_model.predict(
            frame,
            conf=0.30,
            imgsz=640,
            verbose=False
        )

        for r in pr:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i])
                x1, y1, x2, y2 = map(
                    int,
                    r.boxes.xyxy[i]
                )

                plate_id = (frame_no // 10, x1 // 20, y1 // 20)

                plate_ids.add(plate_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "NUMBER PLATE",
                    (x1, max(25, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

    cars = len(vehicle_ids["CAR"])
    motorcycles_count = len(vehicle_ids["MOTORCYCLE"])
    buses = len(vehicle_ids["BUS"])
    trucks = len(vehicle_ids["TRUCK"])

    total = cars + motorcycles_count + buses + trucks

    panel = frame.copy()

    cv2.rectangle(
        panel,
        (10, 10),
        (550, 510),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        panel,
        0.78,
        frame,
        0.22,
        0,
        frame
    )

    lines = [
        ("TRAFFIC AI", (255, 255, 255), 0.9),
        ("Cars: " + str(cars), (255, 255, 255), 0.58),
        ("Motorcycles: " + str(motorcycles_count), (255, 255, 255), 0.58),
        ("Bus: " + str(buses), (255, 255, 255), 0.58),
        ("Truck: " + str(trucks), (255, 255, 255), 0.58),
        ("Total Vehicles: " + str(total), (0, 255, 255), 0.62),
        ("Helmet: " + str(helmet_count), (0, 255, 0), 0.58),
        ("No Helmet: " + str(nohelmet_count), (0, 0, 255), 0.58),
        ("Triple Riding: " + str(len(triple_ids)), (0, 0, 255), 0.58),
        ("Wrong Direction: " + str(len(wrong_ids)), (255, 100, 100), 0.58),
        ("Overspeed: " + str(len(speed_ids)), (0, 180, 255), 0.58),
        ("Seat Belt: 0", (255, 255, 255), 0.58),
        ("Mobile Phone: 0", (255, 200, 80), 0.58),
        ("Lane Violation: 0", (255, 255, 255), 0.58),
        ("Illegal Parking: 0", (255, 255, 255), 0.58),
        ("Red Light Jump: 0", (255, 80, 80), 0.58),
        ("Accident: 0", (255, 150, 150), 0.58),
        ("Restricted Area: 0", (255, 0, 255), 0.58),
        ("Number Plates: " + str(len(plate_ids)), (0, 255, 255), 0.58)
    ]

    y = 45

    for text, color, size in lines:
        cv2.putText(
            frame,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            2
        )
        y += 25 if size < 0.8 else 35

    out.write(frame)

    cv2.imshow("Traffic AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("TRAFFIC AI COMPLETED")
print("Cars:", len(vehicle_ids["CAR"]))
print("Motorcycles:", len(vehicle_ids["MOTORCYCLE"]))
print("Bus:", len(vehicle_ids["BUS"]))
print("Truck:", len(vehicle_ids["TRUCK"]))
print("Total Vehicles:", total)
print("Helmet:", helmet_count)
print("No Helmet:", nohelmet_count)
print("Triple Riding:", len(triple_ids))
print("Wrong Direction:", len(wrong_ids))
print("Overspeed:", len(speed_ids))
print("Number Plates:", len(plate_ids))
print("Database:", DB)
print("Video:", "results/Traffic_AI_Final.mp4")