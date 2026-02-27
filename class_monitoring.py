import cv2
import numpy as np
from ultralytics import YOLO

# LOAD MODELS
person_detector = YOLO("yolov8s.pt")
yolo_activity = YOLO("best.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("face_names.txt", "r") as f:
    names = f.read().splitlines()

# WEBCAM
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECTION
    res_p = person_detector(frame, conf=0.35)
    res_a = yolo_activity(frame, conf=0.10, imgsz=640)

    students = []
    phones = []

    # PERSON + PHONE DETECTION
    for b in res_p[0].boxes:
        cls_id = int(b.cls[0])
        c = b.xyxy[0].cpu().numpy().astype(int).tolist()

        if cls_id == 0:
            students.append({"box": c, "name": "Unknown", "action": "Attentive"})

        if cls_id == 67:
            phones.append(c)

    # ACTIVITY DETECTIONS
    acts = []
    for b in res_a[0].boxes:
        c = b.xyxy[0].cpu().numpy().astype(int).tolist()
        lbl = yolo_activity.names[int(b.cls[0])].lower()
        acts.append({"box": c, "label": lbl})

    # PROCESS STUDENTS
    for s in students:
        sx1, sy1, sx2, sy2 = s["box"]

        person_gray = gray[sy1:sy2, sx1:sx2]
        faces = face_cascade.detectMultiScale(person_gray, 1.2, 5)

        for (fx, fy, fw, fh) in faces:
            face_roi = person_gray[fy:fy+fh, fx:fx+fw]
            face_roi = cv2.resize(face_roi, (200, 200))

            id_, dist = recognizer.predict(face_roi)

            if id_ < len(names) and dist < 70:
                s["name"] = names[id_]

        # PHONE DETECTION
        for px1, py1, px2, py2 in phones:
            if not (px2 < sx1 or px1 > sx2 or py2 < sy1 or py1 > sy2):
                s["action"] = "Using Mobile"

        # ACTIVITY ASSIGNMENT
        found_acts = []
        for a in acts:
            ax1, ay1, ax2, ay2 = a["box"]

            ix1, iy1 = max(sx1, ax1), max(sy1, ay1)
            ix2, iy2 = min(sx2, ax2), min(sy2, ay2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                act_area = (ax2 - ax1) * (ay2 - ay1)

                if (inter_area / act_area) > 0.20:
                    found_acts.append(a["label"])

        if s["action"] != "Using Mobile":
            if "sleep" in found_acts:
                s["action"] = "Sleeping"
            elif "hand-raising" in found_acts:
                s["action"] = "Hand Raising"
            elif "writing" in found_acts:
                s["action"] = "Writing"
            elif "reading" in found_acts:
                s["action"] = "Reading"

    # DRAW OUTPUT
    for s in students:
        x1, y1, x2, y2 = s["box"]

        if s["action"] == "Using Mobile":
            color = (0, 0, 255)
        elif s["name"] == "Unknown":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{s['name']} : {s['action']}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Classroom Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()