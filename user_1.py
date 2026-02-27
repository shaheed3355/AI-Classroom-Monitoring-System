import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter

# --- 1. SETUP ---
person_detector = YOLO("yolov8s.pt") 
yolo_activity = YOLO("best.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("face_names.txt", "r") as f:
    names = f.read().splitlines()

# Storage
locked_names = {}
recognition_votes = {} 
activity_history = {}
SMOOTH_FRAMES = 30 # For stable activity switching

cap = cv2.VideoCapture("hod\class1234.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Run detections
    res_p = person_detector.track(frame, persist=True, conf=0.25, verbose=False)
    res_a = yolo_activity(frame, conf=0.20, verbose=False)

    # 1. Map all phone detections (Class 67)
    phones = []
    if res_p[0].boxes is not None:
        for b in res_p[0].boxes:
            if int(b.cls[0]) == 67:
                phones.append(b.xyxy[0].cpu().numpy().astype(int).tolist())

    # 2. Map custom activities
    acts = [{"box": b.xyxy[0].cpu().numpy().astype(int).tolist(), 
             "label": yolo_activity.names[int(b.cls[0])].lower()} for b in res_a[0].boxes]

    if res_p[0].boxes.id is not None:
        track_ids = res_p[0].boxes.id.int().cpu().tolist()
        boxes = res_p[0].boxes.xyxy.int().cpu().tolist()

        for track_id, box in zip(track_ids, boxes):
            sx1, sy1, sx2, sy2 = box
            
            # --- IDENTITY LOCK (STAYS FIXED) ---
            if locked_names.get(track_id, "Unknown") == "Unknown":
                if track_id not in recognition_votes: recognition_votes[track_id] = []
                
                roi_gray = cv2.cvtColor(frame[max(0,sy1):sy2, max(0,sx1):sx2], cv2.COLOR_BGR2GRAY)
                if roi_gray.size > 0:
                    roi_gray = cv2.equalizeHist(roi_gray)
                    faces = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
                    for (fx, fy, fw, fh) in faces:
                        face_crop = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                        id_label, dist = recognizer.predict(face_crop)
                        if dist < 85: # Threshold for high accuracy
                            recognition_votes[track_id].append(names[id_label])
                
                # Verify over 10 frames before locking name forever
                if len(recognition_votes[track_id]) >= 10:
                    top_vote = Counter(recognition_votes[track_id]).most_common(1)[0]
                    if top_vote[1] >= 7: # 70% agreement required
                        locked_names[track_id] = top_vote[0]

            # --- ACTIVITY DETECTION ---
            raw_act = "Attentive"
            # Check Phone overlap first
            for px1, py1, px2, py2 in phones:
                if not (px2 < sx1 or px1 > sx2 or py2 < sy1 or py1 > sy2):
                    raw_act = "Using Mobile"
                    break

            # If no phone, check custom actions
            if raw_act == "Attentive":
                for a in acts:
                    ax1, ay1, ax2, ay2 = a["box"]
                    if "phone" in a["label"] or "mobile" in a["label"]:
                        if not (ax2 < sx1 or ax1 > sx2 or ay2 < sy1 or ay1 > sy2):
                            raw_act = "Using Mobile"
                            break
                    else:
                        # Logic for Reading, Writing, Sleeping
                        ix1, iy1, ix2, iy2 = max(sx1, ax1), max(sy1, ay1), min(sx2, ax2), min(sy2, ay2)
                        if ix2 > ix1 and iy2 > iy1:
                            if ((ix2-ix1)*(iy2-iy1)) / ((ax2-ax1)*(ay2-ay1)) > 0.25:
                                raw_act = a["label"].title()

            # --- SMOOTHING ---
            if track_id not in activity_history: activity_history[track_id] = deque(maxlen=SMOOTH_FRAMES)
            activity_history[track_id].append(raw_act)
            smoothed_act = Counter(activity_history[track_id]).most_common(1)[0][0]

            # --- BOX COLOR LOGIC ---
            name = locked_names.get(track_id, "Unknown")
            
            if "Mobile" in smoothed_act:
                color = (0, 0, 255)   # RED for Mobile
            elif name == "Unknown":
                color = (0, 255, 255) # YELLOW for Unknown
            else:
                color = (0, 255, 0)   # GREEN for Identified & Safe

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(frame, f"{name} : {smoothed_act}", (sx1, sy1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("Red Alert Classroom Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          