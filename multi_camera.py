import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

class ThreadedCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Auto-loop

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- 1. SETUP MODELS ---
# Using 'n' model for speed in multi-cam environments
person_model = YOLO("yolov8n.pt") 
activity_model = YOLO("best.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("face_names.txt", "r") as f:
    names = f.read().splitlines()

# Tracking Storage
id_labels = {} 

# Initialize 3 Cameras
sources = ["hod/cam1.mp4", "hod/cam2.mp4", "hod/cam3.mp4"]
caps = [ThreadedCamera(s).start() for s in sources]
time.sleep(2)

def process_frame(frame, cam_idx):
    if frame is None: return None
    
    # 2. DETECT & TRACK PERSONS (Main pass)
    per_results = person_model.track(frame, persist=True, conf=0.3, verbose=False)

    if per_results[0].boxes is not None and per_results[0].boxes.id is not None:
        ids = per_results[0].boxes.id.int().cpu().tolist()
        boxes = per_results[0].boxes.xyxy.int().cpu().tolist()

        for track_id, p_box in zip(ids, boxes):
            uid = f"c{cam_idx}_{track_id}"
            px1, py1, px2, py2 = p_box

            # --- ROI CROP FOR BETTER ACTIVITY DETECTION ---
            # We crop the person so the activity model can see small objects (phones) clearly
            person_roi = frame[max(0, py1):min(frame.shape[0], py2), 
                               max(0, px1):min(frame.shape[1], px2)]
            
            current_act = "Attentive"
            if person_roi.size > 0:
                # Run activity model on the crop with low threshold for small objects
                act_results = activity_model(person_roi, conf=0.15, verbose=False)
                if len(act_results[0].boxes) > 0:
                    # Get the most confident activity
                    top_act_idx = int(act_results[0].boxes[0].cls[0])
                    current_act = activity_model.names[top_act_idx]

            # --- FACE RECOGNITION (Inside Person Box) ---
            name = id_labels.get(uid, "Unknown")
            if name == "Unknown":
                # Look at the top 45% of the person for the face
                head_roi = person_roi[0:int(person_roi.shape[0]*0.45), :]
                if head_roi.size > 100:
                    gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_head, 1.1, 3)
                    for (fx, fy, fw, fh) in faces:
                        f_roi = cv2.resize(gray_head[fy:fy+fh, fx:fx+fw], (200, 200))
                        idx, dist = recognizer.predict(f_roi)
                        if dist < 100: # Forgiving threshold for CCTV
                            id_labels[uid] = names[idx]
                            name = names[idx]

            # --- COLOR PRIORITY LOGIC ---
            # 1. Start with Yellow (Unknown)
            color = (0, 255, 255) 
            
            # 2. Switch to Green if Identified
            if name != "Unknown":
                color = (0, 255, 0)
            
            # 3. Switch to Red if Mobile Detected (Highest Priority)
            act_lower = current_act.lower()
            if "mobile" in act_lower or "phone" in act_lower:
                color = (0, 0, 255)
                current_act = "USING MOBILE"

            # 4. DRAWING
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            # Tag background
            cv2.rectangle(frame, (px1, py1-22), (px1+180, py1), color, -1)
            cv2.putText(frame, f"{name}: {current_act}", (px1+5, py1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    return frame

# --- MAIN LOOP ---
while True:
    all_cam_frames = []
    for i, cap in enumerate(caps):
        f = cap.read()
        if f is not None:
            processed = process_frame(f.copy(), i)
            all_cam_frames.append(cv2.resize(processed, (400, 300)))

    if len(all_cam_frames) == len(sources):
        combined = cv2.hconcat(all_cam_frames)
        cv2.imshow("Multi-CCTV Advanced Monitor", combined)

    if cv2.waitKey(1) & 0xFF == 27: break # ESC to quit

for cap in caps: cap.stop()
cv2.destroyAllWindows()