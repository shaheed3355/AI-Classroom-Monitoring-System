import cv2
import os
import numpy as np
import mediapipe as mp

# 1. SETUP PATHS
dataset_path = "face_dataset"
face_model_path = "face_model.yml"
names_file = "face_names.txt"

# 2. INITIALIZE MEDIAPIPE (Much more accurate than Haar Cascades)
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 3. INITIALIZE LBPH RECOGNIZER
# Optimized parameters for classroom distances
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

faces = []
labels = []
names = []
label_id = 0

print("🔍 Starting Training with CLAHE Enhancement...")

# 4. PROCESS DATASET
for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    names.append(person_name)
    print(f"📁 Training for: {person_name}")

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_img)

        if results.detections:
            for detection in results.detections:
                # Get Bounding Box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # Crop face and convert to Gray
                face_crop = img[max(0, y):y+bh, max(0, x):x+bw]
                if face_crop.size == 0: continue
                
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                # --- ADVANCED LIGHTING CORRECTION ---
                # This matches the correction in our image_monitor.py
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_face = clahe.apply(gray_face)
                
                # Resize to standard size
                gray_face = cv2.resize(gray_face, (200, 200))

                faces.append(gray_face)
                labels.append(label_id)
                break # Only one face per training image

    label_id += 1

# 5. TRAIN & SAVE
if len(faces) > 0:
    recognizer.train(faces, np.array(labels))
    recognizer.save(face_model_path)
    
    with open(names_file, "w") as f:
        for name in names:
            f.write(name + "\n")
            
    print(f"✅ Training Complete! {len(faces)} images trained for {len(names)} people.")
    print(f"📂 Model saved as {face_model_path}")
else:
    print("❌ Error: No faces detected in the dataset folders.")