import cv2
from ultralytics import YOLO

print("Starting activity detection...")

# Load trained YOLO model
model = YOLO("runs/detect/train11/weights/best.pt")

cap = cv2.VideoCapture(0)

print("Opening webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLO detection
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        # Convert class ID → activity name
        activity_name = model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, activity_name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Activity Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()