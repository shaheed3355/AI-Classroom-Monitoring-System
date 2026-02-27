from ultralytics import YOLO
import cv2

# Load trained model
# Change folder name if needed (train13/train13/etc.)
model = YOLO("runs/detect/train13/weights/best.pt")
print(model.names)  # Print class names to verify correct loading
# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, conf=0.25)

    # Activity counting dictionary
    activity_count = {}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            activity_count[label] = activity_count.get(label, 0) + 1

    print(activity_count)

    # Show annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("Classroom Monitoring", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()