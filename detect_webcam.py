import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")

print("Opening webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()