from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
img = cv2.imread("hod/real.jpeg")

results = model(img, conf=0.01, imgsz=640)

box_id = 1

for b in results[0].boxes:
    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
    cls = int(b.cls[0])
    conf = float(b.conf[0])
    label = model.names[cls]

    # Text to display
    text = f"#{box_id} {label} {conf:.2f}"

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label above box
    cv2.putText(img, text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print(f"Box {box_id}: {label} | confidence={conf:.3f}")

    box_id += 1

cv2.imwrite("activity_debug.jpg", img)
print("\nSaved: activity_debug.jpg")