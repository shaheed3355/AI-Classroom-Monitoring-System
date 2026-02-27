import cv2
import os

name = input("Enter student name: ")

folder = f"face_dataset/{name}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print("Press SPACE to capture image")
print("Press ESC to finish")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        img_path = f"{folder}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()