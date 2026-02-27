import cv2
import os

classes = ["sleeping", "using_mobile", "hand_raise", "attentive"]

base = "dataset"
img_train = os.path.join(base, "images/train")
lbl_train = os.path.join(base, "labels/train")

os.makedirs(img_train, exist_ok=True)
os.makedirs(lbl_train, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
class_id = 0

print("Press S to capture image, N to switch class, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Class: {classes[class_id]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dataset Builder", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        img_name = f"{classes[class_id]}_{count}.jpg"
        img_path = os.path.join(img_train, img_name)
        cv2.imwrite(img_path, frame)

        label_path = os.path.join(lbl_train, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1 1\n")

        print("Saved:", img_name)
        count += 1

    if key == ord('n'):
        class_id = (class_id + 1) % 4

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()