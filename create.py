import os

dataset_path = "dataset/images"

class_map = {
    "sleep": 0,
    "mobile": 1,
    "hand": 2,
    "attentive": 3
}

for split in ["train", "val"]:
    image_dir = os.path.join(dataset_path, split)
    label_dir = os.path.join("dataset/labels", split)

    os.makedirs(label_dir, exist_ok=True)

    for img in os.listdir(image_dir):
        if img.endswith(".jpg") or img.endswith(".png"):
            img_lower = img.lower()

            class_id = None
            for key in class_map:
                if key in img_lower:
                    class_id = class_map[key]
                    break

            if class_id is not None:
                label_file = os.path.join(label_dir, img.replace(".jpg", ".txt").replace(".png", ".txt"))
                with open(label_file, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1 1\n")

print("Labels created!")