import os
import requests
import time

# 1. Setup path
unknown_path = "face_dataset/Unknown"
if not os.path.exists(unknown_path):
    os.makedirs(unknown_path)
    print(f"Created folder: {unknown_path}")

print("Downloading 50 random faces to improve accuracy...")

# 2. Download 50 random faces
for i in range(1, 51):
    try:
        # Using a public random face API
        url = f"https://thispersondoesnotexist.com/" 
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(f"{unknown_path}/stranger_{i}.jpg", 'wb') as f:
                f.write(response.content)
            print(f"Downloaded face {i}/50")
        
        # Respectful delay between requests
        time.sleep(1) 
    except Exception as e:
        print(f"Failed to download image {i}: {e}")

print(f"\n✅ Done! Now run your 'train_faces.py' to include these in the model.")