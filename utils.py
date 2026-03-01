import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ==========================
# CONFIGURATION
# ==========================

SOURCE_DIR = "COVID-19_Radiography_Dataset"  # original dataset folder
OUTPUT_DIR = "formatted_dataset"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

random.seed(42)

# ==========================
# CREATE OUTPUT STRUCTURE
# ==========================

splits = ["train", "val", "test"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ==========================
# PROCESS EACH CLASS
# ==========================

classes = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

for class_name in classes:

    class_path = os.path.join(SOURCE_DIR, class_name, "images")

    if not os.path.exists(class_path):
        print(f"Skipping {class_name}, folder not found.")
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.endswith((".png", ".jpg", ".jpeg"))]

    # Train split
    train_images, temp_images = train_test_split(
        images, test_size=(1 - TRAIN_SPLIT), random_state=42
    )

    # Validation & test split
    val_images, test_images = train_test_split(
        temp_images, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=42
    )

    # Create class folders inside splits
    for split in splits:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

    # Move files
    def move_files(file_list, split_name):
        for file in file_list:
            src = os.path.join(class_path, file)
            dst = os.path.join(OUTPUT_DIR, split_name, class_name, file)
            shutil.copy2(src, dst)

    move_files(train_images, "train")
    move_files(val_images, "val")
    move_files(test_images, "test")

    print(f"{class_name} split completed.")

print("\nDataset successfully formatted!")