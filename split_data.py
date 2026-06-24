import os
import shutil
import random
from pathlib import Path

random.seed(42)

src = Path("data")
images_src = src / "images"
labels_src = src / "labels"

for split in ["train", "val"]:
    (images_src / split).mkdir(parents=True, exist_ok=True)
    (labels_src / split).mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
all_images = [f for f in images_src.iterdir() if f.suffix.lower() in IMG_EXTS]

random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_pair(img_path, split):
    shutil.move(str(img_path), str(images_src / split / img_path.name))
    label_path = labels_src / (img_path.stem + ".txt")
    if label_path.exists():
        shutil.move(str(label_path), str(labels_src / split / label_path.name))
    else:
        print(f"[WARN] No label found for {img_path.name}")

for img in train_images:
    move_pair(img, "train")
for img in val_images:
    move_pair(img, "val")

print(f"Done. Train: {len(train_images)} images, Val: {len(val_images)} images")