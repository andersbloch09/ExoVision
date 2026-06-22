"""
Script to split labeled YOLO dataset into train/val/test folders.
Random split: 70% train, 15% val, 15% test
"""

import shutil
import random
from pathlib import Path

# Define paths
base_dir = Path(__file__).parent.parent / "data" / "processed"
train_img_dir = base_dir / "train" / "images"
train_label_dir = base_dir / "train" / "labels"

# Output directories
val_img_dir = base_dir / "val" / "images"
val_label_dir = base_dir / "val" / "labels"
test_img_dir = base_dir / "test" / "images"
test_label_dir = base_dir / "test" / "labels"

# Create output directories if they don't exist
for d in [val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Get all label files (excluding classes.txt)
label_files = sorted([f for f in train_label_dir.glob("*.txt") if f.name != "classes.txt"])

# Create image-label pairs
file_pairs = []
for label_file in label_files:
    stem = label_file.stem
    img_file = train_img_dir / f"{stem}.jpg"
    if img_file.exists():
        file_pairs.append((img_file, label_file))

print(f"Found {len(file_pairs)} image-label pairs")

# Shuffle and split randomly (70% train, 15% val, 15% test)
random.seed(42)  # For reproducibility
random.shuffle(file_pairs)

n_files = len(file_pairs)
train_idx = int(n_files * 0.70)
val_idx = train_idx + int(n_files * 0.15)

train_pairs = file_pairs[:train_idx]
val_pairs = file_pairs[train_idx:val_idx]
test_pairs = file_pairs[val_idx:]

print(f"\nSplit distribution:")
print(f"  Train: {len(train_pairs)} images")
print(f"  Val: {len(val_pairs)} images")
print(f"  Test: {len(test_pairs)} images")

# Move files to appropriate directories
def move_pairs(pairs, img_dest, label_dest, split_name):
    count = 0
    for img_file, label_file in pairs:
        dest_img = img_dest / img_file.name
        dest_label = label_dest / label_file.name
        
        if not dest_img.exists():
            shutil.move(str(img_file), str(dest_img))
            count += 1
        if not dest_label.exists():
            shutil.move(str(label_file), str(dest_label))
    
    return count

# Move val and test files
val_moved = move_pairs(val_pairs, val_img_dir, val_label_dir, "val")
test_moved = move_pairs(test_pairs, test_img_dir, test_label_dir, "test")

# Copy classes.txt to val and test
classes_file = train_label_dir / "classes.txt"
if classes_file.exists():
    shutil.copy(classes_file, val_label_dir / "classes.txt")
    shutil.copy(classes_file, test_label_dir / "classes.txt")

# Count final distribution
train_img_count = len(list(train_img_dir.glob("*.jpg")))
val_img_count = len(list(val_img_dir.glob("*.jpg")))
test_img_count = len(list(test_img_dir.glob("*.jpg")))

print(f"\n✓ Split complete!")
print(f"\nFinal dataset distribution:")
print(f"  Train: {train_img_count} images")
print(f"  Val: {val_img_count} images")
print(f"  Test: {test_img_count} images")
print(f"  Total: {train_img_count + val_img_count + test_img_count} images")
