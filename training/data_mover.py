# Move Session `color/` Images Into YOLO Train Folder

from pathlib import Path
import shutil

# Root directory containing session folders
SOURCE_ROOT = Path(r"C:\Users\ander\Documents\GitHub\ExoVision\data\raw\realsense_d435")

# Destination YOLO train images folder
DESTINATION = Path(r"C:\Users\ander\Documents\GitHub\ExoVision\data\processed\train\images")

# Create destination if it does not exist
DESTINATION.mkdir(parents=True, exist_ok=True)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

moved_count = 0

# Iterate through session folders
for session_dir in SOURCE_ROOT.iterdir():
    if not session_dir.is_dir():
        continue

    color_dir = session_dir / "color"

    if not color_dir.exists():
        print(f"Skipping {session_dir.name} (no color folder)")
        continue

    # Move all images from color folder
    for image_path in color_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Create unique filename using session prefix
        new_name = f"{session_dir.name}_{image_path.name}"
        destination_path = DESTINATION / new_name

        # Move file
        shutil.copy2(image_path, destination_path)

        moved_count += 1
        print(f"Copied: {image_path.name} -> {new_name}")

print(f"\nDone. Copied {moved_count} images.")
