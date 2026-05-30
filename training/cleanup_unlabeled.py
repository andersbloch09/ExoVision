"""
Script to remove images that don't have corresponding label files.
Cleans up train, val, and test splits in the processed dataset.
"""

from pathlib import Path
import os

def cleanup_split(split_name):
    """
    Remove unlabeled images from a dataset split.
    
    Args:
        split_name: 'train', 'val', or 'test'
    """
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    img_dir = base_dir / split_name / "images"
    label_dir = base_dir / split_name / "labels"
    
    if not img_dir.exists():
        print(f"⚠️  {split_name}/images directory not found")
        return 0
    
    if not label_dir.exists():
        print(f"⚠️  {split_name}/labels directory not found")
        return 0
    
    deleted_count = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Get all images
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    for img_file in images:
        # Create corresponding label file path
        label_file = label_dir / f"{img_file.stem}.txt"
        
        # If label doesn't exist, delete the image
        if not label_file.exists():
            try:
                img_file.unlink()
                print(f"  ❌ Deleted: {img_file.name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ⚠️  Error deleting {img_file.name}: {e}")
    
    return deleted_count

if __name__ == "__main__":
    print("🧹 Cleaning up unlabeled images from dataset splits...\n")
    
    total_deleted = 0
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split.upper()}:")
        deleted = cleanup_split(split)
        total_deleted += deleted
        print(f"  Deleted: {deleted} images\n")
    
    print("="*50)
    print(f"✅ Total images deleted: {total_deleted}")
    print("="*50)
