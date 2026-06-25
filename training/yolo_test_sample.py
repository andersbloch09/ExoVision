from ultralytics import YOLO

def test_yolo_model():
    """Test the trained YOLO model on the test set."""
    
    # Load the best trained model
    model = YOLO(r'C:\Users\ander\Documents\GitHub\ExoVision\runs\detect\train\weights\best.pt')
    
    # Create a temporary YAML pointing to test set
    test_yaml = """
path: C:\\Users\\ander\\Documents\\GitHub\\ExoVision\\data\\processed
train: test/images
val: test/images
test: test/images

nc: 3
names:
  0: flat_ground
  1: ascending_stairs
  2: descending_stairs
"""
    
    with open('data/datasets/test_only.yaml', 'w') as f:
        f.write(test_yaml)
    
    # Run validation on test set
    results = model.val(data='data/datasets/test_only.yaml', save=True)
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"✅ Test mAP@0.5: {results.box.map50:.4f}")
    print(f"📊 Test mAP@[.5:.95]: {results.box.map:.4f}")
    print("="*50)

if __name__ == "__main__":
    print("🧪 Testing on test set...")
    test_yolo_model()
