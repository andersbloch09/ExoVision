"""Debug script to test if receiver can load model."""
import os
import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

print("Testing receiver setup...")
print(f"Current directory: {os.getcwd()}")

# Check if config exists
config_path = Path("scripts/stair_config.json")
print(f"Config path: {config_path}")
print(f"Config exists: {config_path.exists()}")

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    stair_config = config.get('stair_detection', {})
    model_path = Path(stair_config.get('model_path', 'runs/detect/train/weights/best.pt'))
    
    print(f"Model path from config: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if model_path.exists():
        print("✓ Model file found!")
        
        # Try importing StairDetector
        try:
            from stair_detector import StairDetector
            print("✓ StairDetector imported successfully")
            
            # Try creating detector
            detector = StairDetector(
                model_path=str(model_path),
                distance_min_m=1.0,
                distance_max_m=3.0,
                confidence_threshold=0.5
            )
            print("✓ StairDetector initialized successfully!")
            print("✓ RECEIVER SHOULD WORK!")
        except Exception as e:
            print(f"✗ Error initializing StairDetector: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Model not found at {model_path}")
        print("  Available runs:")
        for item in Path("runs/detect").glob("*/"):
            print(f"    - {item.name}")
else:
    print(f"✗ Config not found at {config_path}")
