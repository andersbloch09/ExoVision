"""
Test visualization methods available on this system
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("VISUALIZATION CAPABILITY TEST")
print("="*70)

# Load a sample point cloud
dataset_path = 'data/raw/realsense_d435/unified_dataset/descending_ramp/pointcloud/descending_ramp_005_03.npz'
if not os.path.exists(dataset_path):
    print(f"❌ Test data not found: {dataset_path}")
    sys.exit(1)

data = np.load(dataset_path)
points = data['points']
print(f"\n✅ Loaded {len(points):,} points from test data")
print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m\n")

# Test 1: OpenCV imshow (if GUI available)
print("\n" + "="*70)
print("TEST 1: OpenCV imshow() - 2D Depth Visualization")
print("="*70)
try:
    # Create a 2D depth image from point cloud
    z_values = points[:, 2]
    z_normalized = ((z_values - z_values.min()) / (z_values.max() - z_values.min()) * 255).astype(np.uint8)
    depth_img = np.zeros((480, 640), dtype=np.uint8)
    
    # Map points to 2D image (simple projection)
    x_norm = ((points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 639).astype(int)
    y_norm = ((points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 479).astype(int)
    
    for i in range(len(points)):
        if 0 <= x_norm[i] < 640 and 0 <= y_norm[i] < 480:
            depth_img[y_norm[i], x_norm[i]] = z_normalized[i]
    
    print("✅ Created depth image from point cloud")
    print("   Press any key to close the window...")
    print("\n   Starting OpenCV window (should appear on screen)...")
    
    cv2.imshow('Point Cloud Depth Visualization (OpenCV)', depth_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"✅ OpenCV visualization worked! (pressed key: {key})\n")
except Exception as e:
    print(f"❌ OpenCV failed: {e}\n")

# Test 3: Matplotlib interactive 3D plot
print("="*70)
print("TEST 3: Matplotlib 3D - Interactive GUI Window")
print("="*70)
try:
    matplotlib.use('QtAgg')  # Interactive backend
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for plotting
    sample_idx = np.random.choice(len(points), min(10000, len(points)), replace=False)
    x, y, z = points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2]
    
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Point Cloud - Matplotlib 3D (Interactive)\nClose window to continue...')
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    
    print("✅ Interactive matplotlib window available!")
    print("   You can rotate, zoom, and pan with your mouse")
    print("   Displaying window now (close it to continue)...")
    
    plt.show()
    print(f"✅ Interactive matplotlib worked!\n")
except Exception as e:
    print(f"⚠️  Interactive matplotlib not available in this context: {e}")
    print("   (This is expected in terminal/script mode)\n")
