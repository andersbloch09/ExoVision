"""
Visualize multiple planes fitted to different point clouds
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Configuration
POINT_CLOUD_DIR = r'C:\Users\ander\Documents\GitHub\ExoVision\data\raw\realsense_d435\session_20260420_121505\pointcloud'
NUM_FRAMES = None  # If None, use SPECIFIC_FRAMES instead
SPECIFIC_FRAMES = [11, 21]  # Frame indices to visualize (0-indexed). Set NUM_FRAMES=None to use this
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
POINT_SAMPLE_SIZE = 3000  # Points per cloud to display (None = all)

def load_point_cloud(npz_file):
    """Load point cloud from NPZ file"""
    try:
        data = np.load(npz_file)
        return data['points']
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None

def fit_plane_pca(points):
    """
    Fit a plane to point cloud using PCA (iterative with outlier removal)
    Returns: normal vector, distance to origin, inlier mask
    """
    try:
        # First pass: fit plane to all points
        centroid = points.mean(axis=0)
        centered_points = points - centroid
        
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        normal = Vt[-1]
        
        # Find inliers (points close to plane)
        residuals = np.abs(np.dot(points - centroid, normal))
        threshold = np.percentile(residuals, 85)  # Use 85th percentile as threshold
        inliers_mask = residuals < threshold
        
        # Second pass: fit plane ONLY to inliers
        if np.sum(inliers_mask) > 10:  # Need enough points
            inlier_points = points[inliers_mask]
            centroid = inlier_points.mean(axis=0)
            centered_points = inlier_points - centroid
            
            U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
            normal = Vt[-1]
        
        # Distance calculation
        d = np.dot(normal, centroid)
        
        # Recalculate inliers with final plane
        residuals = np.abs(np.dot(points - centroid, normal))
        inliers_mask = residuals < threshold
        
        outlier_count = np.sum(~inliers_mask)
        inlier_count = np.sum(inliers_mask)
        
        return normal, d, centroid, inliers_mask, outlier_count, inlier_count
    except Exception as e:
        print(f"Error fitting plane: {e}")
        return None, None, None, None, 0, 0

def plot_plane_on_ax(ax, normal, distance, centroid, xlim, ylim, alpha=0.3, color='red'):
    """
    Plot a plane on 3D axis
    Plane equation: normal · point = distance
    """
    x = np.linspace(xlim[0], xlim[1], 10)
    y = np.linspace(ylim[0], ylim[1], 10)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z from plane equation: nx*x + ny*y + nz*z = d
    # z = (d - nx*x - ny*y) / nz
    if abs(normal[2]) > 0.01:  # Avoid division by zero
        Z = (distance - normal[0]*X - normal[1]*Y) / normal[2]
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color)
        
        # Mark the plane's center point
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], 
                  c=color, s=200, marker='*', edgecolors='black', linewidths=2, zorder=5)

def main():
    print("="*70)
    print("MULTI-PLANE VISUALIZATION FROM POINT CLOUDS")
    print("="*70)
    
    # Find point cloud files
    if not os.path.exists(POINT_CLOUD_DIR):
        print(f"❌ Directory not found: {POINT_CLOUD_DIR}")
        sys.exit(1)
    
    all_npz_files = sorted([f for f in os.listdir(POINT_CLOUD_DIR) if f.endswith('.npz')])
    
    # Select which frames to process
    if NUM_FRAMES is not None:
        npz_files = all_npz_files[:NUM_FRAMES]
    else:
        npz_files = [all_npz_files[i] for i in SPECIFIC_FRAMES if i < len(all_npz_files)]
    
    print(f"✅ Found {len(all_npz_files)} total files, processing {len(npz_files)} frames\n")
    
    # Load point clouds and fit planes
    all_points = []
    planes = []
    
    for idx, filename in enumerate(npz_files):
        filepath = os.path.join(POINT_CLOUD_DIR, filename)
        print(f"Processing {idx+1}/{len(npz_files)}: {filename}")
        
        points = load_point_cloud(filepath)
        if points is None:
            continue
        
        all_points.append(points)
        
        # Fit plane
        normal, dist, centroid, inliers, outlier_count, inlier_count = fit_plane_pca(points)
        if normal is not None:
            planes.append({
                'normal': normal,
                'distance': dist,
                'centroid': centroid,
                'inliers': inliers,
                'filename': filename,
                'points': points,
                'outlier_count': outlier_count,
                'inlier_count': inlier_count
            })
            print(f"  ✅ Plane fitted. Inliers: {inlier_count}, Outliers: {outlier_count} | " +
                  f"Normal: {normal}")
        else:
            print(f"  ❌ Failed to fit plane")
    
    if not planes:
        print("❌ No planes fitted successfully")
        sys.exit(1)
    
    print(f"\n✅ Successfully fitted {len(planes)} planes\n")
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate axis limits
    all_points_combined = np.vstack(all_points)
    xlim = [all_points_combined[:, 0].min(), all_points_combined[:, 0].max()]
    ylim = [all_points_combined[:, 1].min(), all_points_combined[:, 1].max()]
    zlim = [all_points_combined[:, 2].min(), all_points_combined[:, 2].max()]
    
    # Plot each point cloud and its fitted plane
    for idx, plane_data in enumerate(planes):
        color = COLORS[idx % len(COLORS)]
        points = plane_data['points']
        
        # Sample points for display
        if POINT_SAMPLE_SIZE and len(points) > POINT_SAMPLE_SIZE:
            sample_idx = np.random.choice(len(points), POINT_SAMPLE_SIZE, replace=False)
            display_points = points[sample_idx]
        else:
            display_points = points
        
        # Plot points
        ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                  c=color, s=2, alpha=0.4, label=f"{plane_data['filename']}")
        
        # Plot fitted plane
        plot_plane_on_ax(ax, plane_data['normal'], plane_data['distance'],
                        plane_data['centroid'], xlim, ylim, alpha=0.2, color=color)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Multiple Planes from {len(planes)} Point Clouds\n(Colored by frame)')
    ax.legend(loc='upper left', fontsize=8)
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    print("✅ Visualization complete!")
    print("   You can rotate, zoom, and pan with your mouse")
    print("   Close the window to exit...")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
