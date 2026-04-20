"""
Count stairs in a point cloud
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy import ndimage

# Configuration
STAIRS_DATASET_PATH = r'C:\Users\ander\Documents\GitHub\ExoVision\data\raw\realsense_d435\session_20260420_121030\pointcloud'  # Path to point cloud dataset (directory containing .npz files)
FRAME_TO_ANALYZE = 24  # Which frame to analyze
HEIGHT_BIN_SIZE = 0.01  # 1cm bins for height analysis (m)
MIN_POINTS_PER_STEP = 500  # Minimum points to count as a step (increased to filter noise)
PEAK_DISTANCE = 25  # Minimum distance between peaks in histogram bins (tuned)
MIN_STEP_HEIGHT = 0.30  # Minimum height difference to count as separate step (30cm for 3 stairs)
VISUALIZE = True

def load_point_cloud(npz_file):
    """Load point cloud from NPZ file"""
    try:
        data = np.load(npz_file)
        return data['points']
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None

def count_stairs_by_step_edges(points, bin_size=0.01, min_jump=0.1):
    """
    Detect stairs by finding step edges (discontinuities in depth as height increases).
    
    Algorithm:
    1. Sort points by Y (height)
    2. Bin by Y and find average Z (depth) in each bin
    3. Look for jumps in Z → these are step risers
    4. Extract stairs from the discontinuities
    
    Returns: Stair information with rise heights
    """
    y_values = points[:, 1]
    z_values = points[:, 2]
    
    y_min, y_max = y_values.min(), y_values.max()
    z_min, z_max = z_values.min(), z_values.max()
    
    # Create height bins
    num_bins = int((y_max - y_min) / bin_size) + 1
    y_bins = np.linspace(y_min, y_max, num_bins)
    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
    
    # For each height bin, calculate median depth
    z_at_height = []
    points_per_bin = []
    
    for i in range(len(y_bins) - 1):
        mask = (y_values >= y_bins[i]) & (y_values < y_bins[i+1])
        if np.sum(mask) > 0:
            z_at_height.append(np.median(z_values[mask]))
            points_per_bin.append(np.sum(mask))
        else:
            z_at_height.append(np.nan)
            points_per_bin.append(0)
    
    z_at_height = np.array(z_at_height)
    
    # Find significant jumps in Z (step risers)
    # Calculate the change in depth as we go up
    z_diffs = np.diff(z_at_height)
    
    # Find where depth increases significantly (going down stairs = Z increases)
    # Smooth the differences to avoid noise
    from scipy.ndimage import gaussian_filter1d
    z_diffs_smooth = gaussian_filter1d(z_diffs, sigma=0.5)
    
    # Find peaks in depth difference (step edges)
    # Use lower height threshold to catch more steps
    from scipy.signal import find_peaks
    edge_indices, _ = find_peaks(z_diffs_smooth, height=0.005, distance=2)
    
    # Extract stair information from edges
    stair_info = []
    
    if len(edge_indices) > 0:
        # Start from bottom (lowest Y)
        prev_y_idx = 0
        
        for edge_idx in edge_indices:
            # This edge marks the transition to a new stair
            stair_top_y_idx = edge_idx
            stair_bottom_y_idx = prev_y_idx
            
            # Get Y range for this stair
            stair_y_min = y_bin_centers[stair_bottom_y_idx]
            stair_y_max = y_bin_centers[stair_top_y_idx]
            stair_y_height = stair_y_max - stair_y_min
            
            # Get corresponding Z values
            stair_z_bottom = z_at_height[stair_bottom_y_idx]
            stair_z_top = z_at_height[stair_top_y_idx]
            
            # Points in this stair
            mask = (y_values >= stair_y_min) & (y_values <= stair_y_max)
            num_points = np.sum(mask)
            
            if num_points > 50 and stair_y_height > 0.05:  # Filter out tiny stairs (less than 5cm)
                stair_info.append({
                    'y_height': stair_y_height,
                    'y_min': stair_y_min,
                    'y_max': stair_y_max,
                    'z_at_bottom': stair_z_bottom,
                    'z_at_top': stair_z_top,
                    'z_depth_change': stair_z_top - stair_z_bottom,
                    'point_count': num_points
                })
            
            prev_y_idx = stair_top_y_idx
    
    # Calculate step-to-step heights (Y differences)
    step_heights = []
    if len(stair_info) > 1:
        for i in range(len(stair_info) - 1):
            y_rise = stair_info[i+1]['y_min'] - stair_info[i]['y_max']
            step_heights.append(y_rise)
    
    return {
        'num_stairs': len(stair_info),
        'stair_info': stair_info,
        'step_heights': step_heights,
        'y_bins': y_bin_centers,
        'z_at_height': z_at_height,
        'z_diffs': z_diffs_smooth,
        'edge_indices': edge_indices,
        'y_min': y_min,
        'y_max': y_max,
        'z_min': z_min,
        'z_max': z_max
    }

def segment_stairs_by_height(points, num_stairs_estimate=None):
    """
    Segment point cloud into individual stair steps based on height
    """
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    
    # Use histogram to find natural breakpoints
    num_bins = max(50, int(len(points) / 1000))
    histogram, bin_edges = np.histogram(y_values, bins=num_bins)
    
    # Find valleys in histogram as separation points
    valleys = np.where(histogram < np.percentile(histogram, 10))[0]
    valley_heights = bin_edges[valleys]
    
    # Segment points by height
    segments = []
    current_y = y_values.min()
    
    for valley_height in sorted(valley_heights):
        mask = (y_values >= current_y) & (y_values < valley_height)
        if np.sum(mask) > 100:  # Only count if enough points
            segments.append({
                'points': points[mask],
                'y_range': (current_y, valley_height),
                'point_count': np.sum(mask)
            })
        current_y = valley_height
    
    # Add remaining points
    mask = y_values >= current_y
    if np.sum(mask) > 100:
        segments.append({
            'points': points[mask],
            'y_range': (current_y, y_values.max()),
            'point_count': np.sum(mask)
        })
    
    return segments

def main():
    print("="*70)
    print("STAIR COUNTER - POINT CLOUD ANALYSIS")
    print("="*70)
    
    if not os.path.exists(STAIRS_DATASET_PATH):
        print(f"❌ Path not found: {STAIRS_DATASET_PATH}")
        sys.exit(1)
    
    # Find and load frame
    npz_files = sorted([f for f in os.listdir(STAIRS_DATASET_PATH) if f.endswith('.npz')])
    
    if FRAME_TO_ANALYZE >= len(npz_files):
        print(f"❌ Frame {FRAME_TO_ANALYZE} not found. Available: {len(npz_files)} frames")
        sys.exit(1)
    
    frame_file = npz_files[FRAME_TO_ANALYZE]
    filepath = os.path.join(STAIRS_DATASET_PATH, frame_file)
    
    print(f"\n📂 Loading frame: {frame_file}")
    points = load_point_cloud(filepath)
    
    if points is None or len(points) == 0:
        print("❌ Failed to load point cloud")
        sys.exit(1)
    
    print(f"✅ Loaded {len(points):,} points")
    print(f"   X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"   Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"   Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")
    
    # Analyze stairs
    print("\n" + "="*70)
    print("ANALYZING STAIRS...")
    print("="*70)
    
    results = count_stairs_by_step_edges(points, 
                                         bin_size=0.01,
                                         min_jump=0.05)
    
    num_stairs = results['num_stairs']
    stair_info = results['stair_info']
    step_heights = results['step_heights']
    
    print(f"\n✅ DETECTED STAIRS: {num_stairs}")
    
    if num_stairs > 0:
        print(f"\nStair details (step edge detection):")
        
        for i, stair in enumerate(stair_info):
            print(f"\n  Stair {i+1}:")
            print(f"    Height range (Y): {stair['y_min']:.4f} to {stair['y_max']:.4f} m")
            print(f"    Rise height: {stair['y_height']:.4f} m ({stair['y_height']*100:.2f} cm)")
            print(f"    Depth (Z): {stair['z_at_bottom']:.4f} to {stair['z_at_top']:.4f} m")
            print(f"    Points: {stair['point_count']}")
    
    if len(step_heights) > 0:
        print(f"\nHeight difference between consecutive stairs (Y-axis):")
        for i, diff in enumerate(step_heights):
            print(f"  Stair {i+1} → {i+2}: {diff:.4f} m ({diff*100:.2f} cm)")
        if len(step_heights) > 0:
            print(f"\nAverage rise: {np.mean(step_heights):.4f} m ({np.mean(step_heights)*100:.2f} cm)")
            print(f"Min rise: {np.min(step_heights):.4f} m ({np.min(step_heights)*100:.2f} cm)")
            print(f"Max rise: {np.max(step_heights):.4f} m ({np.max(step_heights)*100:.2f} cm)")
    
    # Segment by height
    print("\n" + "="*70)
    print("SEGMENTING STAIRS BY HEIGHT...")
    print("="*70)
    
    segments = segment_stairs_by_height(points)
    print(f"\n✅ Found {len(segments)} height-based segments:")
    
    for i, seg in enumerate(segments):
        y_min, y_max = seg['y_range']
        print(f"  Segment {i+1}: Y [{y_min:.4f}, {y_max:.4f}] m - {seg['point_count']} points")
    
    # Visualization
    if VISUALIZE:
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS...")
        print("="*70)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: 3D Point cloud colored by height
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        y_values = points[:, 1]
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=y_values, cmap='viridis', s=2, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m) - Height')
        ax1.set_zlabel('Z (m) - Depth')
        ax1.set_title(f'Point Cloud with {num_stairs} Detected Stairs')
        plt.colorbar(scatter, ax=ax1, label='Height Y (m)', shrink=0.5)
        
        # Plot 2: Z vs Height (showing step edges)
        ax2 = fig.add_subplot(2, 2, 2)
        y_bins = results['y_bins']
        z_at_height = results['z_at_height']
        ax2.plot(y_bins, z_at_height, 'b-', linewidth=2, label='Depth at height')
        ax2.scatter(y_bins, z_at_height, c='blue', s=20, alpha=0.5)
        
        # Mark detected step edges
        if len(stair_info) > 0:
            for i, stair in enumerate(stair_info):
                ax2.axvline(stair['y_min'], color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax2.axvline(stair['y_max'], color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_xlabel('Height Y (m)')
        ax2.set_ylabel('Depth Z (m)')
        ax2.set_title('Step Edge Detection (Depth Profile)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Z difference (step edges)
        ax3 = fig.add_subplot(2, 2, 3)
        z_diffs = results['z_diffs']
        ax3.plot(y_bins[:-1], z_diffs, 'g-', linewidth=2, label='Depth change')
        ax3.fill_between(y_bins[:-1], 0, z_diffs, alpha=0.3, color='green')
        
        edge_indices = results['edge_indices']
        if len(edge_indices) > 0:
            ax3.scatter(y_bins[edge_indices], z_diffs[edge_indices], 
                       c='red', s=100, marker='X', label='Detected edges', zorder=5)
        
        ax3.set_xlabel('Height Y (m)')
        ax3.set_ylabel('Depth Change dZ/dY')
        ax3.set_title('Step Edge Discontinuities')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Stair geometry
        ax4 = fig.add_subplot(2, 2, 4)
        z_values = points[:, 2]
        ax4.scatter(z_values, y_values, c=y_values, cmap='viridis', s=2, alpha=0.6)
        
        # Mark detected stairs
        if len(stair_info) > 0:
            for i, stair in enumerate(stair_info):
                z_avg = (stair['z_at_bottom'] + stair['z_at_top']) / 2
                ax4.scatter(z_avg, stair['y_max'], c='red', s=200, marker='*', 
                           edgecolors='black', linewidths=2, zorder=5)
                ax4.text(z_avg, stair['y_max'] + 0.05, f'S{i+1}', 
                        ha='center', fontweight='bold')
        
        ax4.set_xlabel('Depth Z (m) - Camera Distance')
        ax4.set_ylabel('Height Y (m)')
        ax4.set_title('Stair Geometry (Height vs Depth)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save instead of show
        output_path = os.path.join(os.path.dirname(__file__), 'stair_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        plt.close()

if __name__ == "__main__":
    main()
