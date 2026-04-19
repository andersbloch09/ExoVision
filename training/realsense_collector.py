# training/realsense_collector.py
"""
RealSense collector that saves color images + point clouds (3D).
Point clouds are ready for geometric analysis (stairs counting, ramp tilt, etc).
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
from pathlib import Path

class PointCloudConverter:
    """Convert depth maps to 3D point clouds."""
    # RealSense D435 intrinsics
    D435_INTRINSICS = {
        'fx': 614.0,
        'fy': 614.0,
        'cx': 320.0,
        'cy': 240.0,
    }
    
    @staticmethod
    def depth_to_xyz(depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map (mm) to point cloud (meters)."""
        height, width = depth_map.shape
        u = np.arange(width)
        v = np.arange(height)
        u_grid, v_grid = np.meshgrid(u, v)
        
        fx = PointCloudConverter.D435_INTRINSICS['fx']
        fy = PointCloudConverter.D435_INTRINSICS['fy']
        cx = PointCloudConverter.D435_INTRINSICS['cx']
        cy = PointCloudConverter.D435_INTRINSICS['cy']
        
        z = depth_map * 0.001  # mm to meters
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        valid = z.flatten() > 0
        return points[valid]

class InteractiveRealsenseCollector:
    def __init__(self, dataset_dir="data/raw/realsense_d435", dataset_name="unified_dataset"):
        """
        Initialize interactive RealSense collector for manual sampling.
        
        Args:
            dataset_dir: Base directory for datasets
            dataset_name: Name of dataset to append to (creates if doesn't exist)
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(dataset_dir, dataset_name)
        
        # Create dataset structure
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Class directories
        self.class_dirs = {
            'descending_ramp': os.path.join(self.dataset_path, 'descending_ramp'),
            'upgoing_ramp': os.path.join(self.dataset_path, 'upgoing_ramp'),
            'stairs': os.path.join(self.dataset_path, 'stairs'),
            'flat_floor': os.path.join(self.dataset_path, 'flat_floor'),
        }
        
        # Create class subdirectories
        for class_name, class_dir in self.class_dirs.items():
            os.makedirs(os.path.join(class_dir, 'color'), exist_ok=True)
            os.makedirs(os.path.join(class_dir, 'pointcloud'), exist_ok=True)
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Track samples
        self.dataset_metadata = self._load_dataset_metadata()
    
    def _load_dataset_metadata(self):
        """Load or create dataset metadata."""
        metadata_path = os.path.join(self.dataset_path, 'dataset_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # File is not empty
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass  # File is empty or corrupted, create new
        
        return {
            'dataset_name': self.dataset_name,
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'classes': {
                'descending_ramp': [],
                'upgoing_ramp': [],
                'stairs': [],
                'flat_floor': []
            }
        }
    
    def _save_dataset_metadata(self):
        """Save dataset metadata."""
        metadata_path = os.path.join(self.dataset_path, 'dataset_metadata.json')
        self.dataset_metadata['last_updated'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(self.dataset_metadata, f, indent=2)
    
    def _get_next_sample_id(self, class_name):
        """Get next sample ID for a class."""
        return len(self.dataset_metadata['classes'][class_name])
    
    def start(self):
        """Start RealSense streaming."""
        try:
            self.pipeline.start(self.config)
            print("✅ RealSense pipeline started")
        except RuntimeError as e:
            print(f"❌ Failed to start RealSense: {e}")
            raise
    
    def capture_samples(self, class_name, num_frames=5):
        """
        Capture N frames for a specific class on-demand.
        Saves color images + 3D point clouds.
        
        Args:
            class_name: One of 'descending_ramp', 'upgoing_ramp', 'stairs', 'flat_floor'
            num_frames: Number of frames to capture (default 5)        
        Returns:
            Dictionary with capture info
        """
        if class_name not in self.class_dirs:
            print(f"❌ Unknown class: {class_name}")
            return None
        
        class_dir = self.class_dirs[class_name]
        color_dir = os.path.join(class_dir, 'color')
        pointcloud_dir = os.path.join(class_dir, 'pointcloud')
        
        sample_id = self._get_next_sample_id(class_name)
        captured_files = []
        capture_time = datetime.now()
        
        print(f"\n📹 Capturing {num_frames} frames for [{class_name}]")
        print("   Let camera stabilize for 1 second...")
        time.sleep(1)
        
        # Discard first few frames to let camera stabilize
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(timeout_ms=100)
            except:
                pass
        
        for frame_num in range(num_frames):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    print(f"   ⚠️  Frame {frame_num + 1}: No frames available")
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{sample_id:03d}_{frame_num:02d}"
                
                # Save color
                color_path = os.path.join(color_dir, filename + ".jpg")
                cv2.imwrite(color_path, color_image)
                
                # Convert depth to point cloud and save
                try:
                    points = PointCloudConverter.depth_to_xyz(depth_image)
                    pointcloud_path = os.path.join(pointcloud_dir, filename + ".npz")
                    np.savez_compressed(pointcloud_path, points=points)
                    
                    captured_files.append({
                        'frame_id': frame_num,
                        'color_file': os.path.join('color', filename + '.jpg'),
                        'pointcloud_file': os.path.join('pointcloud', filename + '.npz'),
                        'num_points': len(points),
                        'z_range_m': [float(points[:, 2].min()), float(points[:, 2].max())]
                    })
                    
                    print(f"   ✅ Frame {frame_num + 1}/{num_frames} ({len(points)} points)")
                except Exception as e:
                    print(f"   ⚠️  Point cloud conversion failed: {e}")
                
                time.sleep(1)  # Small delay between frames
            
            except Exception as e:
                print(f"   ❌ Frame {frame_num + 1}: {e}")
        
        # Save metadata for this sample set
        if captured_files:
            sample_metadata = {
                'sample_id': sample_id,
                'class': class_name,
                'timestamp': capture_time.isoformat(),
                'num_frames': len(captured_files),
                'camera': 'RealSense D435',
                'frames': captured_files
            }
            
            self.dataset_metadata['classes'][class_name].append(sample_metadata)
            self._save_dataset_metadata()
            
            print(f"✅ Captured {len(captured_files)} frames for {class_name}")
            return sample_metadata
        else:
            print(f"❌ Failed to capture frames")
            return None
    
    def show_stats(self):
        """Show current dataset statistics."""
        print(f"\n📊 Dataset Statistics: {self.dataset_name}")
        print("=" * 50)
        total_samples = 0
        total_frames = 0
        
        for class_name, samples in self.dataset_metadata['classes'].items():
            num_samples = len(samples)
            num_frames = sum(s['num_frames'] for s in samples)
            total_samples += num_samples
            total_frames += num_frames
            print(f"  {class_name:20s}: {num_samples:3d} samples, {num_frames:3d} frames")
        
        print("=" * 50)
        print(f"  {'TOTAL':20s}: {total_samples:3d} samples, {total_frames:3d} frames")
        print(f"  📁 Location: {self.dataset_path}\n")
    
    def interactive_collection(self):
        """Interactive loop for manual sampling."""
        print("\n" + "="*60)
        print("INTERACTIVE REALSENSE COLLECTOR")
        print("="*60)
        print("\nAvailable classes:")
        for i, class_name in enumerate(self.class_dirs.keys(), 1):
            print(f"  {i}: {class_name}")
        print("\nCommands:")
        print("  [1-4]: Select class and capture 5 frames")
        print("  [s]:   Show dataset statistics")
        print("  [q]:   Quit\n")
        
        while True:
            try:
                cmd = input("Enter command: ").strip().lower()
                
                if cmd == 'q':
                    print("👋 Goodbye!")
                    break
                
                elif cmd == 's':
                    self.show_stats()
                
                elif cmd in ['1', '2', '3', '4']:
                    class_names = list(self.class_dirs.keys())
                    class_idx = int(cmd) - 1
                    class_name = class_names[class_idx]
                    
                    # Ask for frame count
                    try:
                        num_frames = int(input(f"   How many frames? (default 5): ") or "5")
                        num_frames = max(1, min(num_frames, 20))  # Limit 1-20
                    except:
                        num_frames = 5
                    
               
                    self.capture_samples(class_name, num_frames)
                
                else:
                    print("❌ Invalid command")
            
            except KeyboardInterrupt:
                print("\n\n👋 Collection interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def stop(self):
        """Stop RealSense streaming."""
        self.pipeline.stop()
        print("🛑 RealSense pipeline stopped")

if __name__ == "__main__":
    collector = InteractiveRealsenseCollector(dataset_name="unified_dataset")
    collector.start()
    
    try:
        collector.interactive_collection()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        collector.stop()
        print("\n✅ Final dataset stats:")
        collector.show_stats()