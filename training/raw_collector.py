"""
Simple raw data collector for RealSense D435.
Captures color images + point clouds continuously to a timestamped folder.
No pre-labeling — all data ready for YOLO annotation afterward.
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


class RawRealsenseCollector:
    def __init__(self, base_dir="data/raw/realsense_d435"):
        """
        Initialize raw RealSense collector.
        
        Args:
            base_dir: Base directory for raw collections
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Create session directory with timestamp
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{self.session_name}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.color_dir = os.path.join(self.session_dir, "color")
        self.pointcloud_dir = os.path.join(self.session_dir, "pointcloud")
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.frame_count = 0
        self.is_recording = False
        
    def start(self):
        """Start RealSense pipeline."""
        try:
            self.pipeline.start(self.config)
            print("✅ RealSense pipeline started")
        except RuntimeError as e:
            print(f"❌ Failed to start RealSense: {e}")
            raise
    
    def stop(self):
        """Stop RealSense pipeline."""
        self.pipeline.stop()
        print("✅ RealSense pipeline stopped")
    
    def record(self, duration_seconds=None):
        """
        Record continuously, saving a frame every second (or custom interval).
        
        Args:
            duration_seconds: Record for N seconds (None = until user presses 'q')
        """
        print(f"\n📹 Recording to: {self.session_dir}")
        print("   Press 'q' to stop recording\n")
        
        self.is_recording = True
        start_time = time.time()
        last_save_time = start_time
        save_interval = 0.5  # Save a frame every second
        
        try:
            while True:
                # Check if duration limit reached
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    print(f"⏱️  Duration limit ({duration_seconds}s) reached")
                    break
                
                # Get frames
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    current_time = time.time()
                    
                    # Save frame every N seconds
                    if (current_time - last_save_time) >= save_interval:
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data())
                        
                        # Generate filename with frame number
                        filename = f"frame_{self.frame_count:06d}"
                        
                        # Save color image
                        color_path = os.path.join(self.color_dir, filename + ".jpg")
                        cv2.imwrite(color_path, color_image)
                        
                        # Convert and save point cloud
                        try:
                            points = PointCloudConverter.depth_to_xyz(depth_image)
                            pointcloud_path = os.path.join(self.pointcloud_dir, filename + ".npz")
                            np.savez_compressed(pointcloud_path, points=points)
                            
                            print(f"  [{self.frame_count:06d}] Saved - {len(points):6d} points")
                            self.frame_count += 1
                            last_save_time = current_time
                            
                        except Exception as e:
                            print(f"  ⚠️  Point cloud save error: {e}")
                    
                    # Show live preview
                    cv2.putText(color_image, f"Frames: {self.frame_count} | Press 'q' to stop", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Recording...", color_image)
                    
                    # Check for 'q' key to stop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("⏹️  Recording stopped by user")
                        break
                        
                except Exception as e:
                    print(f"⚠️  Frame capture error: {e}")
                    continue
        
        finally:
            cv2.destroyAllWindows()
            self.is_recording = False
        
        # Save metadata
        self._save_metadata()
        print(f"\n✅ Captured {self.frame_count} frames")
        print(f"📂 Data saved to: {self.session_dir}")
    
    def _save_metadata(self):
        """Save collection metadata for reference."""
        metadata = {
            'session_name': self.session_name,
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'color_dir': 'color',
            'pointcloud_dir': 'pointcloud',
            'camera_model': 'RealSense D435',
            'resolution': '640x480',
            'fps': 30,
            'save_interval_seconds': 1.0
        }
        
        metadata_path = os.path.join(self.session_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Interactive menu for raw collection."""
    collector = RawRealsenseCollector()
    
    try:
        collector.start()
        
        print("\n" + "="*50)
        print("🎥 RealSense Raw Data Collector")
        print("="*50)
        print("Ready to capture data for YOLO labeling")
        print()
        
        # Interactive menu
        while True:
            print("\nOptions:")
            print("  1 - Record for 30 seconds")
            print("  2 - Record for 60 seconds")
            print("  3 - Record until 'q' pressed (unlimited)")
            print("  4 - Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                collector.record(duration_seconds=30)
            elif choice == '2':
                collector.record(duration_seconds=60)
            elif choice == '3':
                collector.record()
            elif choice == '4':
                print("👋 Exiting...")
                break
            else:
                print("❌ Invalid option")
    
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
