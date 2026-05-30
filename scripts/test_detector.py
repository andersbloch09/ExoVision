"""
Live stair detection test with RealSense camera.
Shows real-time detection with bounding boxes and distance measurements.
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.dirname(__file__))

from stair_detector import StairDetector

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed")
    print("Install with: pip install pyrealsense2")
    sys.exit(1)


class LiveTestDetector:
    """Live test stair detection using RealSense camera."""
    
    def __init__(self, config_path):
        """Initialize camera and detector."""
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        stair_config = config.get('stair_detection', {})
        
        # Build model path
        model_path = Path(config_path).parent.parent / stair_config.get(
            'model_path', 
            'runs/detect/train/weights/best.pt'
        )
        
        print(f"Loading model from: {model_path}")
        
        # Initialize detector
        self.detector = StairDetector(
            model_path=str(model_path),
            distance_min_m=stair_config.get('distance_threshold', {}).get('min_m', 1.0),
            distance_max_m=stair_config.get('distance_threshold', {}).get('max_m', 3.0),
            confidence_threshold=stair_config.get('confidence_threshold', 0.5)
        )
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Start pipeline
        print("Starting RealSense pipeline...")
        self.pipeline.start(self.config)
        
        # Get intrinsics for depth projection
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
    def get_pointcloud_from_depth(self, depth_frame, color_frame):
        """Convert depth frame to point cloud."""
        
        depth_data = np.asanyarray(depth_frame.get_data())
        color_data = np.asanyarray(color_frame.get_data())
        
        # Get depth scale (to convert depth units to meters)
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        
        h, w = depth_data.shape
        
        # Create point cloud
        points = []
        
        for y in range(0, h, 2):  # Sample every 2 pixels for speed
            for x in range(0, w, 2):
                depth = depth_data[y, x] * depth_scale
                
                # Skip invalid depths
                if depth == 0 or depth > 10:  # Max 10 meters
                    continue
                
                # Convert pixel to 3D point
                x_3d = (x - self.depth_intrinsics.ppx) * depth / self.depth_intrinsics.fx
                y_3d = (y - self.depth_intrinsics.ppy) * depth / self.depth_intrinsics.fy
                
                points.append([x_3d, y_3d, depth])
        
        if len(points) == 0:
            return np.array([])
        
        pointcloud = np.array(points, dtype=np.float32)
        
        # Normalize to reasonable ranges
        pointcloud[:, 0] = (pointcloud[:, 0] - pointcloud[:, 0].mean()) / (pointcloud[:, 0].std() + 1e-6)
        pointcloud[:, 1] = (pointcloud[:, 1] - pointcloud[:, 1].mean()) / (pointcloud[:, 1].std() + 1e-6)
        pointcloud[:, 2] = pointcloud[:, 2] / 3.0  # Depth in meters
        
        return pointcloud
    
    def run(self):
        """Run live detection test."""
        
        print("Starting live detection test... Press 'q' to quit")
        frame_count = 0
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                
                # Get point cloud
                pointcloud = self.get_pointcloud_from_depth(depth_frame, color_frame)
                
                # Convert BGR to RGB for detector
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Run detection
                detection = self.detector.detect_stairs(rgb_image, pointcloud)
                
                # Draw results
                display_image = color_image.copy()
                
                if detection['bbox'] is not None:
                    x1, y1, x2, y2 = detection['bbox']
                    
                    # Draw bounding box
                    color = (0, 255, 0) if detection['stair_type'] == 'ascending_stairs' else (0, 165, 255)
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{detection['stair_type']} ({detection['confidence']:.2f})"
                    
                    if detection['distance_m'] > 0:
                        label += f" - {detection['distance_m']:.2f}m"
                        # Draw distance indicator
                        cv2.putText(display_image, 
                                   f"ALERT: {detection['distance_m']:.2f}m", 
                                   (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.5, (0, 0, 255), 3)
                    
                    cv2.putText(display_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(display_image, "No stairs detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                # Add frame info
                frame_count += 1
                cv2.putText(display_image, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show image
                cv2.imshow('Live Stair Detection Test', display_image)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            print("Camera stopped")


def main():
    """Main entry point."""
    
    config_path = Path(__file__).parent / "stair_config.json"
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return
    
    detector = LiveTestDetector(str(config_path))
    detector.run()


if __name__ == "__main__":
    main()
