# training/data_loader.py
"""
Utility for loading and working with collected RealSense data (color + depth).
Makes it easy to load frames for testing stairs detection, visualization, etc.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class RealSenseDataLoader:
    """Load and process collected RealSense color/depth data."""
    
    def __init__(self, session_dir: str):
        """
        Initialize data loader for a collection session.
        
        Args:
            session_dir: Path to session directory (e.g., data/raw/realsense_d435/session_YYYYMMDD_HHMMSS)
        """
        self.session_dir = Path(session_dir)
        if not self.session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        # Load session metadata
        session_metadata_path = self.session_dir / 'session_metadata.json'
        if session_metadata_path.exists():
            with open(session_metadata_path, 'r') as f:
                self.session_metadata = json.load(f)
        else:
            self.session_metadata = {}
        
        self.classes_available = [
            d.name for d in self.session_dir.iterdir() 
            if d.is_dir() and d.name not in ['__pycache__']
        ]
    
    def get_class_metadata(self, class_name: str) -> Dict:
        """Load metadata for a specific class."""
        metadata_path = self.session_dir / class_name / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_frame_count(self, class_name: str) -> int:
        """Get number of frames collected for a class."""
        metadata = self.get_class_metadata(class_name)
        return metadata.get('total_frames', 0)
    
    def load_frame(self, class_name: str, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load color and depth images for a specific frame.
        
        Args:
            class_name: Class name (e.g., 'stairs', 'flat_floor')
            frame_id: Frame index (0-based)
        
        Returns:
            Tuple of (color_image, depth_image) as numpy arrays
        """
        metadata = self.get_class_metadata(class_name)
        if frame_id >= len(metadata.get('frames', [])):
            raise IndexError(f"Frame {frame_id} not found in class {class_name}")
        
        frame_info = metadata['frames'][frame_id]
        
        # Load color
        color_path = self.session_dir / class_name / frame_info['color_file']
        color = cv2.imread(str(color_path))
        
        # Load depth
        depth = None
        if frame_info.get('depth_file'):
            depth_path = self.session_dir / class_name / frame_info['depth_file']
            if depth_path.exists():
                depth = np.load(str(depth_path))
        
        return color, depth
    
    def get_all_frames_for_class(self, class_name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load all frames for a class.
        
        Args:
            class_name: Class name
        
        Returns:
            List of (color, depth) tuples
        """
        frame_count = self.get_frame_count(class_name)
        frames = []
        for i in range(frame_count):
            try:
                color, depth = self.load_frame(class_name, i)
                frames.append((color, depth))
            except Exception as e:
                print(f"Warning: Failed to load frame {i}: {e}")
                continue
        return frames
    
    def visualize_frame(self, class_name: str, frame_id: int, 
                       normalize_depth: bool = True, wait_time: int = 0) -> None:
        """
        Display color and depth images side by side.
        
        Args:
            class_name: Class name
            frame_id: Frame index
            normalize_depth: Normalize depth for visualization
            wait_time: OpenCV waitKey time (0 = wait for key press)
        """
        color, depth = self.load_frame(class_name, frame_id)
        
        # Convert color to RGB for display
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Normalize and colorize depth
        if depth is not None and normalize_depth:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        else:
            depth_colored = np.zeros_like(color)
        
        # Stack side by side
        combined = np.hstack([color_rgb, depth_colored])
        
        # Display
        cv2.imshow(f"{class_name} - Frame {frame_id} (Color | Depth)", combined)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
    def get_depth_statistics(self, class_name: str) -> Dict:
        """Get depth statistics for all frames in a class."""
        metadata = self.get_class_metadata(class_name)
        frames = metadata.get('frames', [])
        
        if not frames:
            return {}
        
        depth_mins = [f['depth_min'] for f in frames]
        depth_maxs = [f['depth_max'] for f in frames]
        depth_means = [f['depth_mean'] for f in frames]
        
        return {
            'min_depth': min(depth_mins),
            'max_depth': max(depth_maxs),
            'mean_depth': np.mean(depth_means),
            'frames_count': len(frames)
        }
    
    def export_frames_to_directory(self, class_name: str, output_dir: str, 
                                   include_depth_visualization: bool = True) -> None:
        """
        Export frames with optional depth visualization.
        
        Args:
            class_name: Class name
            output_dir: Output directory for exported frames
            include_depth_visualization: Save depth as colored images
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_count = self.get_frame_count(class_name)
        
        for i in range(frame_count):
            color, depth = self.load_frame(class_name, i)
            
            # Save color
            color_out = output_path / f"{class_name}_{i:04d}_color.jpg"
            cv2.imwrite(str(color_out), color)
            
            # Save depth visualization if requested
            if depth is not None and include_depth_visualization:
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_out = output_path / f"{class_name}_{i:04d}_depth.jpg"
                cv2.imwrite(str(depth_out), depth_colored)
        
        print(f"✅ Exported {frame_count} frames to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Find latest session
    raw_dir = Path("data/raw/realsense_d435")
    sessions = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    
    if sessions:
        latest_session = sessions[-1]
        print(f"Loading from: {latest_session}")
        
        loader = RealSenseDataLoader(str(latest_session))
        
        print(f"Available classes: {loader.classes_available}")
        
        # Show depth statistics for stairs
        if 'stairs' in loader.classes_available:
            stats = loader.get_depth_statistics('stairs')
            print(f"\nStairs depth statistics:")
            print(f"  Min depth: {stats.get('min_depth')} mm")
            print(f"  Max depth: {stats.get('max_depth')} mm")
            print(f"  Mean depth: {stats.get('mean_depth'):.1f} mm")
            print(f"  Frames: {stats.get('frames_count')}")
            
            # Visualize first frame
            if loader.get_frame_count('stairs') > 0:
                print("\nDisplaying first frame (press 'q' to quit)...")
                loader.visualize_frame('stairs', 0, wait_time=0)
    else:
        print("No sessions found in data/raw/realsense_d435/")
