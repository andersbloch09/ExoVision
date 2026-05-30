"""
Stair detection module using YOLOv8 and RealSense point cloud data.
Detects stairs in RGB images and calculates distance to first step from point cloud.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import os


class StairDetector:
    """
    Detect stairs in RGB images and calculate distance from point cloud.
    """
    
    def __init__(self, model_path: str, 
                 distance_min_m: float = 1.0,
                 distance_max_m: float = 3.0,
                 confidence_threshold: float = 0.5):
        """
        Initialize stair detector.
        
        Args:
            model_path: Path to trained YOLO model weights
            distance_min_m: Minimum distance to alert (meters)
            distance_max_m: Maximum distance to alert (meters)
            confidence_threshold: YOLO confidence threshold (0-1)
        """
        self.model = YOLO(model_path)
        self.distance_min = distance_min_m
        self.distance_max = distance_max_m
        self.confidence_threshold = confidence_threshold
        
        # Class names mapping
        self.class_names = {
            0: "flat_ground",
            1: "ascending_stairs",
            2: "descending_stairs"
        }
    
    def detect_stairs(self, rgb_image: np.ndarray, 
                     pointcloud: np.ndarray) -> dict:
        """
        Detect stairs in RGB image and calculate distance from point cloud.
        
        Args:
            rgb_image: RGB image array (H x W x 3)
            pointcloud: Point cloud array (N x 3) where each row is [x, y, z]
        
        Returns:
            dict with keys:
                - stair_type: "ascending_stairs", "descending_stairs", "flat_ground", or "none"
                - confidence: float (0-1)
                - distance_m: float or -1 if not applicable
                - bbox: [x1, y1, x2, y2] bounding box or None
        """
        
        # Run YOLO inference
        results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            # No detections
            return {
                'stair_type': 'none',
                'confidence': 0.0,
                'distance_m': -1,
                'bbox': None
            }
        
        # Get best detection (highest confidence)
        detections = results[0].boxes
        best_idx = np.argmax(detections.conf.cpu().numpy())
        best_box = detections[best_idx]
        
        class_id = int(best_box.cls[0])
        confidence = float(best_box.conf[0])
        stair_type = self.class_names.get(class_id, "unknown")
        
        # Extract bounding box
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        # Calculate distance only for ascending stairs
        distance = -1
        if stair_type == "ascending_stairs":
            distance = self._calculate_distance_to_first_step(
                pointcloud, bbox, rgb_image.shape
            )
        
        return {
            'stair_type': stair_type,
            'confidence': confidence,
            'distance_m': distance,
            'bbox': bbox
        }
    
    def _calculate_distance_to_first_step(self, pointcloud: np.ndarray,
                                         bbox: list, 
                                         image_shape: tuple) -> float:
        """
        Calculate distance to first step from point cloud region.
        
        Args:
            pointcloud: Point cloud (N x 3)
            bbox: Bounding box [x1, y1, x2, y2] in image coordinates
            image_shape: RGB image shape (height, width, ...)
        
        Returns:
            Distance in meters, or -1 if calculation fails
        """
        try:
            x1, y1, x2, y2 = bbox
            img_h, img_w = image_shape[:2]
            
            # Normalize bbox to 0-1 range
            x1_norm = x1 / img_w
            y1_norm = y1 / img_h
            x2_norm = x2 / img_w
            y2_norm = y2 / img_h
            
            # Filter points in bbox region
            # Assuming pointcloud is in normalized camera coordinates (-1 to 1 or 0 to 1)
            # Adjust if your pointcloud uses different coordinate system
            pc_x = pointcloud[:, 0]
            pc_y = pointcloud[:, 1]
            pc_z = pointcloud[:, 2]  # Depth/distance
            
            # Create mask for points in bbox
            mask = (
                (pc_x >= x1_norm) & (pc_x <= x2_norm) &
                (pc_y >= y1_norm) & (pc_y <= y2_norm)
            )
            
            if np.sum(mask) < 50:
                # Too few points in region
                return -1
            
            cropped_pc = pointcloud[mask]
            
            # Detect first step using depth discontinuities
            distance = self._detect_step_distance(cropped_pc)
            
            # Apply distance threshold
            if self.distance_min <= distance <= self.distance_max:
                return distance
            else:
                return -1
                
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return -1
    
    def _detect_step_distance(self, cropped_pointcloud: np.ndarray) -> float:
        """
        Detect distance to first step by analyzing depth changes.
        
        Simplified version for ascending stairs - finds the closest step edge
        by analyzing depth discontinuities in the Y-axis (height).
        
        Args:
            cropped_pointcloud: Point cloud cropped to stair region (N x 3)
        
        Returns:
            Distance in meters (Z-axis), or -1 if not found
        """
        try:
            if len(cropped_pointcloud) < 50:
                return -1
            
            y_values = cropped_pointcloud[:, 1]  # Height
            z_values = cropped_pointcloud[:, 2]  # Depth/distance
            
            y_min, y_max = y_values.min(), y_values.max()
            z_min, z_max = z_values.min(), z_values.max()
            
            # Bin by height
            bin_size = 0.01  # 1cm
            num_bins = int((y_max - y_min) / bin_size) + 1
            if num_bins < 3:
                return -1
            
            y_bins = np.linspace(y_min, y_max, num_bins)
            z_at_height = []
            
            for i in range(len(y_bins) - 1):
                mask = (y_values >= y_bins[i]) & (y_values < y_bins[i+1])
                if np.sum(mask) > 0:
                    z_at_height.append(np.median(z_values[mask]))
                else:
                    z_at_height.append(np.nan)
            
            z_at_height = np.array(z_at_height)
            
            # Find discontinuities (step edges)
            z_diffs = np.diff(z_at_height)
            z_diffs_smooth = gaussian_filter1d(np.nan_to_num(z_diffs), sigma=0.5)
            
            # Find peaks (depth increases = step edge)
            edges, _ = find_peaks(z_diffs_smooth, height=0.005, distance=2)
            
            if len(edges) > 0:
                # Distance to first step is the depth at the first edge
                first_step_idx = edges[0]
                distance_to_step = z_at_height[first_step_idx]
                
                if distance_to_step > 0:
                    return distance_to_step
            
            # Fallback: return median depth of closest 10% points
            closest_points = np.percentile(z_values, 10)
            if closest_points > 0:
                return closest_points
            
            return -1
            
        except Exception as e:
            print(f"Error detecting step distance: {e}")
            return -1
