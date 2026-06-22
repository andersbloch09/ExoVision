import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class StairDetector:
    """
    Detect stairs in RGB images and calculate distance from point cloud regions.
    """
    
    def __init__(self, model_path: str, 
                 confidence_threshold: float = 0.5):
        """
        Initialize stair detector.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class names mapping
        self.class_names = {
            0: "flat_ground",
            1: "ascending_stairs",
            2: "descending_stairs"
        }
    
    def detect_stairs(self, rgb_image: np.ndarray, 
                     depth_image: np.ndarray,
                     camera_intrinsics: dict) -> dict:
        """
        Detect stairs in RGB image and calculate distance using isolated depth regions.
        """
        # Run YOLO inference
        results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
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
            # Isolate matrix array crop and project only the pixels within the box boundary
            pointcloud = self._generate_bbox_pointcloud(depth_image, bbox, camera_intrinsics)
            
            if pointcloud is not None and len(pointcloud) > 50:
                distance = self._detect_step_distance(pointcloud)
            
            # Sanity filter
            if distance < 0 or distance > 10:
                distance = -1
        
        return {
            'stair_type': stair_type,
            'confidence': confidence,
            'distance_m': distance,
            'bbox': bbox
        }
    
    def _generate_bbox_pointcloud(
        self,
        depth_img: np.ndarray,
        bbox: list,
        intrinsics: dict
    ) -> np.ndarray:
        """
        Generates a 3D point cloud ONLY for pixels inside the YOLO bounding box.
        All coordinates are kept in full-image space (consistent projection).
        """

        try:
            x1, y1, x2, y2 = bbox

            h_img, w_img = depth_img.shape[:2]

            # Clip bbox safely
            x1 = max(0, min(w_img - 1, x1))
            x2 = max(0, min(w_img, x2))
            y1 = max(0, min(h_img - 1, y1))
            y2 = max(0, min(h_img, y2))

            if x2 <= x1 or y2 <= y1:
                return None

            # Intrinsics
            fx = intrinsics.get('fx', 500.0)
            fy = intrinsics.get('fy', 500.0)
            cx = intrinsics.get('cx', w_img / 2.0)
            cy = intrinsics.get('cy', h_img / 2.0)
            depth_scale = intrinsics.get('scale', 0.001)

            # -----------------------------
            # Step 1: extract ROI depth
            # -----------------------------
            roi = depth_img[y1:y2:2, x1:x2:2].astype(np.float32)
            roi_depth = roi * depth_scale

            if roi_depth.size == 0:
                return None

            # -----------------------------
            # Step 2: build FULL-FRAME pixel coordinates (IMPORTANT FIX)
            # -----------------------------
            ys, xs = np.mgrid[y1:y2:2, x1:x2:2]

            # Flatten everything
            z = roi_depth.reshape(-1)
            x_pix = xs.reshape(-1)
            y_pix = ys.reshape(-1)

            # -----------------------------
            # Step 3: valid depth filtering
            # -----------------------------
            valid = (z > 0) & (z < 10)

            if np.sum(valid) < 50:
                return None

            z = z[valid]
            x_pix = x_pix[valid]
            y_pix = y_pix[valid]

            # -----------------------------
            # Step 4: correct projection (consistent frame)
            # -----------------------------
            x = (x_pix - cx) * z / fx
            y = (y_pix - cy) * z / fy

            return np.stack((x, y, z), axis=-1)

        except Exception as e:
            print(f"Error generating bbox point cloud: {e}")
            return None    
        
        
    def _detect_step_distance(self, cropped_pointcloud: np.ndarray) -> float:
        """
        Robust stair detection using height-binned depth profile + stable step transitions.
        """
        try:
            if len(cropped_pointcloud) < 100:
                return -1

            y = cropped_pointcloud[:, 1]  # height
            z = cropped_pointcloud[:, 2]  # depth

            # Clean invalid depth
            valid_mask = (z > 0) & (z < 10)
            y = y[valid_mask]
            z = z[valid_mask]

            if len(z) < 100:
                return -1

            # Height binning
            bin_size = 0.01  # 1cm
            y_min, y_max = np.min(y), np.max(y)

            if y_max - y_min < 0.1:
                # Not enough vertical structure variation
                return float(np.median(z))

            bins = np.arange(y_min, y_max + bin_size, bin_size)

            z_profile = []
            for i in range(len(bins) - 1):
                mask = (y >= bins[i]) & (y < bins[i + 1])
                if np.sum(mask) > 30:  # Reject sparse depth dropouts
                    z_profile.append(np.median(z[mask]))
                else:
                    z_profile.append(np.nan)

            z_profile = np.array(z_profile)

            # Interpolate missing profile values
            valid = ~np.isnan(z_profile)
            if np.sum(valid) < 5:
                return float(np.median(z))

            z_profile = np.interp(
                np.arange(len(z_profile)),
                np.where(valid)[0],
                z_profile[valid]
            )

            # Smooth noise out
            z_profile = gaussian_filter1d(z_profile, sigma=1.0)

            # Compute depth gradient
            dz = np.diff(z_profile)
            dz = gaussian_filter1d(dz, sigma=1.0)

            # Detect dynamic stable peaks
            thresh = max(np.std(dz) * 0.8, 0.01)  # Added 1cm safety floor threshold
            edges, _ = find_peaks(dz, height=thresh, distance=3)

            # Identify structural jump boundaries
            if len(edges) > 0:
                verified_steps = []
                
                # 1. Scan all peaks first to verify how many are actual steps
                for idx in edges:
                    if idx < 2 or idx >= len(z_profile) - 2:
                        continue

                    pre = np.mean(z_profile[max(0, idx-3):idx])
                    post = np.mean(z_profile[idx:idx+3])
                    jump = post - pre

                    if jump > 0.05:  # 5cm minimum real step transition
                        verified_steps.append(pre)

                # 2. Print the final count to the terminal if steps were found
                if len(verified_steps) > 0:
                    print(f"🪜 Visible steps counted in terminal: {len(verified_steps)}")
                    
                    # 3. Return ONLY the distance to the first step (matching your original code)
                    return float(verified_steps[0])

            # Fallback: closest target surface point
            return float(np.percentile(z, 10))

        except Exception as e:
            print(f"Step detection error: {e}")
            return -1