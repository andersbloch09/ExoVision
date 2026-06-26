import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import os
import json
import time
import sys
import subprocess

class StairDetector:
    """
    Detect stairs in RGB images and calculate distance from point cloud regions.
    """
    
    def __init__(self, model_path: str, 
                 confidence_threshold: float = 0.5,
                 sample_confidence_threshold: float = 0.8,
                 training_threshold: int = 3000,
                 enable_training: bool = True):
        """
        Initialize stair detector.
        """

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.sample_confidence_threshold = sample_confidence_threshold
        self.enable_training = enable_training
        self.sample_path = "data/samples/train/images"
        self.sample_count = len(os.listdir(self.sample_path)) if os.path.exists(self.sample_path) else 0
        self.training_threshold = training_threshold
        self.added_training_flag = self.sample_count
        self.training_in_progress = False

        # Class names mapping
        self.class_names = {
            0: "flat_ground",
            1: "ascending_stairs",
            2: "descending_stairs"
        }

        self.last_frame = None  # For frame difference check


    def swap_model(self):
        """
        Replace best.pt with best_new.pt and reload YOLO model.
        """
        self.training_in_progress = False  # Reset training flag
        old_model_path = r"scripts/models/best.pt"
        new_model_path = r"scripts/models/best_new.pt"

        if not os.path.exists(new_model_path):
            raise FileNotFoundError(f"New model not found at {new_model_path}")

        # Delete old model if it exists
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
            print(f"Deleted old model: {old_model_path}")

        # Rename new → old name
        os.rename(new_model_path, old_model_path)
        print(f"Renamed {new_model_path} → {old_model_path}")

        # Reload model
        self.model = YOLO(old_model_path)
        print("Model reloaded successfully")
        

    def _convert_bbox_to_yolo(self, bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox

        x_center = (x1 + x2) / 2.0 / img_w
        y_center = (y1 + y2) / 2.0 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        return x_center, y_center, width, height


    def save_sample(self, rgb_image, bbox, stair_type, confidence):
        try:
            if stair_type == "none":
                return
            
            base_dir = "data/samples"
            images_dir = os.path.join(base_dir, "train", "images")
            labels_dir = os.path.join(base_dir, "train", "labels")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            timestamp = int(time.time() * 1000)
            img_name = f"{timestamp}.jpg"

            h, w = rgb_image.shape[:2]

            # 1. Save image
            img_path = os.path.join(images_dir, img_name)
            cv2.imwrite(img_path, rgb_image)

            self.added_training_flag += 1


            if self.added_training_flag - self.sample_count >= self.training_threshold and not self.training_in_progress and self.enable_training:
                print(f"⚠️  Training threshold reached: {self.added_training_flag} samples. Starting training...")
                self.start_training()
                self.sample_count = len(os.listdir(self.sample_path)) if os.path.exists(self.sample_path) else 0
                self.added_training_flag = self.sample_count

            # 2. Convert label
            class_map = {
                "flat_ground": 0,
                "ascending_stairs": 1,
                "descending_stairs": 2
            }

            class_id = class_map.get(stair_type, -1)
            if class_id == -1:
                return

            x, y, bw, bh = self._convert_bbox_to_yolo(bbox, w, h)

            # 3. Save label file
            label_path = os.path.join(labels_dir, f"{timestamp}.txt")

            with open(label_path, "w") as f:
                f.write(f"{class_id} {x} {y} {bw} {bh}\n")

            print(f"Saved YOLO sample: {img_name}")

        except Exception as e:
            print(f"Logging error: {e}")


    def start_training(self):
        self.training_in_progress = True
        subprocess.Popen([
            sys.executable,
            "training/train.py"
        ])

    def detect_stairs(self, rgb_image: np.ndarray, 
                     depth_image: np.ndarray,
                     camera_intrinsics: dict) -> dict:
        """
        Detect stairs in RGB image and calculate distance using isolated depth regions.
        """
        if os.path.exists(r"scripts/models/best_new.pt"):
            self.swap_model()
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
        confidence = float(best_box.conf[0])

        if len(results) > 0 and len(results[0].boxes) > 0:
            if self.last_frame is not None:
                diff = np.mean(np.abs(rgb_image.astype(np.float32) - self.last_frame.astype(np.float32)))

                if diff < 5:
                    return {
                'stair_type': 'none',
                'confidence': 0.0,
                'distance_m': -1,
                'bbox': None
            }

            self.last_frame = rgb_image.copy()

            if confidence > self.sample_confidence_threshold and self.training_in_progress == False:
                self.save_sample(
                    rgb_image=rgb_image,
                    bbox=results[0].boxes.xyxy[0].cpu().numpy().astype(int),
                    stair_type=results[0].names[int(results[0].boxes.cls[0])],
                    confidence=confidence,
                )

        
        class_id = int(best_box.cls[0])
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