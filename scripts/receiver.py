import grpc
from concurrent import futures
import os
import time
from dotenv import load_dotenv
import numpy as np
import cv2
import json
import vision_pb2
import vision_pb2_grpc
from stair_detector import StairDetector
import threading

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("Warning: open3d not installed. Install with: pip install open3d")

load_dotenv()

DATA_DIR = os.path.join("data", "images")
os.makedirs(DATA_DIR, exist_ok=True)

class VisionModelService(vision_pb2_grpc.VisionModelServicer):
    def __init__(self):
        self.model_version = "1.0"
        self.pointcloud_viz = None
        self.viz_thread = None
        
        try:
            # Load configuration
            config_path = os.path.join(
                os.path.dirname(__file__),
                "stair_config.json"
            )
            
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            stair_config = config.get('stair_detection', {})
            viz_config = config.get('visualization', {})
            self.visualization_enabled = viz_config.get('enabled', False)
            self.rgb_display = viz_config.get('rgb_display', True)
            self.pointcloud_display = viz_config.get('pointcloud_display', True)
            
            # Build full path to model
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                stair_config.get('model_path', 'runs/detect/train/weights/best.pt')
            )
            
            print(f"Model path: {model_path}")
            print(f"Model exists: {os.path.exists(model_path)}")
            print(f"Visualization: {'enabled' if self.visualization_enabled else 'disabled'}")
            if self.visualization_enabled:
                print(f"  RGB display: {'on' if self.rgb_display else 'off'}")
                print(f"  Point cloud display: {'on' if self.pointcloud_display else 'off'}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Initialize stair detector with config values
            print("Initializing StairDetector...")
            self.stair_detector = StairDetector(
                model_path=model_path,
                distance_min_m=stair_config.get('distance_threshold', {}).get('min_m', 1.0),
                distance_max_m=stair_config.get('distance_threshold', {}).get('max_m', 3.0),
                confidence_threshold=stair_config.get('confidence_threshold', 0.5)
            )
            print("✓ StairDetector initialized successfully!")
        
        except Exception as e:
            print(f"✗ Error initializing VisionModelService: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def visualize_pointcloud(self, pointcloud, bbox=None, stair_type=None):
        """Visualize point cloud with optional bounding box."""
        if not self.pointcloud_display or o3d is None:
            return
        
        if len(pointcloud) == 0:
            return
        
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            
            # Color by height (Y-axis) for step visualization
            colors = np.zeros_like(pointcloud)
            if len(pointcloud) > 0:
                y_min, y_max = pointcloud[:, 1].min(), pointcloud[:, 1].max()
                y_range = y_max - y_min if y_max > y_min else 1.0
                # Red to green gradient by height
                normalized_y = (pointcloud[:, 1] - y_min) / y_range
                colors[:, 0] = 1 - normalized_y  # Red decreases
                colors[:, 1] = normalized_y      # Green increases
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Create or update visualizer
            if self.pointcloud_viz is None:
                self.pointcloud_viz = o3d.visualization.Visualizer()
                self.pointcloud_viz.create_window(window_name="Point Cloud Stream")
                self.pointcloud_viz.add_geometry(pcd)
            else:
                self.pointcloud_viz.clear_geometries()
                self.pointcloud_viz.add_geometry(pcd)
            
            # Update camera view
            self.pointcloud_viz.poll_events()
            self.pointcloud_viz.update_renderer()
            
        except Exception as e:
            print(f"Warning: Point cloud visualization error: {e}")
    
    def StreamInference(self, request_iterator, context):
        """
        Receive RGB image frames and point clouds, return stair detections.
        Bidirectional streaming for continuous inference.
        """
        for frame in request_iterator:
            start_inference_time = time.time()
            
            try:
                # Deserialize RGB image
                nparr = np.frombuffer(frame.image_data, np.uint8)
                rgb_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if rgb_image is None:
                    context.abort(grpc.StatusCode.INTERNAL, "Failed to decode image")
                
                # Deserialize point cloud (actually depth image from ROS)
                pointcloud = None
                if frame.pointcloud_data:
                    try:
                        # Decode PNG depth image from ROS
                        depth_img = cv2.imdecode(
                            np.frombuffer(frame.pointcloud_data, np.uint8),
                            cv2.IMREAD_UNCHANGED
                        )
                        
                        if depth_img is None:
                            print(f"Warning: Failed to decode depth image")
                            pointcloud = np.array([])
                        else:
                            print(f"✓ Decoded depth image: {depth_img.shape}, dtype: {depth_img.dtype}")
                            
                            # Convert depth image to point cloud using camera intrinsics
                            # Assuming Orbbec ASTRA S approximate intrinsics
                            fx = 500.0   # Focal length x
                            fy = 500.0   # Focal length y
                            cx = depth_img.shape[1] / 2.0  # Principal point x
                            cy = depth_img.shape[0] / 2.0  # Principal point y
                            depth_scale = 0.001  # Typically 1mm = 0.001m for depth images
                            
                            h, w = depth_img.shape
                            points = []
                            
                            for y in range(0, h, 2):  # Sample every 2 pixels
                                for x in range(0, w, 2):
                                    depth = depth_img[y, x] * depth_scale
                                    
                                    if depth == 0 or depth > 10:  # Skip invalid/too far
                                        continue
                                    
                                    # Convert to 3D point using intrinsics
                                    x_3d = (x - cx) * depth / fx
                                    y_3d = (y - cy) * depth / fy
                                    
                                    points.append([x_3d, y_3d, depth])
                            
                            if len(points) > 0:
                                pointcloud = np.array(points, dtype=np.float32)
                                print(f"✓ Generated point cloud: {len(pointcloud)} points")
                            else:
                                print(f"Warning: No valid depth points generated")
                                pointcloud = np.array([])
                                
                    except Exception as e:
                        print(f"Warning: Failed to deserialize point cloud: {e}")
                        import traceback
                        traceback.print_exc()
                        pointcloud = np.array([])
                
                # Run stair detection
                if pointcloud is not None and len(pointcloud) > 0:
                    detection = self.stair_detector.detect_stairs(
                        rgb_image, 
                        pointcloud
                    )
                else:
                    # No point cloud, return no detection
                    detection = {
                        'stair_type': 'none',
                        'confidence': 0.0,
                        'distance_m': -1,
                        'bbox': None
                    }
                
                # Calculate inference time
                inference_time_ms = int((time.time() - start_inference_time) * 1000)
                
                # Visualization with bounding boxes
                if self.visualization_enabled:
                    # RGB visualization
                    if self.rgb_display:
                        display_image = rgb_image.copy()
                        
                        # Draw bounding box if detection found
                        if detection['bbox'] is not None:
                            x1, y1, x2, y2 = detection['bbox']
                            # Color based on stair type
                            if detection['stair_type'] == 'ascending_stairs':
                                color = (0, 255, 0)  # Green
                            elif detection['stair_type'] == 'descending_stairs':
                                color = (0, 165, 255)  # Orange
                            else:
                                color = (0, 0, 255)  # Red
                            
                            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add detection info text
                        text = f"{detection['stair_type']} ({detection['confidence']:.2f})"
                        cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if detection['distance_m'] > 0:
                            dist_text = f"Distance: {detection['distance_m']:.2f}m"
                            cv2.putText(display_image, dist_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Display
                        cv2.imshow('Incoming Stream', display_image)
                        cv2.waitKey(1)
                    
                    # Point cloud visualization (runs in same thread, non-blocking)
                    if self.pointcloud_display and pointcloud is not None and len(pointcloud) > 0:
                        self.visualize_pointcloud(pointcloud, bbox=detection['bbox'], stair_type=detection['stair_type'])
                
                # Create prediction response
                prediction = vision_pb2.Prediction(
                    filename=frame.filename,
                    stair_type=detection['stair_type'],
                    confidence=detection['confidence'],
                    distance_m=detection['distance_m'],
                    inference_time_ms=inference_time_ms
                )
                
                print(f"Detection: {detection['stair_type']}, "
                      f"Distance: {detection['distance_m']:.2f}m, "
                      f"Confidence: {detection['confidence']:.3f}, "
                      f"Time: {inference_time_ms}ms")
                
                yield prediction
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def UpdateWeights(self, request, context):
        """Receive and apply model weight updates."""
        try:
            model_path = os.path.join("models", f"model_v{request.version}.pt")
            with open(model_path, "wb") as f:
                f.write(request.model_weights)
            self.model_version = request.version
            print(f"Updated model to version {request.version}")
            return vision_pb2.UpdateResponse(
                status="success",
                message=f"Model updated to {request.version}"
            )
        except Exception as e:
            return vision_pb2.UpdateResponse(
                status="error",
                message=str(e)
            )

def receive_and_respond():
    """Start gRPC server."""
    print("Starting gRPC server initialization...")
    try:
        service = VisionModelService()
        print("✓ VisionModelService created successfully")
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        vision_pb2_grpc.add_VisionModelServicer_to_server(service, server)
        server.add_insecure_port("[::]:50051")
        print("gRPC server listening on port 50051...")
        server.start()
        server.wait_for_termination()
    except Exception as e:
        print(f"✗ Error starting gRPC server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    receive_and_respond()