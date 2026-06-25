from concurrent import futures
import os
import time
from dotenv import load_dotenv
import numpy as np
import cv2
import json
import grpc
import vision_pb2
import vision_pb2_grpc
from stair_detector import StairDetector

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("Warning: open3d not installed. Install with: pip install open3d")

load_dotenv()

class VisionModelService(vision_pb2_grpc.VisionModelServicer):
    def __init__(self):
        self.model_version = "1.0"
        self.pointcloud_viz = None
        
        try:
            # Load configuration parameters
            config_path = os.path.join(os.path.dirname(__file__), "stair_config.json")
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            stair_config = config.get('stair_detection', {})
            viz_config = config.get('visualization', {})
            self.visualization_enabled = viz_config.get('enabled', False)
            self.rgb_display = viz_config.get('rgb_display', True)
            self.pointcloud_display = viz_config.get('pointcloud_display', True)
            
            # Build full path to model weight file
            model_path = os.path.join(
                os.path.dirname(__file__), "..",
                stair_config.get('model_path', 'scripts/models/best.pt')
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print("Initializing StairDetector...")
            self.stair_detector = StairDetector(
                model_path=model_path,
                confidence_threshold=stair_config.get('confidence_threshold', 0.5),
                sample_confidence_threshold=stair_config.get('sample_confidence_threshold', 0.8),
                training_threshold=stair_config.get('training_threshold', 3000)
            )
            print("✓ StairDetector initialized successfully!")
        
        except Exception as e:
            print(f"✗ Error initializing VisionModelService: {e}")
            raise
    
    def visualize_pointcloud(self, pointcloud, bbox=None, stair_type=None):
        """Visualize isolated bounding box point cloud data."""
        if not self.pointcloud_display or o3d is None or pointcloud is None or len(pointcloud) == 0:
            return
        
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            
            # Height visual gradient colors
            colors = np.zeros_like(pointcloud)
            y_min, y_max = pointcloud[:, 1].min(), pointcloud[:, 1].max()
            y_range = y_max - y_min if y_max > y_min else 1.0
            normalized_y = (pointcloud[:, 1] - y_min) / y_range
            colors[:, 0] = 1 - normalized_y  
            colors[:, 1] = normalized_y      
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if self.pointcloud_viz is None:
                self.pointcloud_viz = o3d.visualization.Visualizer()
                self.pointcloud_viz.create_window(window_name="Point Cloud BBox Stream", width=640, height=480)
                self.pointcloud_viz.add_geometry(pcd)
            else:
                self.pointcloud_viz.clear_geometries()
                self.pointcloud_viz.add_geometry(pcd)
            
            self.pointcloud_viz.poll_events()
            self.pointcloud_viz.update_renderer()
            
        except Exception as e:
            print(f"Warning: Point cloud visualization error: {e}")
    
    def StreamInference(self, request_iterator, context):
        """
        Receive RGB frames and depth images, return isolated bounding box stair evaluations.
        """
        for frame in request_iterator:
            start_inference_time = time.time()
            
            try:
                # Deserialize incoming RGB frame
                nparr = np.frombuffer(frame.image_data, np.uint8)
                rgb_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if rgb_image is None:
                    context.abort(grpc.StatusCode.INTERNAL, "Failed to decode image frame")
                
                h, w = rgb_image.shape[:2]
                
                # -------------------------------------------------------------
                # RETUNED: Optimized default profile guess for Orbbec Astra S (640x480)
                # -------------------------------------------------------------
                intrinsics = {
                    'fx': 575.0,     # Orbbec factory standard focal length X (~570-580)
                    'fy': 575.0,     # Orbbec factory standard focal length Y (~570-580)
                    'cx': 327.5,     # True optical center offset (typically right-biased)
                    'cy': 242.5,     # True optical center offset (typically slightly lowered)
                    'scale': 0.001   # Converts 16-bit millimeter depth data into meters
                }
                
                detection = {
                    'stair_type': 'none',
                    'confidence': 0.0,
                    'distance_m': -1,
                    'bbox': None
                }
                pointcloud_for_viz = None
                
                # Check for active depth map data
                if frame.pointcloud_data:
                    depth_img = cv2.imdecode(
                        np.frombuffer(frame.pointcloud_data, np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )
                    
                    if depth_img is not None:
                        # Pass the raw matrix straight into the detector engine
                        detection = self.stair_detector.detect_stairs(
                            rgb_image, 
                            depth_img, 
                            intrinsics
                        )
                        
                        # Handle targeted isolated pointcloud calculations for visualization
                        if self.visualization_enabled and self.pointcloud_display and detection['bbox'] is not None:
                            pointcloud_for_viz = self.stair_detector._generate_bbox_pointcloud(
                                depth_img, detection['bbox'], intrinsics
                            )
                
                inference_time_ms = int((time.time() - start_inference_time) * 1000)
                
                # Performance rendering metrics
                if self.visualization_enabled:
                    if self.rgb_display:
                        display_image = rgb_image.copy()
                        
                        if detection['bbox'] is not None:
                            x1, y1, x2, y2 = detection['bbox']
                            color = (0, 255, 0) if detection['stair_type'] == 'ascending_stairs' else (0, 165, 255)
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                        
                        text = f"{detection['stair_type']} ({detection['confidence']:.2f})"
                        cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if detection['distance_m'] > 0:
                            dist_text = f"Distance: {detection['distance_m']:.2f}m"
                            cv2.putText(display_image, dist_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.imshow('Incoming Stream', display_image)
                        cv2.waitKey(1)
                    
                    if self.pointcloud_display and pointcloud_for_viz is not None and len(pointcloud_for_viz) > 0:
                        self.visualize_pointcloud(pointcloud_for_viz, bbox=detection['bbox'], stair_type=detection['stair_type'])
                
                yield vision_pb2.Prediction(
                    filename=frame.filename,
                    stair_type=detection['stair_type'],
                    confidence=detection['confidence'],
                    distance_m=detection['distance_m'],
                    inference_time_ms=inference_time_ms
                )
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def UpdateWeights(self, request, context):
        try:
            model_path = os.path.join("models", f"model_v{request.version}.pt")
            with open(model_path, "wb") as f:
                f.write(request.model_weights)
            self.model_version = request.version
            return vision_pb2.UpdateResponse(
                status="success",
                message=f"Model updated to {request.version}"
            )
        except Exception as e:
            return vision_pb2.UpdateResponse(status="error", message=str(e))

def serve():
    print("Starting gRPC server initialization...")
    try:
        service = VisionModelService()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        vision_pb2_grpc.add_VisionModelServicer_to_server(service, server)
        server.add_insecure_port("[::]:50051")
        server.start()
        print("gRPC server listening on port 50051...")
        server.wait_for_termination()
    except Exception as e:
        print(f"✗ Error starting gRPC server: {e}")

if __name__ == "__main__":
    serve()