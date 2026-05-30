import grpc
import os
import time
import asyncio
import numpy as np
from dotenv import load_dotenv
import vision_pb2
import vision_pb2_grpc

try:
    import pyrealsense2 as rs
except ImportError:
    print("Warning: pyrealsense2 not installed. Install with: pip install pyrealsense2")
    rs = None

load_dotenv()


class RealSenseStreamer:
    """Stream RGB and point cloud from RealSense camera."""
    
    def __init__(self):
        """Initialize RealSense pipeline."""
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        
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
        
        # Get intrinsics
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    
    def get_pointcloud_from_depth(self, depth_frame):
        """Convert depth frame to point cloud."""
        
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        
        h, w = depth_data.shape
        points = []
        
        for y in range(0, h, 2):  # Sample every 2 pixels
            for x in range(0, w, 2):
                depth = depth_data[y, x] * depth_scale
                
                if depth == 0 or depth > 10:  # Skip invalid/too far
                    continue
                
                # Convert to 3D point
                x_3d = (x - self.depth_intrinsics.ppx) * depth / self.depth_intrinsics.fx
                y_3d = (y - self.depth_intrinsics.ppy) * depth / self.depth_intrinsics.fy
                
                points.append([x_3d, y_3d, depth])
        
        if len(points) == 0:
            return np.array([], dtype=np.float32)
        
        return np.array(points, dtype=np.float32)
    
    def get_next_frame(self):
        """Get next RGB image and point cloud."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        rgb_data = np.asanyarray(color_frame.get_data())
        pointcloud = self.get_pointcloud_from_depth(depth_frame)
        
        return rgb_data, pointcloud
    
    def stop(self):
        """Stop camera."""
        self.pipeline.stop()


async def send_frames():
    """Stream live frames to receiver via gRPC."""
    denmark_host = os.getenv("DENMARK_HOST", "localhost")
    
    print(f"Connecting to server at {denmark_host}:50051...")
    
    # Connect to gRPC server
    channel = grpc.aio.insecure_channel(f"{denmark_host}:50051")
    stub = vision_pb2_grpc.VisionModelStub(channel)
    
    try:
        # Initialize camera
        streamer = RealSenseStreamer()
        
        frame_count = 0
        
        async def frame_generator():
            """Generate frames from camera."""
            nonlocal frame_count
            
            try:
                while True:
                    rgb_data, pointcloud = streamer.get_next_frame()
                    
                    if rgb_data is None or pointcloud is None or len(pointcloud) == 0:
                        continue
                    
                    # Encode RGB as JPEG bytes
                    import cv2
                    _, rgb_bytes = cv2.imencode('.jpg', rgb_data)
                    
                    # Encode point cloud as binary
                    pc_bytes = pointcloud.tobytes()
                    
                    frame_count += 1
                    
                    frame = vision_pb2.ImageFrame(
                        image_data=rgb_bytes.tobytes(),
                        pointcloud_data=pc_bytes,
                        filename=f"frame_{frame_count:06d}.jpg",
                        timestamp=int(time.time() * 1000)
                    )
                    
                    yield frame
                    
                    # Send at ~30fps
                    await asyncio.sleep(0.033)
            
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                streamer.stop()
        
        # Stream frames and receive predictions
        async for prediction in stub.StreamInference(frame_generator()):
            print(f"\n[Frame {frame_count}]")
            print(f"  Type: {prediction.stair_type}")
            print(f"  Confidence: {prediction.confidence:.3f}")
            
            if prediction.distance_m > 0:
                print(f"  ⚠️  ALERT - Distance: {prediction.distance_m:.2f}m")
            else:
                print(f"  Distance: N/A")
            
            print(f"  Inference: {prediction.inference_time_ms}ms")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(send_frames())