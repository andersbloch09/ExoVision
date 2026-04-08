import grpc
from concurrent import futures
import os
import time
from dotenv import load_dotenv
import vision_pb2
import vision_pb2_grpc

load_dotenv()

DATA_DIR = os.path.join("data", "images")
os.makedirs(DATA_DIR, exist_ok=True)

class VisionModelService(vision_pb2_grpc.VisionModelServicer):
    def __init__(self):
        self.model_version = "1.0"
        self.model_path = "models/current_model.pt"
    
    def StreamInference(self, request_iterator, context):
        """
        Receive image frames and send back predictions.
        Bidirectional streaming for continuous inference.
        """
        for frame in request_iterator:
            server_receive_time = int(time.time() * 1000)  # Timestamp when received
            
            # Save received image
            file_path = os.path.join(
                DATA_DIR, 
                f"{int(time.time())}_{frame.filename}"
            )
            try:
                with open(file_path, "wb") as f:
                    f.write(frame.image_data)
                print(f"Saved image: {file_path}")
            except Exception as e:
                context.abort(grpc.StatusCode.INTERNAL, f"Save failed: {str(e)}")
            
            # Measure actual inference time
            start_time = time.time()
            
            # Simulate YOLO inference (replace with actual model)
            # Simulating ~50ms for YOLOv8 nano on Jetson
            time.sleep(0.050)  # 50ms simulated inference
            detections = [
                {"class": "person", "confidence": 0.95},
                {"class": "hand", "confidence": 0.87}
            ]
            
            inference_time_ms = int((time.time() - start_time) * 1000)
            server_send_time = int(time.time() * 1000)
            
            prediction = vision_pb2.Prediction(
                filename=frame.filename,
                confidence=0.95,
                label="exoplanet",
                inference_time_ms=inference_time_ms,
                server_receive_time_ms=server_receive_time,
                server_send_time_ms=server_send_time
            )
            print(f"Inference: {inference_time_ms}ms - Detections: {len(detections)}")
            yield prediction
    
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vision_pb2_grpc.add_VisionModelServicer_to_server(
        VisionModelService(), server
    )
    server.add_insecure_port("[::]:50051")
    print("gRPC server listening on port 50051...")
    server.start()
    server.wait_for_termination()