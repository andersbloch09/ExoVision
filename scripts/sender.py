import grpc
import os
import time
from dotenv import load_dotenv
import vision_pb2
import vision_pb2_grpc

load_dotenv()

def send_image():
    """Stream images to receiver via gRPC."""
    denmark_host = os.getenv("DENMARK_HOST")
    if not denmark_host:
        raise ValueError("DENMARK_HOST is not set in .env")
    
    # Connect to gRPC server
    channel = grpc.aio.secure_channel(
        f"{denmark_host}:50051",
        grpc.aio.ssl_channel_credentials()
    )
    stub = vision_pb2_grpc.VisionModelStub(channel)
    
    image_path = "data/images/test.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    async def send_frames():
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        frame = vision_pb2.ImageFrame(
            image_data=image_data,
            filename=os.path.basename(image_path),
            timestamp=int(time.time() * 1000)
        )
        yield frame
    
    async def receive_predictions():
        async for prediction in stub.StreamInference(send_frames()):
            print(f"Prediction: {prediction.label} ({prediction.confidence:.2f})")
            print(f"Inference time: {prediction.inference_time_ms}ms")
    
    # Run async client
    import asyncio
    asyncio.run(receive_predictions())
    channel.close()

def update_model_weights(model_bytes, version):
    """Send updated model weights to receiver."""
    denmark_host = os.getenv("DENMARK_HOST")
    channel = grpc.secure_channel(
        f"{denmark_host}:50051",
        grpc.ssl_channel_credentials()
    )
    stub = vision_pb2_grpc.VisionModelStub(channel)
    
    update = vision_pb2.WeightUpdate(
        model_weights=model_bytes,
        version=version,
        timestamp=int(time.time() * 1000)
    )
    
    response = stub.UpdateWeights(update)
    print(f"Update response: {response.status} - {response.message}")
    channel.close()