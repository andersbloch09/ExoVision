import grpc
import os
import time
import asyncio
from dotenv import load_dotenv
import vision_pb2
import vision_pb2_grpc

load_dotenv()

async def send_image():
    """Stream images to receiver via gRPC."""
    denmark_host = os.getenv("DENMARK_HOST")
    if not denmark_host:
        raise ValueError("DENMARK_HOST is not set in .env")
    
    # Connect to gRPC server
    channel = grpc.aio.insecure_channel(
        f"{denmark_host}:50051"
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
    
    try:
        client_send_time = int(time.time() * 1000)
        send_frames_gen = send_frames()
        async for prediction in stub.StreamInference(send_frames_gen):
            client_recv_time = int(time.time() * 1000)
            
            # Calculate latencies
            send_to_recv = prediction.server_receive_time_ms - client_send_time
            send_to_response = client_recv_time - client_send_time
            network_latency = send_to_response - prediction.inference_time_ms
            
            print(f"✓ Prediction: {prediction.label} ({prediction.confidence:.2f})")
            print(f"  Inference on server: {prediction.inference_time_ms}ms")
            print(f"  Network latency: {network_latency}ms")
            print(f"  Total round-trip: {send_to_response}ms")
    finally:
        await channel.close()

async def update_model_weights(model_bytes, version):
    """Send updated model weights to receiver."""
    denmark_host = os.getenv("DENMARK_HOST")
    channel = grpc.aio.insecure_channel(
        f"{denmark_host}:50051"
    )
    stub = vision_pb2_grpc.VisionModelStub(channel)
    
    update = vision_pb2.WeightUpdate(
        model_weights=model_bytes,
        version=version,
        timestamp=int(time.time() * 1000)
    )
    
    try:
        response = stub.UpdateWeights(update)
        print(f"Update response: {response.status} - {response.message}")
    finally:
        await channel.close()

async def main():
    """Main function to send image and update model."""
    await send_image()
    
    # Example of updating model weights (replace with actual bytes)
    # new_model_bytes = b"..."
    # await update_model_weights(new_model_bytes, version="1.1")

if __name__ == "__main__":
    asyncio.run(main())