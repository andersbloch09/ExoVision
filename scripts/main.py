from dotenv import load_dotenv
import os
import asyncio
from sender import send_frames
from receiver import serve


load_dotenv()

role = os.getenv("ROLE", "sender").lower()

if role == "sender":
    print("Running as sender (gRPC client)...")
    asyncio.run(send_frames())
elif role == "receiver":
    print("Running as receiver (gRPC server)...")
    serve()


else:
    raise ValueError(f"Unknown ROLE: {role}")


