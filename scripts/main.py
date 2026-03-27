# main.py
from dotenv import load_dotenv
import os
from sender import send_image
from receiver import receive_and_respond

# Load environment variables from .env
load_dotenv()  # looks for .env in the same directory

# Determine role from environment variable
role = os.getenv("ROLE", "sender").lower()

if role == "sender":
    print("Running as sender...")
    send_image()
elif role == "receiver":
    print("Running as receiver...")
    receive_and_respond()
else:
    raise ValueError(f"Unknown ROLE: {role}")