import os
from sender import send_image
from receiver import receive_and_respond

# Determine role from environment variable
role = os.getenv("ROLE", "sender").lower()

if role == "sender":
    send_image()
elif role == "receiver":
    receive_and_respond()
else:
    raise ValueError(f"Unknown ROLE: {role}")