import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/vision/ExoVision/.env")

def send_image():
    """Send an image to the server defined in DENMARK_HOST."""
    denmark_host = os.getenv("DENMARK_HOST")
    if not denmark_host:
        raise ValueError("DENMARK_HOST is not set in .env")

    url = f"http://{denmark_host}:5000/upload"

    image_path = "data/images/test.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        try:
            response = requests.post(url, files=files)
        except requests.exceptions.ConnectionError:
            print(f"Could not connect to {url}. Is the receiver running?")
            return

    if response.status_code == 200:
        print("Server response:", response.json())
    else:
        print("Failed:", response.text)