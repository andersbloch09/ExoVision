import os
import requests

def send_image():
    denmark_host = os.getenv("DENMARK_HOST")
    url = f"http://{denmark_host}:5000/upload"  # port must match receiver

    image_path = "data/images/test.jpg"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Server response:", response.json())
    else:
        print("Failed:", response.text)