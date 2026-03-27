from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)
DATA_DIR = os.path.join("data", "images")
os.makedirs(DATA_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@app.route("/upload", methods=["POST"])
def upload_image():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    if uploaded_file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(DATA_DIR, f"{int(time.time())}_{uploaded_file.filename}")
    try:
        uploaded_file.save(file_path)
        print(f"Saved image: {file_path}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = {
        "status": "received",
        "filename": os.path.basename(file_path),
        "size_bytes": os.path.getsize(file_path)
    }
    return jsonify(response)

def receive_and_respond():
    app.run(host="0.0.0.0", port=5000, threaded=True)