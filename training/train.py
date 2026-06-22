from ultralytics import YOLO
import torch
import gc

def train_yolo_model():
    # model types include: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolov8n.pt')

    """model.train(
        data='data/datasets/exovision_v1.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        lr0=0.001,
        weight_decay=0.0005,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        val=True,             # Enable validation during training
        patience=10,
        workers=0,
    )"""

    model.train(
        data='data/datasets/exovision_v1.yaml',
        epochs=25,
        imgsz=640,
        batch=16,  # or 32 if GPU allows
        lr0=0.001,
        weight_decay=0.0005,
        device='cuda',
        val=True,
        patience=5,  # reduced
        workers=4,   # increased
        cache='ram', # added
    )



    metrics = model.val(save=True)
    print(f"✅ Final mAP@0.5: {metrics.box.map50:.4f}")
    print(f"📊 mAP@[.5:.95]: {metrics.box.map:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print("🚀 Starting YOLOv8 training...")
    train_yolo_model()
    