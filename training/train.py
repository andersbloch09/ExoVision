from ultralytics import YOLO
import torch
import gc

def train_yolo_model():
    # model types include: yolo26n.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
    model = YOLO('yolo26n.pt')

    model.train(
        data='data/numbers.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        lr0=0.001,
        weight_decay=0.0005,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        val=True,             # Enable validation during training
        patience=10,
        workers=0,
    )

    metrics = model.val(save=True)
    print(f"✅ Final mAP@0.5: {metrics.box.map50:.4f}")
    print(f"📊 mAP@[.5:.95]: {metrics.box.map:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print("🚀 Starting YOLO26 improved training...")
    train_yolo_model()
    