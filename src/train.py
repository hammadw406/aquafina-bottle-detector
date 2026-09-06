"""
Stage 3: Train the Aquafina bottle detector (YOLOv8) on Kaggle or locally.

On Kaggle (GPU notebook):
    !pip install ultralytics -q
    !python src/train.py --data /kaggle/input/aquafina-dataset/data.yaml --epochs 100

Locally:
    python src/train.py --data data/processed/data.yaml --epochs 100 --device cpu
"""
import argparse

from ultralytics import YOLO


def train(data_yaml: str, model_name: str, epochs: int, imgsz: int, batch: int, device: str, patience: int):
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project="runs/detect",
        name="aquafina",
        plots=True,
    )
    print("Training complete. Best weights at: runs/detect/aquafina/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Aquafina bottle detector")
    parser.add_argument("--data", required=True, help="Path to data.yaml (YOLO format)")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, args.patience)


if __name__ == "__main__":
    main()
