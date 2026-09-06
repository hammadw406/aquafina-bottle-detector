"""
Stage 5: Export the trained YOLOv8 model to ONNX for browser deployment.

Usage:
    python src/export.py --weights runs/detect/aquafina/weights/best.pt --out models/best.onnx
    python src/export.py --weights runs/detect/aquafina/weights/best.pt --out models/best.onnx --quantize
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export(weights: str, out_path: str, imgsz: int, quantize: bool):
    model = YOLO(weights)

    exported = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, out_path)
    print(f"ONNX model saved to {out_path}")

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("Install onnxruntime for quantization: pip install onnxruntime")
            return

        quant_path = str(Path(out_path).with_name(Path(out_path).stem + "_quant.onnx"))
        quantize_dynamic(out_path, quant_path, weight_type=QuantType.QUInt8)
        print(f"Quantized ONNX model saved to {quant_path} (smaller, faster in-browser)")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX for web deployment")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--out", default="models/best.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quantize", action="store_true", help="Also produce a quantized (uint8) ONNX model")
    args = parser.parse_args()

    export(args.weights, args.out, args.imgsz, args.quantize)


if __name__ == "__main__":
    main()
