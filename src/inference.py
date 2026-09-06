"""
Sanity-check the exported ONNX model locally before wiring it into the browser demo.

Usage:
    python src/inference.py --model models/best.onnx --image path/to/test.jpg --conf 0.4
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["aquafina"]


def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def preprocess(img, imgsz=640):
    canvas, scale, pad_x, pad_y = letterbox(img, imgsz)
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
    blob = blob.transpose(2, 0, 1)[None, ...]  # HWC->CHW, add batch dim
    return np.ascontiguousarray(blob), scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y, conf_thresh=0.4, iou_thresh=0.45):
    preds = output[0].T  # (num_boxes, 4 + num_classes)
    boxes, scores = preds[:, :4], preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_thresh
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    # xywh -> xyxy, undo letterbox
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), confidences.tolist(), conf_thresh, iou_thresh)
    indices = np.array(indices).flatten() if len(indices) else []

    return xyxy[indices], confidences[indices], class_ids[indices]


def main():
    parser = argparse.ArgumentParser(description="Run local ONNX inference sanity check")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = cv2.imread(args.image)
    blob, scale, pad_x, pad_y = preprocess(img, args.imgsz)

    start = time.time()
    output = session.run(None, {input_name: blob})[0]
    elapsed = (time.time() - start) * 1000

    boxes, confidences, class_ids = postprocess(output, scale, pad_x, pad_y, args.conf)
    print(f"Inference time: {elapsed:.1f} ms | Detections: {len(boxes)}")

    for box, conf, cid in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{CLASS_NAMES[cid]} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    out_path = "inference_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()
