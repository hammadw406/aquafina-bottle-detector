# Architecture Notes

## Pipeline overview

```
Raw images ──► Roboflow labeling ──► YOLO-format dataset
     │                                      │
     ▼                                      ▼
data/raw/                          data/processed/ (data.yaml)
                                            │
                                            ▼
                              src/train.py (Ultralytics YOLOv8, on Kaggle GPU)
                                            │
                                            ▼
                              runs/detect/aquafina/weights/best.pt
                                            │
                                            ▼
                              src/export.py ──► models/best.onnx
                                            │
                                            ▼
                              web/app.js (ONNX Runtime Web, browser inference)
```

## Why YOLOv8n specifically

- Nano variant: ~6MB ONNX file after export — downloads fast in-browser.
- Anchor-free head simplifies the JS-side postprocessing (no anchor box math).
- Good accuracy/speed tradeoff for a single, visually distinctive class.

## Browser inference notes

- `onnxruntime-web` uses WASM backend by default (works everywhere, no WebGPU dependency).
- Preprocessing (letterbox resize, normalize, HWC→CHW) is reimplemented in `app.js` to
  exactly mirror Ultralytics' Python-side preprocessing — mismatches here are the #1
  source of "works in Python, garbage in browser" bugs.
- NMS is implemented manually in JS since ONNX Runtime Web doesn't ship a built-in NMS op
  for this export configuration.
