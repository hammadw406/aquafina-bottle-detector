# Results Log

Update this after every training run. Keeping a running log (instead of overwriting) is what turns this into a credible experiment history for your internship writeup.

## Run history

| Run | Model | Epochs | Img size | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|---|---|---|---|---|---|---|---|---|
| v0-baseline | yolov8n | 100 | 640 | — | — | — | — | First run, no hard negatives |
| v1-hardneg | yolov8n | 100 | 640 | — | — | — | — | Added competitor-brand negatives |

## Error analysis (Stage 4)

- **False positives:** _(e.g. confuses with X brand under Y lighting)_
- **False negatives:** _(e.g. misses bottle when >60% occluded)_
- **Fix applied:** _(e.g. added N more hard-negative images of X, re-ran training)_

## Inference benchmarks (Stage 5)

| Format | Device | Avg inference time | Model size |
|---|---|---|---|
| PyTorch (.pt) | Kaggle GPU | — | — |
| ONNX (fp32) | Browser (CPU) | — | — |
| ONNX (quantized) | Browser (CPU) | — | — |

## Confusion matrix / PR curve

_(Embed images exported from Ultralytics' `runs/detect/train/` folder, e.g.)_

```md
![Confusion Matrix](../assets/confusion_matrix.png)
![PR Curve](../assets/PR_curve.png)
```
