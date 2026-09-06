"""
Smoke tests for the inference pipeline's pure-logic pieces (no model file required
to run in CI — those need actual trained weights, which aren't committed to git).
"""
import numpy as np

from src.inference import letterbox, postprocess


def test_letterbox_output_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(img, new_shape=640)
    assert canvas.shape == (640, 640, 3)
    assert 0 < scale <= 1
    assert pad_x >= 0 and pad_y >= 0


def test_postprocess_filters_low_confidence():
    # Fake YOLO output: 1 box, 1 class, confidence below threshold
    fake_output = np.array([[[100, 100, 50, 50, 0.1]]]).transpose(0, 2, 1)
    boxes, scores, ids = postprocess(fake_output, scale=1.0, pad_x=0, pad_y=0, conf_thresh=0.4)
    assert len(boxes) == 0


def test_postprocess_keeps_high_confidence():
    fake_output = np.array([[[100, 100, 50, 50, 0.9]]]).transpose(0, 2, 1)
    boxes, scores, ids = postprocess(fake_output, scale=1.0, pad_x=0, pad_y=0, conf_thresh=0.4)
    assert len(boxes) == 1
    assert scores[0] == 0.9
