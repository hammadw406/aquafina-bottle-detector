# Model Weights

Trained weights (`.pt`, `.onnx`) are **not committed to git** — they're binary and bloat the repo history fast. Instead:

- Attach `best.pt` and `best.onnx` to a **GitHub Release** for each trained version (e.g. `v1.0-model`), or
- Host them on **Hugging Face Hub** / **Google Drive** and link here, or
- Use **DVC** or **Git LFS** if you want them version-controlled alongside code.

| Version | Format | mAP@0.5 | Link |
|---|---|---|---|
| v0.1 | `.pt` | — | _TBD_ |
| v0.1 | `.onnx` | — | _TBD_ |

Update `web/app.js`'s `MODEL_PATH` constant to point at wherever you host `best.onnx` for the live demo.
