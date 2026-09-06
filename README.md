# 🍼 Aquafina Bottle Detector

Single-class real-time object detector that identifies **Aquafina water bottles** in images/video, trained with YOLOv8 and deployed as a fully client-side web app (ONNX Runtime Web — no backend, no server costs).

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in--progress-yellow.svg)

**🔗 Live demo:** _(add link once deployed, e.g. GitHub Pages / Vercel)_
**📊 Model card / metrics:** [docs/results.md](docs/results.md)
**📁 Dataset card:** [docs/dataset_card.md](docs/dataset_card.md)

---

## 🎯 Problem statement

Build a lightweight, single-class object detector that distinguishes **Aquafina bottles** from visually similar bottled-water competitors (Dasani, Nestlé Pure Life, Kirkland, generic PET bottles), robust to lighting, angle, occlusion, and background variation — then ship it as an in-browser demo.

## 🗺️ Project roadmap

This repo is built and committed **stage by stage** — each stage is its own branch/PR and its own commit history, so progress is fully auditable.

- [ ] **Stage 1 — Data collection** (`stage/01-data-collection`)
      Collect 150–300 positive + 50–100 hard-negative images. See [`src/data/scrape_images.py`](src/data/scrape_images.py).
- [ ] **Stage 2 — Labeling & dataset versioning** (`stage/02-labeling`)
      Annotate with Roboflow, export YOLO format, document in [docs/dataset_card.md](docs/dataset_card.md).
- [ ] **Stage 3 — Baseline training on Kaggle** (`stage/03-baseline-train`)
      Train YOLOv8n baseline. Notebook: [`notebooks/03_train_yolov8_kaggle.ipynb`](notebooks/03_train_yolov8_kaggle.ipynb).
- [ ] **Stage 4 — Evaluation & error analysis** (`stage/04-evaluation`)
      Confusion matrix, PR curve, hard-negative mining, retrain. Logged in [docs/results.md](docs/results.md).
- [ ] **Stage 5 — Export & optimization** (`stage/05-export`)
      Export to ONNX, quantize, benchmark inference speed. See [`src/export.py`](src/export.py).
- [ ] **Stage 6 — Web deployment** (`stage/06-deploy-web`)
      Browser inference with ONNX Runtime Web (upload + live webcam). See [`web/`](web/).
- [ ] **Stage 7 — CI/CD & polish** (`stage/07-ci-polish`)
      GitHub Actions for linting/tests, demo GIF, final write-up.

> Track these as GitHub **Issues** + a **Project board** (Backlog → In Progress → Review → Done) — internships love visible project management, not just code.

## 📂 Repo structure

```
aquafina-bottle-detector/
├── data/                  # raw/processed data (gitignored — use DVC or Roboflow link, not git, for images)
├── notebooks/             # Kaggle/Colab training & EDA notebooks
├── src/
│   ├── data/scrape_images.py
│   ├── train.py           # CLI training script (Ultralytics YOLOv8)
│   ├── export.py          # ONNX export + quantization
│   └── inference.py       # local Python inference/benchmark
├── web/                   # client-side deployment (HTML/JS + ONNX Runtime Web)
├── models/                # model weights are NOT committed — see models/README.md
├── docs/                  # dataset card, architecture notes, results log
├── tests/                 # smoke tests (pytest)
└── .github/workflows/     # CI: lint + tests on every PR
```

## ⚡ Quickstart

```bash
git clone https://github.com/<your-username>/aquafina-bottle-detector.git
cd aquafina-bottle-detector
pip install -r requirements.txt

# 1. Train (on Kaggle/Colab with GPU, or locally)
python src/train.py --data data/processed/data.yaml --epochs 100 --model yolov8n.pt

# 2. Export to ONNX for web
python src/export.py --weights runs/detect/train/weights/best.pt --out models/best.onnx

# 3. Serve the web demo locally
cd web && python -m http.server 8000
# open http://localhost:8000
```

## 🧠 Why these design choices

| Decision | Reasoning |
|---|---|
| YOLOv8**n** (nano) | Smallest model that still hits good mAP — keeps the ONNX file small enough for fast browser download/inference |
| Single class + hard negatives | Forces the model to learn brand-specific features (label design, cap color) instead of "any bottle" |
| Client-side ONNX Runtime Web | Zero backend cost, works offline after first load, privacy-friendly (image never leaves the browser) |
| Roboflow for labeling | Free tier, built-in augmentation, direct YOLO-format export |

## 📈 Results

See [docs/results.md](docs/results.md) for mAP@0.5, precision/recall curves, confusion matrix, and inference speed benchmarks (updated after each training run).

## 🙌 Acknowledgements

Built as part of an ML/CV internship project.

## 📄 License

MIT — see [LICENSE](LICENSE).
