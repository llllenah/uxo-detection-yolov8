<p align="center">
  <img src="assets/banner.png" alt="UXO Detection Banner" width="800"/>
</p>

<h1 align="center">🎯 UXO & Destruction Detection on Satellite Imagery</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-FF6F00.svg" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/RT--DETR-transformer-9B59B6.svg" alt="RT-DETR"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/dataset-6049_images-orange.svg" alt="Dataset"/>
</p>

<p align="center">
  <b>Computer vision pipeline for detecting unexploded ordnance (UXO), building destruction, and damaged military vehicles on satellite and aerial imagery of Southern Ukraine.</b>
</p>

---

## 🔍 Problem Statement

After armed conflicts, **unexploded ordnance (UXO)** poses an extreme threat to civilian safety. Manual mine clearance is slow, dangerous, and expensive. This project explores **automated detection** of UXO, structural destruction, and damaged vehicles using deep learning on satellite and aerial imagery — a potential force multiplier for humanitarian demining operations.

**Key design principle:** Missing a UXO is far more dangerous than a false alarm. Therefore, we optimize for **recall** and use **F2-score** as the primary metric.

## 📊 Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F2 | Inference (ms) |
|-------|---------|---------------|-----------|--------|----|----------------|
| YOLOv8n | 0.083 | 0.041 | 0.213 | 0.094 | 0.106 | **109** |
| **YOLOv8s** | **0.137** | **0.067** | 0.281 | **0.178** | **0.192** | 245 |
| RT-DETR | 0.022 | 0.011 | **0.397** | 0.036 | 0.044 | 1314 |

> **YOLOv8s** achieves the best recall–precision balance. RT-DETR shows highest precision but critically low recall — unsuitable for safety-critical applications.

## 🏗 Architecture

```
uxo-detection-yolov8/
├── README.md
├── requirements.txt
├── notebooks/
│   └── main.ipynb              # Full analytical pipeline
├── src/
│   ├── data_loader.py          # ETL: collection, validation, VOC→YOLO conversion
│   ├── metrics.py              # IoU, P, R, F1, F2, mAP, FNR
│   ├── train.py                # Training YOLOv8 / RT-DETR + hyperparameter search
│   └── visualization.py        # Comparison plots, confusion matrices, export
├── data/
│   ├── sources/                # Raw Pascal VOC annotations
│   │   ├── source_1_aerial/    # Sentinel-2 + UNOSAT (destruction)
│   │   ├── source_2_uav/       # Sentinel-2 + UNOSAT (UXO/craters)
│   │   └── source_3_terrestrial/ # UC Merced (vehicles, aerial)
│   └── warehouse/              # YOLO-format dataset (train/val/test)
└── runs/
    └── final_comparison/       # Model metrics (CSV + JSON)
```

## 🔬 Methodology

1. **ETL Pipeline** — Automated ingestion from 3 independent open data sources, Pascal VOC → YOLO format conversion, data integrity validation
2. **Data Leakage Prevention** — Location-aware split ensuring no aerial patch appears in both train and val/test
3. **Model Training** — YOLOv8n, YOLOv8s, RT-DETR with 20 epochs, augmentation, early stopping
4. **Hyperparameter Search** — Grid search over learning rate, batch size, confidence thresholds
5. **Safety Metrics** — F2-score (recall-weighted), False Negative Rate (FNR) for mine clearance evaluation

## 🗃 Data Sources

| # | Source | Type | Classes | Images |
|---|--------|------|---------|--------|
| 1 | Sentinel-2 + UNOSAT | Satellite (~10 m/px) | destruction, vehicle | ~2,000 |
| 2 | Sentinel-2 + UNOSAT | Satellite (~10 m/px) | uxo, destruction | ~2,000 |
| 3 | UC Merced + ESRI | Aerial (~0.3 m/px) | vehicle, destruction | ~2,000 |

**Total:** 6,049 images · 27,078 annotated objects · 70/15/15 train/val/test split

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/llllenah/uxo-detection-yolov8.git
cd uxo-detection-yolov8
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/main.ipynb
```

**Google Colab (recommended for GPU):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/llllenah/uxo-detection-yolov8/blob/main/notebooks/colab_train.ipynb)

## 🛠 Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `RT-DETR` · `PyTorch` · `OpenCV` · `SQLite` · `Pandas` · `Matplotlib` · `Seaborn`

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Olena Serhiienko** — [@llllenah](https://github.com/llllenah)  
Igor Sikorsky Kyiv Polytechnic Institute · Computer Engineering · IP-42
