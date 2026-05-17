# UXO and Destruction Detection on Satellite Imagery

Computer vision pipeline for detecting UXO-like objects, building destruction, and damaged vehicles in satellite and aerial imagery.

## Overview

This project explores automated detection workflows for safety-oriented geospatial imagery analysis. It combines dataset preparation, YOLO-format conversion, object-detection training, and evaluation with recall-weighted metrics.

Missing a potential UXO is more costly than a false alarm, so the evaluation emphasizes recall, F2-score, and false-negative rate alongside standard object-detection metrics.

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F2 | Inference (ms) |
|-------|---------|---------------|-----------|--------|----|----------------|
| YOLOv8n | 0.083 | 0.041 | 0.213 | 0.094 | 0.106 | 109 |
| YOLOv8s | 0.137 | 0.067 | 0.281 | 0.178 | 0.192 | 245 |
| RT-DETR | 0.022 | 0.011 | 0.397 | 0.036 | 0.044 | 1314 |

YOLOv8s produced the best recall-oriented balance in this experiment. RT-DETR had higher precision but low recall, which is less suitable for safety-critical screening.

## Methodology

- Data ingestion from multiple open imagery sources.
- Pascal VOC to YOLO annotation conversion.
- Dataset validation and train/validation/test splitting.
- YOLOv8n, YOLOv8s, and RT-DETR model comparison.
- Safety-oriented evaluation using F2-score and false-negative rate.
- Notebook reporting with plots, confusion matrices, and comparison tables.

## Repository Structure

```text
uxo-detection-yolov8/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── main.ipynb
│   └── colab_train.ipynb
└── src/
    ├── data_loader.py
    ├── metrics.py
    ├── train.py
    └── visualization.py
```

Large datasets, model weights, and training runs are intentionally excluded from the repository.

## Quick Start

```bash
git clone https://github.com/llllenah/uxo-detection-yolov8.git
cd uxo-detection-yolov8
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/main.ipynb
```

GPU training is available through the Colab notebook:

[Open `colab_train.ipynb` in Google Colab](https://colab.research.google.com/github/llllenah/uxo-detection-yolov8/blob/main/notebooks/colab_train.ipynb)

## Tech Stack

Python, YOLOv8, RT-DETR, PyTorch, OpenCV, SQLite, Pandas, Matplotlib, Seaborn.

## Disclaimer

This is a research and portfolio project. It is not an operational demining system and should not be used for field decisions without expert validation, high-quality data, and formal safety testing.
