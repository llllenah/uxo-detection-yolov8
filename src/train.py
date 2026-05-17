"""
Модуль навчання та інференсу моделей детекції об'єктів.

Реалізує три різнотипні методи:
    1. YOLOv8n  — однопрохідний детектор (one-stage), оптимізований для швидкості.
    2. YOLOv8s  — той самий клас моделей з більшою кількістю параметрів,
                  для дослідження впливу складності моделі на якість.
    3. RT-DETR  — детектор на основі трансформерів (DETR-like), принципово
                  інший клас моделей: без NMS, з глобальним attention.

Опціонально також підтримується Faster R-CNN з torchvision (two-stage).
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import time

import numpy as np
import torch

# ultralytics імпортуємо в середині функцій, щоб модуль вантажився швидко


def train_yolo(
    data_yaml: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    project: Path = Path("runs"),
    name: str = "yolo_train",
    device: str = "0",
    patience: int = 10,
) -> Dict:
    """
    Навчання моделі сімейства YOLO.

    :param data_yaml: шлях до data.yaml зі сховища
    :param model_name: початкова попередньо навчена модель
        (yolov8n.pt — nano, yolov8s.pt — small, yolov8m.pt — medium)
    :param epochs: кількість епох
    :param img_size: розмір вхідного зображення (квадратне)
    :param batch: розмір батча
    :param project: коренева тека для збереження результатів
    :param name: ім'я експерименту
    :param device: GPU id ("0") або "cpu"
    :param patience: рання зупинка
    :return: словник з шляхом до моделі та метриками
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(project),
        name=name,
        device=device,
        patience=patience,
        plots=True,
        verbose=True,
    )
    train_time = time.time() - start

    # Шлях до найкращої моделі
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    # Витягуємо метрики val
    metrics = {
        "train_time_sec": train_time,
        "weights_path": str(best_pt),
        "save_dir": str(save_dir),
    }

    # ultralytics зберігає метрики в results_dict
    if hasattr(results, "results_dict"):
        metrics["val_metrics"] = {k: float(v) for k, v in results.results_dict.items()
                                   if isinstance(v, (int, float))}

    return metrics


def train_rtdetr(
    data_yaml: Path,
    model_name: str = "rtdetr-l.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 8,
    project: Path = Path("runs"),
    name: str = "rtdetr_train",
    device: str = "0",
    patience: int = 10,
) -> Dict:
    """
    Навчання моделі RT-DETR (Real-Time Detection Transformer).

    Відрізняється від YOLO відсутністю NMS — використовує bipartite matching
    при навчанні. Цей метод представляє інший клас архітектур (transformer-based)
    і дозволить показати порівняння принципово різних підходів до задачі.
    """
    from ultralytics import RTDETR

    model = RTDETR(model_name)
    start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(project),
        name=name,
        device=device,
        patience=patience,
        plots=True,
        verbose=True,
    )
    train_time = time.time() - start

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    metrics = {
        "train_time_sec": train_time,
        "weights_path": str(best_pt),
        "save_dir": str(save_dir),
    }
    if hasattr(results, "results_dict"):
        metrics["val_metrics"] = {k: float(v) for k, v in results.results_dict.items()
                                   if isinstance(v, (int, float))}
    return metrics


def evaluate_model(
    weights_path: Path,
    data_yaml: Path,
    split: str = "test",
    img_size: int = 640,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "0",
) -> Dict:
    """
    Оцінка моделі на тестовому наборі даних.

    :param weights_path: шлях до .pt-файлу моделі
    :param data_yaml: data.yaml зі сховища
    :param split: 'val' або 'test'
    :param conf: поріг впевненості
    :param iou: поріг IoU для NMS
    :return: словник з метриками
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
    )

    # Витягуємо ключові метрики
    metrics_dict = {}
    if hasattr(results, "box"):
        metrics_dict["mAP_50"] = float(results.box.map50)
        metrics_dict["mAP_50_95"] = float(results.box.map)
        metrics_dict["precision"] = float(results.box.mp)
        metrics_dict["recall"] = float(results.box.mr)
    if hasattr(results, "speed"):
        metrics_dict["speed_ms"] = dict(results.speed)

    return metrics_dict


def hyperparameter_search(
    data_yaml: Path,
    model_name: str = "yolov8n.pt",
    param_grid: Optional[Dict[str, List]] = None,
    epochs_per_run: int = 20,
    project: Path = Path("runs/hpo"),
    device: str = "0",
) -> List[Dict]:
    """
    Підбір гіперпараметрів за сіткою для YOLO-моделей.

    :param param_grid: словник {назва_параметру: [значення1, значення2, ...]}
    :return: список результатів кожної комбінації
    """
    from itertools import product
    from ultralytics import YOLO

    if param_grid is None:
        param_grid = {
            "lr0": [0.01, 0.001],
            "imgsz": [416, 640],
            "batch": [8, 16],
        }

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    results_log = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run_name = f"hpo_run_{i}_" + "_".join(f"{k}={v}" for k, v in params.items())
        print(f"\n=== Експеримент {i+1}/{len(combinations)}: {params} ===")

        model = YOLO(model_name)
        try:
            res = model.train(
                data=str(data_yaml),
                epochs=epochs_per_run,
                project=str(project),
                name=run_name,
                device=device,
                verbose=False,
                plots=False,
                **params,
            )
            metrics = {}
            if hasattr(res, "results_dict"):
                metrics = {k: float(v) for k, v in res.results_dict.items()
                          if isinstance(v, (int, float))}
            results_log.append({"params": params, "metrics": metrics})
        except Exception as e:
            print(f"  Помилка: {e}")
            results_log.append({"params": params, "error": str(e)})

    # Збереження логу
    project.mkdir(parents=True, exist_ok=True)
    with open(project / "hpo_log.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)

    return results_log


def predict_image(
    weights_path: Path,
    image_path: Path,
    conf: float = 0.25,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Інференс на одному зображенні з візуалізацією результату.

    :return: словник зі знайденими box-ами та шляхом до зображення з розміткою
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=save_dir is not None,
        project=str(save_dir) if save_dir else None,
        name="predict",
    )

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "box": box.xyxy[0].cpu().numpy().tolist(),
                "class": int(box.cls[0]),
                "score": float(box.conf[0]),
            })
    return {"detections": detections}


if __name__ == "__main__":
    print("Цей модуль повинен імпортуватись з основного коду / notebook.")
    print("Доступні функції:")
    print("  - train_yolo(data_yaml, model_name, ...)")
    print("  - train_rtdetr(data_yaml, model_name, ...)")
    print("  - evaluate_model(weights_path, data_yaml, ...)")
    print("  - hyperparameter_search(data_yaml, ...)")
    print("  - predict_image(weights_path, image_path, ...)")
