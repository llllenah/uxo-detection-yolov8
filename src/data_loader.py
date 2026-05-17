"""
Модуль завантаження та підготовки даних для задачі детекції UXO.

Реалізує ETL-процес: збір з кількох джерел -> валідація -> очищення ->
конвертація у єдиний формат YOLO -> формування сховища даних.
"""

import os
import json
import re
import shutil
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
import cv2


# Класи для детекції (відповідно до задачі)
CLASSES = {
    0: "uxo",          # нерозірваний боєприпас (міна, снаряд)
    1: "destruction",  # руйнування будівель/інфраструктури
    2: "vehicle",      # пошкоджена техніка
}

# Розширення файлів зображень
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_sources(sources_dir: Path) -> pd.DataFrame:
    """
    EXTRACT-етап: збираємо метадані про всі вхідні джерела даних.

    Очікувана структура:
        sources_dir/
            source_1_satellite/
                images/, annotations/
            source_2_aerial/
                images/, annotations/
            ...

    :param sources_dir: коренева тека з вихідними джерелами даних
    :return: DataFrame з метаінформацією про кожен файл
    """
    records = []
    for source in sorted(sources_dir.iterdir()):
        if not source.is_dir():
            continue
        img_dir = source / "images"
        ann_dir = source / "annotations"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            ann_path = ann_dir / (img_path.stem + ".xml")
            records.append({
                "source": source.name,
                "image_path": str(img_path),
                "annotation_path": str(ann_path) if ann_path.exists() else None,
                "image_name": img_path.name,
            })
    return pd.DataFrame(records)


def validate_image(img_path: Path) -> Dict:
    """
    Перевірка коректності зображення: відкривається, має валідні розміри,
    не пошкоджене.

    :param img_path: шлях до файлу зображення
    :return: словник з метриками якості
    """
    info = {
        "path": str(img_path),
        "valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "error": None,
    }
    try:
        with Image.open(img_path) as img:
            img.verify()
        # Повторно відкриваємо, бо verify() закриває файл
        with Image.open(img_path) as img:
            info["width"] = img.width
            info["height"] = img.height
            info["channels"] = len(img.getbands())
            info["valid"] = (img.width > 0 and img.height > 0)
    except Exception as e:
        info["error"] = str(e)
    return info


def parse_pascal_voc(xml_path: Path) -> List[Dict]:
    """
    Парсинг анотацій у форматі Pascal VOC (XML) -> список об'єктів.

    :param xml_path: шлях до XML-анотації
    :return: список словників з координатами та класами
    """
    if xml_path is None or not Path(xml_path).exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower().strip()
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Валідація: координати мають бути в межах зображення
        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "class": name,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
            "img_width": width, "img_height": height,
        })
    return objects


def voc_to_yolo(obj: Dict, class_to_id: Dict[str, int]) -> Tuple[int, float, float, float, float] | None:
    """
    TRANSFORM: конвертація box-у з Pascal VOC у формат YOLO
    (нормалізовані x_center, y_center, width, height).

    :param obj: словник з parse_pascal_voc
    :param class_to_id: відображення назви класу у числовий ID
    :return: кортеж (class_id, x, y, w, h) або None якщо клас невідомий
    """
    cls_name = obj["class"]
    if cls_name not in class_to_id:
        return None

    iw, ih = obj["img_width"], obj["img_height"]
    x_center = (obj["xmin"] + obj["xmax"]) / 2.0 / iw
    y_center = (obj["ymin"] + obj["ymax"]) / 2.0 / ih
    w = (obj["xmax"] - obj["xmin"]) / iw
    h = (obj["ymax"] - obj["ymin"]) / ih
    return (class_to_id[cls_name], x_center, y_center, w, h)


def build_data_warehouse(
    sources_dir: Path,
    output_dir: Path,
    class_to_id: Dict[str, int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    LOAD: повний ETL-процес — формування єдиного сховища даних
    у форматі YOLO з розбиттям train/val/test.

    Структура виходу:
        output_dir/
            images/{train,val,test}/
            labels/{train,val,test}/
            data.yaml

    :return: статистика сховища даних
    """
    if class_to_id is None:
        class_to_id = {v: k for k, v in CLASSES.items()}

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Сума пропорцій повинна = 1.0"

    # Створення структури теки
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. EXTRACT
    df = collect_sources(sources_dir)
    print(f"[EXTRACT] Знайдено {len(df)} файлів з {df['source'].nunique()} джерел")

    # 2. VALIDATE
    valid_records = []
    invalid_count = 0
    for _, row in df.iterrows():
        info = validate_image(Path(row["image_path"]))
        if info["valid"]:
            valid_records.append({**row.to_dict(), **info})
        else:
            invalid_count += 1
    print(f"[VALIDATE] Валідних: {len(valid_records)}, відкинуто: {invalid_count}")

    # 3. TRANSFORM + LOAD
    # Geographic split: group by base location (aerial_XXXXX) so that all
    # crops/versions of the same aerial photo land in ONE split only.
    # Splitting individual v-images randomly causes ~100% train/val/test leakage.
    random.seed(seed)

    groups: Dict[str, list] = defaultdict(list)
    for rec in valid_records:
        img_name = Path(rec["image_path"]).name
        match = re.match(r"(aerial_\d+)", img_name)
        base = match.group(1) if match else img_name
        groups[base].append(rec)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n = len(group_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = group_keys[:n_train]
    val_keys   = group_keys[n_train:n_train + n_val]
    test_keys  = group_keys[n_train + n_val:]

    splits = {
        "train": [rec for k in train_keys for rec in groups[k]],
        "val":   [rec for k in val_keys   for rec in groups[k]],
        "test":  [rec for k in test_keys  for rec in groups[k]],
    }

    stats = {"per_split": {}, "per_class": {c: 0 for c in class_to_id}}
    for split_name, records in splits.items():
        objs_count = 0
        for rec in records:
            src_img = Path(rec["image_path"])
            dst_img = output_dir / "images" / split_name / src_img.name
            shutil.copy2(src_img, dst_img)

            # Конвертація анотацій
            objs = parse_pascal_voc(rec["annotation_path"])
            yolo_lines = []
            for obj in objs:
                yolo = voc_to_yolo(obj, class_to_id)
                if yolo is not None:
                    cls_id, x, y, w, h = yolo
                    yolo_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    cls_name = [k for k, v in class_to_id.items() if v == cls_id][0]
                    stats["per_class"][cls_name] += 1

            label_path = output_dir / "labels" / split_name / (src_img.stem + ".txt")
            label_path.write_text("\n".join(yolo_lines))
            objs_count += len(yolo_lines)

        stats["per_split"][split_name] = {
            "images": len(records),
            "objects": objs_count,
        }
        print(f"[LOAD] {split_name}: {len(records)} зобр., {objs_count} об'єктів")

    # 4. data.yaml для YOLO
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_to_id)}\n"
        f"names: {[k for k, _ in sorted(class_to_id.items(), key=lambda x: x[1])]}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content)

    return stats


def get_dataset_summary(warehouse_dir: Path) -> pd.DataFrame:
    """
    Підрахунок описової статистики сховища даних.

    :param warehouse_dir: тека сховища
    :return: DataFrame з кількістю зображень/об'єктів за split та класами
    """
    rows = []
    for split in ["train", "val", "test"]:
        labels_dir = warehouse_dir / "labels" / split
        if not labels_dir.exists():
            continue
        cls_count = {c: 0 for c in CLASSES.values()}
        n_images = 0
        for lbl_file in labels_dir.glob("*.txt"):
            n_images += 1
            for line in lbl_file.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id = int(line.split()[0])
                cls_count[CLASSES[cls_id]] += 1
        rows.append({
            "split": split,
            "images": n_images,
            **cls_count,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ETL для UXO datasets")
    parser.add_argument("--sources", type=Path, required=True,
                        help="Тека з вихідними джерелами")
    parser.add_argument("--output", type=Path, required=True,
                        help="Вихідна тека сховища")
    args = parser.parse_args()

    stats = build_data_warehouse(args.sources, args.output)
    print("\n=== Статистика сховища ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
