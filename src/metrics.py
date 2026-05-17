"""
Метрики якості для задачі детекції об'єктів.

Реалізує IoU, Precision, Recall, F1, mAP, а також спеціалізовані метрики
для задачі розмінування: false negative rate (критично важлива метрика —
пропуск UXO може коштувати життя).
"""

from typing import List, Tuple, Dict
import numpy as np


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Обчислення IoU (Intersection over Union) для двох box-ів у форматі
    [x1, y1, x2, y2].

    :param box_a: координати першого box-у
    :param box_b: координати другого box-у
    :return: значення IoU у діапазоні [0, 1]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    preds: List[Dict],
    gts: List[Dict],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Зіставлення передбачень з ground truth для одного зображення.

    :param preds: список словників {"box": [x1,y1,x2,y2], "class": int, "score": float}
    :param gts: список словників {"box": [x1,y1,x2,y2], "class": int}
    :param iou_threshold: поріг IoU для зарахування TP
    :return: (TP, FP, FN)
    """
    matched_gt = set()
    tp = 0
    fp = 0

    # Сортування передбачень за впевненістю (від високої до низької)
    preds_sorted = sorted(preds, key=lambda p: p.get("score", 1.0), reverse=True)

    for pred in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            if pred["class"] != gt["class"]:
                continue
            iou = iou_xyxy(np.array(pred["box"]), np.array(gt["box"]))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Обчислення precision, recall, F1.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Average Precision у стилі COCO (інтерполяція 11 точок).

    :param precisions: масив значень precision
    :param recalls: масив значень recall
    :return: значення AP у діапазоні [0, 1]
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = max([p for p, r in zip(precisions, recalls) if r >= t], default=0.0)
        ap += p_at_t / 11.0
    return ap


def compute_map(
    all_preds: List[List[Dict]],
    all_gts: List[List[Dict]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Обчислення mAP для всього датасету.

    :param all_preds: список передбачень для кожного зображення
    :param all_gts: список ground truth для кожного зображення
    :param num_classes: кількість класів
    :param iou_threshold: поріг IoU
    :return: словник з AP для кожного класу та середнім mAP
    """
    aps = {}
    for cls_id in range(num_classes):
        # Збираємо всі передбачення цього класу
        cls_preds = []
        n_gt = 0
        for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
            n_gt += sum(1 for g in gts if g["class"] == cls_id)
            for p in preds:
                if p["class"] == cls_id:
                    cls_preds.append((img_idx, p))

        if n_gt == 0:
            aps[cls_id] = 0.0
            continue

        # Сортуємо за впевненістю
        cls_preds.sort(key=lambda x: x[1].get("score", 1.0), reverse=True)

        tp_arr = np.zeros(len(cls_preds))
        fp_arr = np.zeros(len(cls_preds))
        matched_gt_per_img = {}

        for i, (img_idx, pred) in enumerate(cls_preds):
            gts_img = [g for g in all_gts[img_idx] if g["class"] == cls_id]
            matched = matched_gt_per_img.setdefault(img_idx, set())

            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts_img):
                if j in matched:
                    continue
                iou = iou_xyxy(np.array(pred["box"]), np.array(gt["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx >= 0 and best_iou >= iou_threshold:
                tp_arr[i] = 1
                matched.add(best_idx)
            else:
                fp_arr[i] = 1

        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)
        recalls = (cum_tp / n_gt).tolist()
        precisions = (cum_tp / (cum_tp + cum_fp + 1e-9)).tolist()
        aps[cls_id] = compute_ap(precisions, recalls)

    return {"per_class": aps, "mAP": float(np.mean(list(aps.values())))}


def safety_metrics(tp: int, fp: int, fn: int) -> Dict:
    """
    Метрики безпеки специфічні для задачі розмінування.

    Особливо критичною є FNR — частка пропущених UXO. Хибне негативне
    спрацювання у цій задачі може коштувати життя сапера, тому recall
    розглядається як пріоритетна метрика порівняно з precision.

    :return: словник з метриками безпеки
    """
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
    fpr_relative = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Зважений показник, де recall вдвічі важливіший за precision (β = 2)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "F2": f2,
        "FNR": fnr,
        "FP_rate_relative": fpr_relative,
    }


if __name__ == "__main__":
    # Невеликий тест
    preds = [
        {"box": [10, 10, 100, 100], "class": 0, "score": 0.9},
        {"box": [200, 200, 300, 300], "class": 0, "score": 0.7},
    ]
    gts = [
        {"box": [12, 12, 102, 102], "class": 0},
        {"box": [400, 400, 500, 500], "class": 0},
    ]
    tp, fp, fn = match_predictions(preds, gts)
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(safety_metrics(tp, fp, fn))
