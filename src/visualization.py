"""
Модуль порівняння моделей та візуалізації результатів.

Реалізує:
    - Порівняння кількох моделей за ключовими метриками (mAP, P, R, F1, FNR);
    - Побудову графіків (confusion matrix, PR-curve, порівняльні стовпчасті);
    - Експорт результатів у CSV для подальшого використання у звіті.
"""

from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


def comparison_table(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Будує зведену таблицю з метриками кількох моделей.

    :param model_results: {ім'я_моделі: {метрика: значення, ...}}
    :return: DataFrame для подальшого порівняння
    """
    rows = []
    for name, metrics in model_results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df


def plot_metrics_comparison(
    df: pd.DataFrame,
    metrics_to_plot: List[str] = None,
    save_path: Path = None,
    title: str = "Порівняння методів",
) -> plt.Figure:
    """
    Стовпчастий графік порівняння кількох метрик для кількох моделей.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [c for c in ["precision", "recall", "F1", "mAP_50", "mAP_50_95"]
                          if c in df.columns]

    n = len(metrics_to_plot)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))

    x = np.arange(len(df.index))
    width = 0.8 / n
    colors = sns.color_palette("Set2", n)

    for i, metric in enumerate(metrics_to_plot):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, df[metric].values, width, label=metric, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=15)
    ax.set_ylabel("Значення метрики")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = None,
    title: str = "Матриця невідповідностей",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Візуалізація матриці помилок класифікації.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Прогнозований клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_class_distribution(
    summary_df: pd.DataFrame,
    save_path: Path = None,
) -> plt.Figure:
    """
    Графік розподілу об'єктів за класами та split-ами сховища.
    """
    cls_columns = [c for c in summary_df.columns if c not in ("split", "images")]
    df_melted = summary_df.melt(id_vars=["split"], value_vars=cls_columns,
                                 var_name="клас", value_name="кількість")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_melted, x="split", y="кількість", hue="клас",
                ax=ax, palette="Set2")
    ax.set_title("Розподіл об'єктів за класами та вибірками")
    ax.set_xlabel("Вибірка")
    ax.set_ylabel("Кількість об'єктів")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_inference_speed(
    model_results: Dict[str, Dict],
    save_path: Path = None,
) -> plt.Figure:
    """
    Порівняння швидкості інференсу різних моделей.

    Два панелі: загальний час (inference домінує) + збільшений вид pre/postprocess.
    Швидкість критична для практичного використання у польових умовах.
    """
    rows = []
    for name, m in model_results.items():
        speed = m.get("speed_ms", {})
        if isinstance(speed, dict):
            rows.append({
                "модель": name,
                "preprocess": speed.get("preprocess", 0),
                "inference": speed.get("inference", 0),
                "postprocess": speed.get("postprocess", 0),
            })
    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("модель")
    pal = sns.color_palette("Set2", len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- ліва панель: загальний час (inference + pre + post) ---
    total = df.sum(axis=1)
    bars = axes[0].bar(total.index, total.values, color=pal, width=0.5, edgecolor="white")
    for bar, val in zip(bars, total.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + total.max() * 0.02,
            f"{val:.0f} мс",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )
    axes[0].set_ylabel("Час на одне зображення, мс")
    axes[0].set_title("Загальний час інференсу (CPU)")
    axes[0].set_ylim(0, total.max() * 1.18)
    axes[0].set_xticklabels(total.index, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    # --- права панель: збільшений pre/postprocess ---
    x = np.arange(len(df))
    w = 0.35
    b1 = axes[1].bar(x - w / 2, df["preprocess"],  w, label="preprocess",  color=pal[0])
    b2 = axes[1].bar(x + w / 2, df["postprocess"], w, label="postprocess", color=pal[2])
    for bars_grp in (b1, b2):
        for bar in bars_grp:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=10,
            )
    axes[1].set_ylabel("Час, мс")
    axes[1].set_title("Preprocess / Postprocess (збільшено)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df.index, rotation=15)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Швидкість обробки одного зображення (CPU)", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def export_results(
    model_results: Dict[str, Dict],
    output_dir: Path,
):
    """
    Експорт усіх результатів у CSV та JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = comparison_table(model_results)
    df.to_csv(output_dir / "comparison.csv", encoding="utf-8")
    with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Результати збережено: {output_dir}")


def visualize_predictions(
    image_path: Path,
    predictions: List[Dict],
    class_names: List[str],
    save_path: Path = None,
) -> plt.Figure:
    """
    Візуалізація передбачень моделі на одному зображенні.
    """
    from PIL import Image
    img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)

    palette = sns.color_palette("husl", len(class_names))
    for det in predictions:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class"]
        score = det.get("score", 1.0)
        color = palette[cls_id % len(palette)]

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{class_names[cls_id]}: {score:.2f}",
                color="white", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"))

    ax.axis("off")
    ax.set_title(f"Детекції: {len(predictions)} об'єктів")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Демонстрація на синтетичних даних
    demo = {
        "YOLOv8n": {"precision": 0.78, "recall": 0.82, "F1": 0.80, "mAP_50": 0.79},
        "YOLOv8s": {"precision": 0.83, "recall": 0.86, "F1": 0.84, "mAP_50": 0.84},
        "RT-DETR": {"precision": 0.85, "recall": 0.81, "F1": 0.83, "mAP_50": 0.82},
    }
    df = comparison_table(demo)
    print(df)
    plot_metrics_comparison(df, save_path=Path("demo_compare.png"))
