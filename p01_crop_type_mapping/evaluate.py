"""Shared metrics and plots so every model is reported the same way."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config as C


def summarize(y_true, y_pred, classes) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(classes)))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_per_class": {c: round(float(f), 4) for c, f in zip(classes, per_class)},
    }


def save_metrics(name: str, metrics: dict):
    """Merge into results/metrics.json under `name`, so all models share one file."""
    path = C.RESULTS / "metrics.json"
    C.RESULTS.mkdir(exist_ok=True)
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[name] = metrics
    path.write_text(json.dumps(allm, indent=2))


def plot_confusion(y_true, y_pred, classes, title: str, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
