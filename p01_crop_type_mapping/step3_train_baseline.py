"""Step 3: Random Forest baseline, evaluated with random vs. spatial cross-validation.

Run from the repo root:  python -m p01_crop_type_mapping.step3_train_baseline
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common.spatial_cv import make_folds

from . import config as C
from . import dataset
from .evaluate import plot_confusion, save_metrics, summarize


def cross_val_predict(X, y, folds):
    pred = np.empty_like(y)
    importances = []
    for train, test in folds:
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=C.SEED)
        rf.fit(X[train], y[train])
        pred[test] = rf.predict(X[test])
        importances.append(rf.feature_importances_)
    return pred, np.mean(importances, axis=0)


def plot_profiles(d, path):
    """Median NDVI per month for each class: the signal the models learn from."""
    ndvi = d["X"][..., d["channels"].index("NDVI")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(d["classes"]):
        s = ndvi[d["y"] == k]
        med = np.median(s, axis=0)
        q1, q3 = np.percentile(s, [25, 75], axis=0)
        ax.plot(range(len(med)), med, label=f"{name} (n={len(s)})", lw=2)
        ax.fill_between(range(len(med)), q1, q3, alpha=0.15)
    ax.set_xticks(range(len(d["months"])), [m[2:] for m in d["months"]], rotation=45)
    ax.set_ylabel("NDVI (median, IQR shaded)")
    ax.set_title("Seasonal NDVI profile by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_month_importance(imp, d, path):
    per_month = imp.reshape(len(d["months"]), len(d["channels"])).sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(per_month)), per_month, color="#3a7d44")
    ax.set_xticks(range(len(per_month)), [m[2:] for m in d["months"]], rotation=45)
    ax.set_ylabel("RF importance")
    ax.set_title("Which months matter most?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = dataset.load()
    X = d["X"].reshape(len(d["X"]), -1)  # RF sees months x channels as flat features
    y = d["y"]
    C.RESULTS.mkdir(exist_ok=True)
    print(f"{len(y)} fields, {X.shape[1]} features, {d['groups'].max() + 1} spatial blocks")

    for split, groups in [("random", None), ("spatial", d["groups"])]:
        folds = make_folds(y, groups, seed=C.SEED)
        pred, imp = cross_val_predict(X, y, folds)
        m = summarize(y, pred, d["classes"])
        save_metrics(f"random_forest/{split}_cv", m)
        print(f"RF {split:>7} CV  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
        if split == "spatial":
            plot_confusion(y, pred, d["classes"], "Random Forest (spatial CV)",
                           C.RESULTS / "confusion_rf_spatial.png")
            plot_month_importance(imp, d, C.RESULTS / "month_importance_rf.png")

    plot_profiles(d, C.RESULTS / "ndvi_profiles.png")
    print(f"Results written to {C.RESULTS}")


if __name__ == "__main__":
    main()
