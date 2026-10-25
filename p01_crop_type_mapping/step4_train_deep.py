"""Step 4: train a deep model from models.py with the same spatial CV as the baseline.

Run from the repo root:
    python -m p01_crop_type_mapping.step4_train_deep --model linear
    python -m p01_crop_type_mapping.step4_train_deep --model tempcnn --epochs 80
"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.spatial_cv import make_folds

from . import config as C
from . import dataset, models
from .evaluate import plot_confusion, save_metrics, summarize


def standardize(X_train, X_test):
    # Per-channel statistics from the training fold only, to avoid leakage
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std


def train_fold(name, X_train, y_train, X_test, n_classes, epochs, lr, device):
    torch.manual_seed(C.SEED)
    model = models.build(name, X_train.shape[2], X_train.shape[1], n_classes).to(device)

    # Weight classes by inverse frequency so the model doesn't ignore oilseeds
    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.tensor(len(y_train) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="linear", choices=list(models.MODELS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = dataset.load()
    X, y = d["X"], d["y"].astype(np.int64)
    pred = np.empty_like(y)
    folds = make_folds(y, d["groups"], seed=C.SEED)
    for k, (train, test) in enumerate(folds):
        Xtr, Xte = standardize(X[train], X[test])
        pred[test] = train_fold(args.model, Xtr, y[train], Xte, len(d["classes"]),
                                args.epochs, args.lr, device)
        print(f"fold {k + 1}/{len(folds)} done")

    m = summarize(y, pred, d["classes"])
    save_metrics(f"{args.model}/spatial_cv", m)
    plot_confusion(y, pred, d["classes"], f"{args.model} (spatial CV)",
                   C.RESULTS / f"confusion_{args.model}_spatial.png")
    print(f"{args.model} spatial CV  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
