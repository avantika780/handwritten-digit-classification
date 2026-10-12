"""Data loading, stratified splitting, normalisation and dataset-analysis tables."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FIGURES, RESULTS, SEED


def load_data(smoke=False):
    """Returns dict with X_train/X_val/X_test (float32, N x 28 x 28 x 1) and y_*.
    smoke=True uses tiny random data purely to test that the code runs."""
    if smoke:
        rng = np.random.default_rng(0)
        X = rng.integers(0, 256, (600, 28, 28), dtype=np.uint8)
        y = np.arange(600) % 10
        Xt = rng.integers(0, 256, (200, 28, 28), dtype=np.uint8)
        yt = np.arange(200) % 10
    else:
        from tensorflow.keras.datasets import mnist
        (X, y), (Xt, yt) = mnist.load_data()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.10, random_state=SEED, stratify=y)

    def prep(a):  # scale 0-255 -> 0-1, add channel axis
        return (a.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

    return {"X_train": prep(X_tr), "y_train": y_tr,
            "X_val": prep(X_val), "y_val": y_val,
            "X_test": prep(Xt), "y_test": yt}


def dataset_tables(d):
    """Writes class distribution, split distribution, imbalance and preprocessing tables + figure."""
    classes = np.arange(10)
    rows = []
    for c in classes:
        tr, va, te = (int((d[k] == c).sum()) for k in ("y_train", "y_val", "y_test"))
        rows.append({"Class": int(c), "Train": tr, "Validation": va, "Test": te,
                     "Total": tr + va + te})
    cd = pd.DataFrame(rows)
    cd["% of dataset"] = (cd["Total"] / cd["Total"].sum() * 100).round(2)
    cd.to_csv(os.path.join(RESULTS, "dataset_class_distribution.csv"), index=False)

    n = {k: len(d[f"y_{k}"]) for k in ("train", "val", "test")}
    total = sum(n.values())
    sp = pd.DataFrame([
        {"Split": "Train", "Samples": n["train"], "Percentage": round(n["train"] / total * 100, 2)},
        {"Split": "Validation", "Samples": n["val"], "Percentage": round(n["val"] / total * 100, 2)},
        {"Split": "Test", "Samples": n["test"], "Percentage": round(n["test"] / total * 100, 2)},
        {"Split": "Total", "Samples": total, "Percentage": 100.0},
    ])
    sp.to_csv(os.path.join(RESULTS, "dataset_split_distribution.csv"), index=False)

    tot = cd["Total"]
    imb = pd.DataFrame([{
        "Largest class": int(cd.loc[tot.idxmax(), "Class"]), "Largest count": int(tot.max()),
        "Smallest class": int(cd.loc[tot.idxmin(), "Class"]), "Smallest count": int(tot.min()),
        "Imbalance ratio (max/min)": round(tot.max() / tot.min(), 3),
    }])
    imb.to_csv(os.path.join(RESULTS, "dataset_imbalance.csv"), index=False)

    pre = pd.DataFrame([
        {"Step": "Train/validation split", "Parameters": "10% of the 60,000 training images, stratified, random_state=42"},
        {"Step": "Pixel scaling", "Parameters": "uint8 [0,255] -> float32 [0,1] (divide by 255)"},
        {"Step": "Reshape", "Parameters": "(N, 28, 28) -> (N, 28, 28, 1)"},
        {"Step": "Augmentation (proposed model, training data only)",
         "Parameters": "rotation +/-10 deg, translation +/-10%, zoom +/-10%, black fill"},
    ])
    pre.to_csv(os.path.join(RESULTS, "dataset_preprocessing.csv"), index=False)

    plt.figure(figsize=(8, 4))
    w = 0.27
    for i, k in enumerate(["Train", "Validation", "Test"]):
        plt.bar(classes + (i - 1) * w, cd[k], w, label=k)
    plt.xticks(classes)
    plt.xlabel("Digit class")
    plt.ylabel("Samples")
    plt.title("MNIST class distribution per split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "class_distribution.png"), dpi=200)
    plt.close()
    return cd, sp, imb, pre
