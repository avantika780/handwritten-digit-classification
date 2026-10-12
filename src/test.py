"""Evaluation utilities. CLI: python src/test.py --model models/proposed_seed42.keras"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

import preprocessing
from config import FIGURES, RESULTS


def evaluate_probs(y_true, prob):
    """All metrics required by the project brief. Returns (summary dict, per-class df, cm)."""
    prob = prob.astype("float64")
    prob = prob / prob.sum(axis=1, keepdims=True)
    y_pred = prob.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    spec = tn / (tn + fp)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(10),
                                                 zero_division=0)
    auc_c = [roc_auc_score((y_true == c).astype(int), prob[:, c]) for c in range(10)]
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                    zero_division=0)
    summary = {
        "accuracy": float((y_pred == y_true).mean()),
        "precision_macro": float(p.mean()), "precision_weighted": float(pw),
        "recall_macro": float(r.mean()), "recall_weighted": float(rw),
        "f1_macro": float(f.mean()), "f1_weighted": float(fw),
        "specificity_macro": float(spec.mean()),
        "roc_auc_macro_ovr": float(roc_auc_score(y_true, prob, multi_class="ovr",
                                                 average="macro", labels=list(range(10)))),
        "errors": int((y_pred != y_true).sum()),
    }
    per_class = pd.DataFrame({"Class": range(10), "Precision": p, "Recall": r, "F1": f,
                              "Specificity": spec, "ROC-AUC (OvR)": auc_c, "Support": s})
    return summary, per_class, cm


def plot_confusion(cm, title, stem):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
    ax[0].set(xlabel="Predicted label", ylabel="Actual label", title=f"{title} - counts")
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".3f", cmap="Blues", ax=ax[1])
    ax[1].set(xlabel="Predicted label", ylabel="Actual label", title=f"{title} - row-normalised")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{stem}.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = preprocessing.load_data(a.smoke)
    m = tf.keras.models.load_model(a.model)
    summ, pc, cm = evaluate_probs(d["y_test"], m.predict(d["X_test"], verbose=0))
    stem = os.path.splitext(os.path.basename(a.model))[0]
    pd.DataFrame([summ]).to_csv(os.path.join(RESULTS, f"test_metrics_{stem}.csv"), index=False)
    pc.to_csv(os.path.join(RESULTS, f"per_class_{stem}.csv"), index=False)
    plot_confusion(cm, stem, f"confusion_matrix_{stem}")
    print(pd.DataFrame([summ]).T)
