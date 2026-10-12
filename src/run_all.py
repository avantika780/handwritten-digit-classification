"""Runs the complete pipeline and generates every table/figure in results/ and figures/.

  python src/run_all.py                 # full run (tuning + 3-seed ablation + comparison + robustness)
  python src/run_all.py --quick         # fast, low-quality run to check everything works
  python src/run_all.py --skip-tune     # reuse results/best_hparams.json (or defaults)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage

import preprocessing
import report
import tune
from config import FIGURES, MODELS, RESULTS, SEED, load_hp
from test import evaluate_probs, plot_confusion
from train import ABLATION, train

PRETTY = {"baseline": "Baseline CNN", "proposed": "Proposed (BN + SE + Aug)",
        "lenet5": "LeNet-5", "vgg_small": "VGG-style CNN"}
LITERATURE = [
    ("LeNet-5", "LeCun et al., Proc. IEEE, 1998", 99.05, "single CNN, no augmentation"),
    ("Multi-column DNN", "Ciresan et al., CVPR, 2012", 99.77, "committee of CNNs, augmentation"),
    ("DropConnect", "Wan et al., ICML, 2013", 99.79, "ensemble of 5 networks"),
    ("CapsNet", "Sabour et al., NeurIPS, 2017", 99.75, "capsule network, shift augmentation"),
]


def predict(m, X):
    return m.predict(X, batch_size=512, verbose=0)


def pct(x):
    return round(100 * x, 2)


def save(df, name):
    df.to_csv(os.path.join(RESULTS, name), index=False)


def curves(hist, title, stem):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, lab in ((ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")):
        a.plot(hist[k], label=f"Training {lab.lower()}")
        a.plot(hist["val_" + k], label=f"Validation {lab.lower()}")
        a.set(xlabel="Epoch", ylabel=lab, title=f"{title}: {lab}")
        a.grid(True)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{stem}.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    t_start = time.time()

    data = preprocessing.load_data(a.smoke)
    preprocessing.dataset_tables(data)
    if not a.skip_tune:
        tune.run(data, a.quick, a.smoke)
    hp = load_hp()
    seeds = [SEED] if a.smoke else [SEED + i for i in range(a.seeds)]
    ep = 2 if a.smoke else (3 if a.quick else None)
    Xt, yt = data["X_test"], data["y_test"]

    # ---------- 1. ablation (multiple seeds) ----------
    runs, kept = [], {}
    for name in ABLATION:
        for seed in seeds:
            m, h, info = train(name, data, hp, seed, epochs=ep)
            s, _, _ = evaluate_probs(yt, predict(m, Xt))
            runs.append({"Experiment": name, "seed": seed, "accuracy": s["accuracy"],
                         "f1_macro": s["f1_macro"], "params": info["total_params"]})
            print(name, seed, round(s["accuracy"], 4), flush=True)
            if seed == SEED and name in ("baseline", "proposed"):
                kept[name] = (m, h, info)
                m.save(os.path.join(MODELS, f"{name}_seed{SEED}.keras"))
            else:
                tf.keras.backend.clear_session()
    runs = pd.DataFrame(runs)
    save(runs, "ablation_runs.csv")
    rows = []
    for name, (bn, se, aug) in ABLATION.items():
        r = runs[runs.Experiment == name]
        sd = lambda c: r[c].std(ddof=1) if len(r) > 1 else 0.0
        rows.append({"Experiment": name,
                     "BatchNorm": "✓" if bn else "✗", "SE attention": "✓" if se else "✗",
                     "Augmentation": "✓" if aug else "✗",
                     "F1-macro (%)": f"{100*r.f1_macro.mean():.2f} ± {100*sd('f1_macro'):.2f}",
                     "Accuracy (%)": f"{100*r.accuracy.mean():.2f} ± {100*sd('accuracy'):.2f}",
                     "Params": int(r.params.iloc[0]), "Seeds": len(r)})
    save(pd.DataFrame(rows), "ablation.csv")

    # ---------- 2. full evaluation of the proposed model ----------
    res, prob_store = [], {}
    for key in ("baseline", "proposed"):
        m = kept[key][0]
        prob = predict(m, Xt)
        prob_store[key] = prob
        s, pc, cm = evaluate_probs(yt, prob)
        loss = float(m.evaluate(Xt, yt, verbose=0)[0])
        res.append({"Model": PRETTY[key], "Accuracy": pct(s["accuracy"]),
                    "Precision (macro)": pct(s["precision_macro"]),
                    "Precision (weighted)": pct(s["precision_weighted"]),
                    "Recall (macro)": pct(s["recall_macro"]),
                    "Recall (weighted)": pct(s["recall_weighted"]),
                    "F1 (macro)": pct(s["f1_macro"]), "F1 (weighted)": pct(s["f1_weighted"]),
                    "Specificity (macro)": pct(s["specificity_macro"]),
                    "ROC-AUC (macro OvR)": round(s["roc_auc_macro_ovr"], 5),
                    "Test loss": round(loss, 4), "Misclassified": s["errors"]})
        plot_confusion(cm, PRETTY[key], f"confusion_matrix_{key}")
        if key == "proposed":
            pc = pc.round(4)
            save(pc, "per_class_metrics.csv")
            plot_confusion(cm, PRETTY[key], "confusion_matrix")
            pd.DataFrame(cm).to_csv(os.path.join(RESULTS, "confusion_matrix.csv"), index=False)
    save(pd.DataFrame(res), "results.csv")

    # ---------- 3. training analysis + cost ----------
    ta, cost = [], []
    for key in ("baseline", "proposed"):
        _, h, info = kept[key]
        b = int(np.argmin(h["val_loss"]))
        ta.append({"Model": PRETTY[key], "Epochs run": info["epochs_run"],
                   "Best epoch (min val loss)": b + 1,
                   "Train acc @best (%)": pct(h["accuracy"][b]),
                   "Val acc @best (%)": pct(h["val_accuracy"][b]),
                   "Train-val acc gap (pp)": round(100 * (h["accuracy"][b] - h["val_accuracy"][b]), 2),
                   "Min val loss": round(min(h["val_loss"]), 4)})
        curves(h, PRETTY[key], "training_curve" if key == "proposed" else "training_curve_baseline")
    save(pd.DataFrame(ta), "training_analysis.csv")
    plt.figure(figsize=(7, 4))
    for key in ("baseline", "proposed"):
        plt.plot(kept[key][1]["val_accuracy"], label=PRETTY[key])
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy"); plt.grid(True); plt.legend()
    plt.title("Convergence: validation accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "convergence_comparison.png"), dpi=200)
    plt.close()

    def infer_ms(m):
        predict(m, Xt[:512])  # warm-up
        t = time.time(); predict(m, Xt)
        return round(1000 * (time.time() - t) / len(Xt), 4)

    for key in ("baseline", "proposed"):
        m, _, info = kept[key]
        cost.append({"Model": PRETTY[key], "Trainable params": info["trainable_params"],
                     "Total params": info["total_params"], "Epochs run": info["epochs_run"],
                     "Train time (s)": info["train_time_s"], "Sec/epoch": info["sec_per_epoch"],
                     "Inference (ms/image)": infer_ms(m)})

    # ---------- 4. comparison (same split, same hyperparameters, same test set) ----------
    comp = []
    other = {}
    for key in ("lenet5", "vgg_small"):
        m, h, info = train(key, data, hp, SEED, epochs=ep)
        other[key] = (m, info)
        cost.append({"Model": PRETTY[key], "Trainable params": info["trainable_params"],
                     "Total params": info["total_params"], "Epochs run": info["epochs_run"],
                     "Train time (s)": info["train_time_s"], "Sec/epoch": info["sec_per_epoch"],
                     "Inference (ms/image)": infer_ms(m)})
    save(pd.DataFrame(cost), "computational_cost.csv")
    order = [("lenet5", other["lenet5"][0], other["lenet5"][1]["total_params"]),
             ("vgg_small", other["vgg_small"][0], other["vgg_small"][1]["total_params"]),
             ("baseline", kept["baseline"][0], kept["baseline"][2]["total_params"]),
             ("proposed", kept["proposed"][0], kept["proposed"][2]["total_params"])]
    for key, m, npar in order:
        s, _, _ = evaluate_probs(yt, predict(m, Xt))
        comp.append({"Model": PRETTY[key], "Accuracy": pct(s["accuracy"]),
                     "Precision": pct(s["precision_macro"]), "Recall": pct(s["recall_macro"]),
                     "F1": pct(s["f1_macro"]), "AUC": round(s["roc_auc_macro_ovr"], 5),
                     "Parameters": npar})
    save(pd.DataFrame(comp), "comparison.csv")
    lit = [{"Method": n, "Reference": ref, "Reported accuracy (%)": acc, "Notes": note}
           for n, ref, acc, note in LITERATURE]
    lit.append({"Method": "Proposed (this work)", "Reference": "-",
                "Reported accuracy (%)": comp[-1]["Accuracy"], "Notes": "single model, no ensemble"})
    save(pd.DataFrame(lit), "literature_comparison.csv")

    # ---------- 5. robustness (innovation evidence) ----------
    rng = np.random.default_rng(0)
    tests = [("Clean", 0, Xt)]
    for sg in (0.1, 0.2, 0.3):
        tests.append(("Gaussian noise (sigma)", sg,
                      np.clip(Xt + rng.normal(0, sg, Xt.shape), 0, 1).astype("float32")))
    for ang in (10, 20, 30):
        tests.append(("Rotation (deg)", ang, ndimage.rotate(Xt, ang, axes=(1, 2),
                                                           reshape=False, order=1)))
    for px in (2, 4):
        tests.append(("Horizontal shift (px)", px, ndimage.shift(Xt, (0, 0, px, 0), order=0)))
    rob = []
    for kind, lvl, Xp in tests:
        row = {"Perturbation": kind, "Level": lvl}
        for key in ("baseline", "proposed"):
            row[f"{PRETTY[key]} acc (%)"] = pct(
                (predict(kept[key][0], Xp).argmax(1) == yt).mean())
        rob.append(row)
    rob = pd.DataFrame(rob)
    save(rob, "robustness.csv")
    plt.figure(figsize=(8, 4))
    labels = [f"{r.Perturbation}\n{r.Level}" for r in rob.itertuples()]
    x = np.arange(len(rob))
    for i, key in enumerate(("baseline", "proposed")):
        plt.bar(x + (i - 0.5) * 0.4, rob[f"{PRETTY[key]} acc (%)"], 0.4, label=PRETTY[key])
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Test accuracy (%)"); plt.legend(); plt.title("Robustness to perturbations")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "robustness.png"), dpi=200)
    plt.close()

    # ---------- 6. bookkeeping ----------
    sec = "ReduceLROnPlateau (monitor=val_loss, factor=0.5, patience=2, min_lr=1e-5)"
    hp_rows = [("Learning rate", hp["lr"]), ("Batch size", hp["batch_size"]),
               ("Number of epochs", f"max {hp['epochs']} (early stopping, patience {hp['patience']}; "
                                    f"ran {kept['proposed'][2]['epochs_run']} for proposed)"),
               ("Optimizer", "Adam"), ("Loss function", "Sparse categorical cross-entropy"),
               ("Scheduler", sec), ("Dropout", hp["dropout"]),
               ("Weight decay", f"L2 = {hp['l2']} on the Dense(128) kernel"),
               ("Other parameters", "Conv filters 32/64 (3x3), Dense 128, SE ratio 8, "
                                    f"augmentation ±10° / ±10% shift / ±10% zoom, seeds {seeds}")]
    save(pd.DataFrame(hp_rows, columns=["Hyperparameter", "Value"]), "hyperparameters.csv")
    with open(os.path.join(MODELS, "model_description.txt"), "w") as f:
        f.write("Proposed model: CNN + BatchNorm + Squeeze-and-Excitation, trained with augmentation\n\n")
        kept["proposed"][0].summary(print_fn=lambda s, **k: f.write(s + "\n"))
    report.write_requirements()
    report.fill_readme()
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
