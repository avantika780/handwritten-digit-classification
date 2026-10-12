"""Hyperparameter search on the proposed model. Selection uses VALIDATION data only.
python src/tune.py [--quick] [--smoke]"""
import argparse
import itertools
import json
import os

import pandas as pd

import preprocessing
from config import DEFAULT_HP, RESULTS, SEED
from train import train

GRID = {"lr": [3e-4, 1e-3], "dropout": [0.3, 0.5], "batch_size": [64]}

def run(data, quick=False, smoke=False):
    epochs = 2 if smoke else (4 if quick else 12)
    rows = []
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        hp = dict(DEFAULT_HP, **dict(zip(keys, vals)))
        m, hist, info = train("proposed", data, hp, SEED, epochs=epochs)
        vloss, vacc = m.evaluate(data["X_val"], data["y_val"], verbose=0)
        rows.append({**dict(zip(keys, vals)), "val_accuracy": round(vacc, 5),
                    "val_loss": round(vloss, 5), "epochs_run": info["epochs_run"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
    df["selected"] = ["yes"] + [""] * (len(df) - 1)
    df.to_csv(os.path.join(RESULTS, "hyperparameter_tuning.csv"), index=False)
    best = df.iloc[0]
    with open(os.path.join(RESULTS, "best_hparams.json"), "w") as f:
        json.dump({"lr": float(best["lr"]), "dropout": float(best["dropout"]),
                "batch_size": int(best["batch_size"])}, f, indent=2)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(preprocessing.load_data(a.smoke), a.quick, a.smoke)
