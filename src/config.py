"""Shared paths and hyperparameters."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
for _d in (RESULTS, FIGURES, MODELS):
    os.makedirs(_d, exist_ok=True)

SEED = 42

# Defaults = settings from the baseline notebook (+ ReduceLROnPlateau and a longer
# epoch budget so augmented models can converge). tune.py may overwrite lr,
# dropout and batch_size through results/best_hparams.json.
DEFAULT_HP = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 30,
    "patience": 5,
    "dropout": 0.3,
    "l2": 1e-3,
}


def load_hp():
    hp = dict(DEFAULT_HP)
    path = os.path.join(RESULTS, "best_hparams.json")
    if os.path.exists(path):
        with open(path) as f:
            hp.update(json.load(f))
    return hp
