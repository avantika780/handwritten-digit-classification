"""Training utilities. CLI: python src/train.py --config proposed --seed 42"""
import argparse
import os
import re
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers

import model as M
import preprocessing
from config import MODELS, RESULTS, SEED, load_hp

# name -> (batch-norm, SE attention, augmentation)
ABLATION = {
    "baseline":       (False, False, False),
    "+BN":            (True,  False, False),
    "+BN+SE":         (True,  True,  False),
    "proposed":       (True,  True,  True),   
    "proposed - BN":  (False, True,  True),
    "proposed - SE":  (True,  False, True),
    "proposed - Aug": (True,  True,  False),
}


def make_augmenter():
    kw = dict(fill_mode="constant", fill_value=0.0)
    return tf.keras.Sequential([
        layers.RandomRotation(10 / 360, **kw),      # +/-10 degrees
        layers.RandomTranslation(0.1, 0.1, **kw),   # +/-10 %
        layers.RandomZoom(0.1, **kw),               # +/-10 %
    ])


def build(name, hp):
    if name == "lenet5":
        return M.build_lenet5(), False
    if name == "vgg_small":
        return M.build_vgg_small(hp["dropout"]), False
    bn, se, aug = ABLATION[name]
    return M.build_cnn(bn, se, hp["dropout"], hp["l2"],
                    name=re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")), aug


def train(name, data, hp, seed=SEED, epochs=None, verbose=2):
    """Trains one model; early stopping on val_loss (best weights restored).
    verbose=2 prints one line per epoch so you can see progress.
    Returns (model, history dict, info dict)."""
    tf.keras.utils.set_random_seed(seed)
    model, aug = build(name, hp)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ds = tf.data.Dataset.from_tensor_slices((data["X_train"], data["y_train"]))
    ds = ds.shuffle(len(data["X_train"]), seed=seed).batch(hp["batch_size"])
    if aug:
        augmenter = make_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=hp["patience"],
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                        min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(ds, validation_data=(data["X_val"], data["y_val"]),
                    epochs=epochs or hp["epochs"], callbacks=cbs, verbose=verbose,
                    shuffle=False)  # dataset is already shuffled each epoch
    elapsed = time.time() - t0
    n = len(h.history["loss"])
    info = {"epochs_run": n, "train_time_s": round(elapsed, 1),
            "sec_per_epoch": round(elapsed / n, 2),
            "trainable_params": int(sum(int(tf.size(w)) for w in model.trainable_weights)),
            "total_params": int(model.count_params())}
    return model, h.history, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="proposed",
                    choices=list(ABLATION) + ["lenet5", "vgg_small"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="tiny random data, 2 epochs")
    a = ap.parse_args()
    data = preprocessing.load_data(a.smoke)
    hp = load_hp()
    m, hist, info = train(a.config, data, hp, a.seed, epochs=2 if a.smoke else None, verbose=2)
    tag = f"{a.config.replace(' ', '').replace('+', 'plus')}_seed{a.seed}"
    m.save(os.path.join(MODELS, f"{tag}.keras"))
    pd.DataFrame(hist).to_csv(os.path.join(RESULTS, f"history_{tag}.csv"), index=False)
    print(info)