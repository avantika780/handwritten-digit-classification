"""Finishes the last steps of run_all.py (model description, requirements.txt, README tables) without retraining."""
import os

import tensorflow as tf

import report
from config import MODELS

m = tf.keras.models.load_model(os.path.join(MODELS, "proposed_seed42.keras"))
with open(os.path.join(MODELS, "model_description.txt"), "w", encoding="utf-8") as f:
    f.write("Proposed model: CNN + BatchNorm + Squeeze-and-Excitation, trained with augmentation\n\n")
    m.summary(print_fn=lambda s, **k: f.write(s + "\n"))
report.write_requirements()
report.fill_readme()
print("Finished.")