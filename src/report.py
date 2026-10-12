"""Fills the auto-generated tables of README.md from results/*.csv and writes requirements.txt."""
import os
import re
import sys
from importlib import metadata

import pandas as pd

from config import ROOT, RESULTS

SECTIONS = {
    "dataset_class": ["dataset_class_distribution.csv"],
    "dataset_imbalance": ["dataset_imbalance.csv"],
    "dataset_split": ["dataset_split_distribution.csv"],
    "dataset_preprocessing": ["dataset_preprocessing.csv"],
    "hyperparameters": ["hyperparameters.csv"],
    "tuning": ["hyperparameter_tuning.csv"],
    "results_main": ["results.csv"],
    "results_per_class": ["per_class_metrics.csv"],
    "ablation": ["ablation.csv"],
    "comparison": ["comparison.csv"],
    "literature": ["literature_comparison.csv"],
    "training_analysis": ["training_analysis.csv"],
    "cost": ["computational_cost.csv"],
    "robustness": ["robustness.csv"],
}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def fill_readme():
    path = os.path.join(ROOT, "README.md")
    txt = open(path, encoding="utf-8").read()
    for key, files in SECTIONS.items():
        parts = []
        for f in files:
            p = os.path.join(RESULTS, f)
            parts.append(md_table(pd.read_csv(p)) if os.path.exists(p)
                         else f"_`{f}` not generated yet - run `python src/run_all.py`._")
        block = "\n\n".join(parts)
        pat = re.compile(rf"(<!--BEGIN:{key}-->)(.*?)(<!--END:{key}-->)", re.S)
        txt = pat.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", txt)
    open(path, "w", encoding="utf-8").write(txt)


def write_requirements():
    pkgs = ["tensorflow", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn"]
    v = sys.version_info
    lines = [f"# Python {v.major}.{v.minor}.{v.micro}  (interpreter version, not installable via pip)"]
    for p in pkgs:
        try:
            lines.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            lines.append(p)
    open(os.path.join(ROOT, "requirements.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_requirements()
    fill_readme()
