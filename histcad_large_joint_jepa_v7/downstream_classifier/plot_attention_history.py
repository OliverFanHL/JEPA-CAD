"""Plot the saved 32-epoch training and validation trajectory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "attention_role_tokens_32ep"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output.resolve()
    with (output / "training_history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epoch = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_weighted_loss"]) for row in rows]
    val_loss = [float(row["validation_weighted_loss"]) for row in rows]
    val_macro = [float(row["validation_mean_macro_f1"]) for row in rows]
    form_macro = [float(row["validation_functional_form_macro_f1"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    axes[0].plot(epoch, train_loss, label="train weighted loss", color="#2563eb")
    axes[0].plot(epoch, val_loss, label="validation weighted loss", color="#d97706")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Optimization / overfitting")
    axes[0].legend(frameon=False)
    axes[1].plot(epoch, val_macro, label="mean macro-F1", color="#2563eb")
    axes[1].plot(epoch, form_macro, label="part-form macro-F1", color="#16a34a")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title="Validation task performance")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(1, max(epoch))
    fig.tight_layout()
    fig.savefig(output / "training_curves.png")
    plt.close(fig)
    print(output / "training_curves.png")


if __name__ == "__main__":
    main()
