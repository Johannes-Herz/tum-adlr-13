import math

import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import MetricTracker


def plot_history(
    train_tracker: MetricTracker,
    val_tracker: MetricTracker,
    cols: int = 3,
) -> None:
    """Plot train and validation curves for every tracked metric.

    Each metric gets its own subplot with one train and one val curve over
    the epochs. The best validation epoch, as reported by
    val_tracker.best_metric, is marked with a dotted vertical line and a
    dot, and its value and epoch are shown in the subplot title.

    Args:
        train_tracker: Tracker whose metric names start with "train/".
        val_tracker: Tracker whose metric names start with "val/". Every
            "val/<name>" must have a matching "train/<name>" with the same
            number of epochs.
        cols: Number of subplot columns. Rows are ceil(n_metrics / cols),
            and unused axes are hidden.

    Returns:
        None. The figure is displayed with plt.show().
    """
    train_hist = train_tracker.compute_all()  # {"train/iou_occ": tensor([...]), ...}
    val_hist = val_tracker.compute_all()  # {"val/iou_occ": tensor([...]), ...}
    best_val, best_step = val_tracker.best_metric(return_step=True)

    names: list[str] = [k.removeprefix("val/") for k in val_hist]
    rows: int = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False
    )

    for ax, name in zip(axes.flat, names):
        train_vals: np.ndarray = train_hist[f"train/{name}"].cpu().numpy()
        val_vals: np.ndarray = val_hist[f"val/{name}"].cpu().numpy()
        epochs: list[int] = range(len(val_vals))

        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals, label="val")

        step, value = best_step[f"val/{name}"], best_val[f"val/{name}"]
        ax.axvline(step, color="gray", linestyle=":", linewidth=1)
        ax.scatter([step], [value], color="black", zorder=3, s=20)

        ax.set_title(f"{name}  (best val {value:.3f} @ epoch {step})", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    for ax in axes.flat[len(names) :]:  # hide unused subplots
        ax.set_visible(False)

    axes.flat[0].legend()
    fig.tight_layout()
    plt.show()
