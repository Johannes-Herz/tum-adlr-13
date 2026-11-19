import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import nn

from tactex.sensing.raycasting import cast, sample, traverse
from tactex.typing import OccupancyGrid2D, TraversalGrid2D


@torch.no_grad()
def show(
    model: nn.Module,
    occupancy_grids: OccupancyGrid2D,
    n: int = 50,
    k: int = 3,
    snapshots: int = 5,
    snapshot_interval: int | None = None,
    threshold: float = 0.5,
    probabilities: bool = False,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Show how predictions improve as rays are added one at a time.

    Samples `n` rays per grid once, then for every ray count `1..n` casts the
    first rays, runs the model and computes the IoU against ground truth.

    Layout: per sample two rows (input traversal grid / thresholded
    prediction) at `snapshots` evenly spaced ray counts, plus a ground-truth
    column. A bottom panel plots IoU vs. number of rays for all samples.

    Args:
        model: Maps `(B, 3, H, W)` inputs to `(B, 1, H, W)` logits.
        occupancy_grids: Ground truth of shape `(B, H, W)` or `(B, 1, H, W)`.
        n: Maximum number of rays.
        k: Number of grids to use. Clipped to `B`.
        snapshots: Number of evenly spaced ray counts to visualize as images.
            Ignored if `snapshot_interval` is set.
        snapshot_interval: If set, visualize a snapshot every
            `snapshot_interval` added rays (e.g. 10 -> 10, 20, 30, ...).
            The IoU curve always covers every ray count.
        threshold: Probability threshold for binarizing predictions.
        probabilities: If True, show reconstructions as predicted occupancy
            probabilities (viridis, 0..1) instead of thresholded binary grids.
            IoU is always computed on the thresholded prediction.
        seed: Seed for ray sampling.
        device: Device for the model. Defaults to the model's device.

    Returns:
        IoU array of shape `(k, n)`, entry `[i, j]` is the IoU of sample i
        after `j + 1` rays.
    """
    gt: np.ndarray = np.asarray(occupancy_grids).astype(bool)
    if gt.ndim == 4:
        gt = gt[:, 0]
    k = min(k, gt.shape[0])
    gt = gt[:k]
    resolution: int = gt.shape[-1]

    if device is None:
        device = next(model.parameters()).device
    was_training: bool = model.training
    model.eval()

    rng = np.random.default_rng(seed)
    rays = sample(resolution=resolution, b=k, n=n, rng=rng)
    traversals = traverse(rays=rays, resolution=resolution)

    ray_counts = np.arange(1, n + 1)
    if snapshot_interval is not None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")
        snap_counts = np.arange(snapshot_interval, n + 1, snapshot_interval)
        if snap_counts.size == 0:
            snap_counts = np.array([n])
    else:
        snap_counts = np.unique(np.linspace(1, n, snapshots).round().astype(int))
    ious = np.zeros((k, n))
    snap_inputs: dict[int, TraversalGrid2D] = {}
    snap_preds: dict[int, np.ndarray] = {}

    for j, r in enumerate(ray_counts):
        grids: TraversalGrid2D = cast(
            occupancy_grids=gt, traversals=traversals, num_rays=np.full((k, 1), r)
        )
        x = torch.from_numpy(grids).to(device, torch.float32)
        probs = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        pred = probs >= threshold
        inter = (pred & gt).sum(axis=(1, 2))
        union = np.maximum((pred | gt).sum(axis=(1, 2)), 1)
        ious[:, j] = inter / union
        if r in snap_counts:
            snap_inputs[r] = grids
            snap_preds[r] = probs if probabilities else pred

    model.train(was_training)

    # --- Plot ---
    cols = len(snap_counts) + 1
    fig = plt.figure(figsize=(2.6 * cols, 2.6 * 2 * k + 3.5))
    gs = fig.add_gridspec(2 * k + 1, cols, height_ratios=[1] * (2 * k) + [1.3])
    input_cmap = ListedColormap(["#efefef", "#1f77b4", "#ff1a1a"])
    imshow_kw = dict(origin="lower", interpolation="nearest")
    pred_cmap = "viridis" if probabilities else "binary"

    for i in range(k):
        for c, r in enumerate(snap_counts):
            g = snap_inputs[r][i]
            state = np.zeros(g.shape[1:], dtype=np.int32)
            state[g[1].astype(bool)] = 1
            state[g[0].astype(bool)] = 2

            ax = fig.add_subplot(gs[2 * i, c])
            ax.imshow(state.T, cmap=input_cmap, vmin=0, vmax=2, **imshow_kw)
            ax.set_title(f"#{i} | {r} rays", fontsize=9)
            ax.axis("off")

            ax = fig.add_subplot(gs[2 * i + 1, c])
            ax.imshow(snap_preds[r][i].T, cmap=pred_cmap, vmin=0, vmax=1, **imshow_kw)
            ax.set_title(f"IoU {ious[i, r - 1]:.2f}", fontsize=9)
            ax.axis("off")

        ax = fig.add_subplot(gs[2 * i : 2 * i + 2, -1])
        ax.imshow(gt[i].T, cmap="binary", vmin=0, vmax=1, **imshow_kw)
        ax.set_title(f"Ground Truth #{i}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[-1, :])
    for i in range(k):
        ax.plot(ray_counts, ious[i], alpha=0.6, label=f"#{i}")
    ax.plot(ray_counts, ious.mean(axis=0), color="black", lw=2, label="mean")
    for r in snap_counts:
        ax.axvline(r, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Number of rays")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small", ncol=min(k + 1, 6))

    fig.tight_layout()
    plt.show()
    return ious
