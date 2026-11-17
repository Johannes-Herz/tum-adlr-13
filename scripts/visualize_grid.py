import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import yaml

from tactex.settings import get_settings


def save_grid_preview(
    npz_path: Path,
    output_path: Path | None = None,
    max_masks: int = 80,
    columns: int = 5,
    dpi: int = 150,
    circle_padding: int | None = None,
    circle_color: str = "tab:orange",
) -> Path:
    """Save the first occupancy grids in a .npz file as a tiled PNG preview.

    Args:
        npz_path: Input file containing a "masks" array of shape (N, H, W).
        output_path: Destination image. Defaults to the input path with a
            .png extension. An existing image is overwritten.
        max_masks: Maximum number of grids to display. Must be positive.
        columns: Maximum number of columns in the preview. Must be positive.
        dpi: Output image resolution in dots per inch. Must be positive.
        circle_padding: Circle padding in pixels. None reads padding from the
            adjacent metadata.json, falling back to 2 if unavailable.
        circle_color: Matplotlib color of the circle outline.

    Returns:
        Path to the saved preview image.

    Raises:
        ValueError: If a plotting parameter is invalid or the masks array
            is not three-dimensional and nonempty along every dimension.
        FileNotFoundError: If the input file does not exist.
        KeyError: If the input file has no "masks" entry.
    """
    for name, value in (("max_masks", max_masks), ("columns", columns), ("dpi", dpi)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    with np.load(npz_path) as data:
        masks = data["masks"]
    if masks.ndim != 3 or any(dimension == 0 for dimension in masks.shape):
        raise ValueError(
            f"Expected nonempty masks with shape (N, H, W), got {masks.shape}."
        )

    if circle_padding is None:
        metadata_path = npz_path.with_name("metadata.json")
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                circle_padding = json.load(file).get("padding", 2)
        else:
            circle_padding = 2
    height, width = masks.shape[1:]
    circle_radius = (min(height, width) - 1) / 2 - circle_padding
    if circle_padding < 0 or circle_radius <= 0:
        raise ValueError("Circle padding must be nonnegative and leave a positive radius.")

    if output_path is None:
        output_path = npz_path.with_suffix(".png")
    if output_path.resolve() == npz_path.resolve():
        raise ValueError("The preview output path must differ from the input path.")

    count = min(len(masks), max_masks)
    columns = min(count, columns)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False
    )
    try:
        for index, ax in enumerate(axes.flat):
            if index < count:
                ax.imshow(
                    masks[index], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                )
                ax.add_patch(
                    Circle(
                        ((width - 1) / 2, (height - 1) / 2),
                        circle_radius,
                        fill=False,
                        edgecolor=circle_color,
                        linewidth=1.2,
                    )
                )
                ax.set_title(f"Mask {index}")
            ax.axis("off")

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)
    return output_path


def main() -> None:
    """Load preview settings from YAML and save the configured image.

    Relative input and output paths are resolved against Settings.data_root.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "visualize_grid.yaml",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(
            "Config must contain a YAML mapping of parameter names to values."
        )

    npz_path = Path(config["npz_path"])
    output_path = config["output_path"]
    if not npz_path.is_absolute():
        npz_path = get_settings().data_root / npz_path
    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = get_settings().data_root / output_path

    saved_path = save_grid_preview(
        npz_path=npz_path,
        output_path=output_path,
        max_masks=config["max_masks"],
        columns=config["columns"],
        dpi=config["dpi"],
        circle_padding=config.get("circle_padding"),
        circle_color=config.get("circle_color", "tab:orange"),
    )
    print(f"Saved preview to {saved_path}")


if __name__ == "__main__":
    main()
