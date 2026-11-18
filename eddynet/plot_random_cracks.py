from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data_step=0.3e-3"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "crack_profile_img"


def _save_plot(figure, data_dir):
    """Save a PNG named after the data folder, adding a suffix for existing names."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = data_dir.name.removeprefix("data_")
    for index in count():
        suffix = f"_{index}" if index else ""
        image_path = IMAGE_DIR / f"{name}{suffix}.png"
        try:
            image_file = image_path.open("xb")
        except FileExistsError:
            continue
        with image_file:
            figure.savefig(image_file, format="png", dpi=150, bbox_inches="tight")
        return image_path


def plot_random_cracks(root_dir=DATA_DIR, rows=5):
    """Save cracks beside real and imaginary responses, returning the image path.

    Each response panel has its own color scale and colorbar, centered at zero.
    """
    root_dir = Path(root_dir)
    frequency_dirs = sorted(
        root_dir.glob("*MHz"), key=lambda path: float(path.name.removesuffix("MHz"))
    )
    if len(frequency_dirs) != 3:
        raise ValueError("Expected three frequency folders in the data directory")
    crack_files = sorted((root_dir / "cracks").glob("*.npz"))
    if not 1 <= rows <= len(crack_files):
        raise ValueError(f"rows must be between 1 and {len(crack_files)}")
    selected = np.random.default_rng().choice(crack_files, size=rows, replace=False)
    response_maps = []
    for directory in frequency_dirs:
        maps = []
        for crack_file in selected:
            with np.load(directory / crack_file.name) as sample:
                maps.append(sample["delta_z"])
        response_maps.append(np.asarray(maps))

    figure, axes = plt.subplots(
        rows, 7, figsize=(23, 3 * rows), squeeze=False, constrained_layout=True
    )
    for row, crack_file in enumerate(selected):
        with np.load(crack_file) as sample:
            axes[row, 0].imshow(
                sample["crack"],
                cmap="gray_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        axes[row, 0].set_ylabel(f"Crack {crack_file.stem}")
    axes[0, 0].set_title("Crack profile")
    for component_index, (label, component) in enumerate(
        (("Real", np.real), ("Imaginary", np.imag))
    ):
        for frequency_index, directory in enumerate(frequency_dirs):
            column = 1 + component_index * len(frequency_dirs) + frequency_index
            maps = component(response_maps[frequency_index])
            for row, response in enumerate(maps):
                # Scale each image independently while keeping zero gray.
                limit = np.max(np.abs(response)) or 1.0
                response_image = axes[row, column].imshow(
                    response,
                    cmap="gray",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
                figure.colorbar(
                    response_image,
                    ax=axes[row, column],
                    fraction=0.046,
                    pad=0.04,
                    ticks=[-limit, 0, limit],
                    label="delta_z [ohm]",
                )
            axes[0, column].set_title(f"{directory.name} {label.lower()}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    try:
        return _save_plot(figure, root_dir)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    print(plot_random_cracks())
