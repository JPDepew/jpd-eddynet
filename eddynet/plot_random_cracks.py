from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def plot_random_cracks(root_dir=DATA_DIR, rows=5):
    """Plot random crack masks beside their response magnitudes at each frequency."""
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
    magnitudes = []
    for directory in frequency_dirs:
        maps = []
        for crack_file in selected:
            with np.load(directory / crack_file.name) as sample:
                maps.append(np.abs(sample["delta_z"]))
        magnitudes.append(maps)

    figure, axes = plt.subplots(
        rows, 4, figsize=(14, 3 * rows), squeeze=False, constrained_layout=True
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
        for column, maps in enumerate(magnitudes, start=1):
            axes[row, column].imshow(
                maps[row],
                cmap="viridis",
                vmin=0,
                vmax=max(response.max() for response in maps),
                interpolation="nearest",
            )
    axes[0, 0].set_title("Crack profile")
    for column, directory in enumerate(frequency_dirs, start=1):
        axes[0, column].set_title(f"{directory.name} response magnitude")
        figure.colorbar(
            axes[0, column].images[0], ax=axes[:, column].tolist(), label="|delta_z|"
        )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.show()
    return figure, axes


if __name__ == "__main__":
    plot_random_cracks()
